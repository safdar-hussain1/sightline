import importlib.util
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from sightline.cli import main
from conftest import make_drive

from sightline.models import SHIPPED_LAGS

REPO = Path(__file__).resolve().parents[1]

ARGS = lambda tmp, *extra: ["train", "--drive", "drivetest", "--data-dir", str(tmp / "data"),
                            "--runs-dir", str(tmp / "runs"), "--epochs", "1", "--device", "cpu",
                            "--block", "100", *extra]

@pytest.fixture
def drive(tmp_path):
    make_drive(tmp_path / "data", "drivetest", n=1600)
    return tmp_path

@pytest.fixture
def short_drive(tmp_path):
    """700 frames: still every label of a 100-frame blocked split, a third of the training time."""
    make_drive(tmp_path / "data", "drivetest", n=700)
    return tmp_path

def test_train_writes_a_complete_run(drive):
    assert main(ARGS(drive, "--model", "sightlinenet", "--protocol", "blocked", "--firewall", "20")) == 0
    run = drive / "runs" / "drivetest_sightlinenet_blocked_s0"
    h = json.loads((run / "history.json").read_text())
    assert h["config"]["frame_lags"] == list(SHIPPED_LAGS)
    assert h["config"]["notes"]["block"] == 100 and h["config"]["notes"]["firewall"] == 20
    assert h["config"]["augment"] == "strong" and h["config"]["notes"]["protocol"] == "blocked"
    for f in ("best.pt", "train_idx.npy", "val_idx.npy", "test_idx.npy"):
        assert (run / f).is_file()

def test_train_refuses_to_overwrite_a_run(drive, capsys):
    args = ARGS(drive, "--model", "pilotnet", "--protocol", "leaky")
    assert main(args) == 0
    assert main(args) == 2
    assert "already exists" in capsys.readouterr().err
    assert main(args + ["--overwrite"]) == 0

def test_saved_split_is_the_protocol_split(drive):
    from sightline.data.splits import make_split
    main(ARGS(drive, "--model", "pilotnet", "--protocol", "blocked", "--firewall", "20", "--seed", "1"))
    run = drive / "runs" / "drivetest_pilotnet_blocked_s1"
    ref = make_split("blocked", 1600, seed=1, firewall=20, block=100)
    np.testing.assert_array_equal(np.load(run / "test_idx.npy"), ref.test)


# --- what a run records ---------------------------------------------------------------

def test_history_records_the_whole_config_and_where_the_code_came_from(short_drive):
    from sightline.data.labels import load_labels, repair_dropouts
    from sightline.train import code_version

    drive = short_drive
    assert main(ARGS(drive, "--model", "pilotnet", "--protocol", "forward", "--firewall", "20",
                     "--patience", "3", "--seed", "2")) == 0
    h = json.loads((drive / "runs" / "drivetest_pilotnet_forward_s2" / "history.json").read_text())
    assert set(h) == {"config", "code_version", "device", "n_params", "best_val_mae_deg",
                      "best_epoch", "history"}
    c = h["config"]
    assert set(c) == {"model", "epochs", "batch_size", "lr", "weight_decay", "seed", "augment",
                      "dropout", "patience", "frame_lags", "notes"}
    assert (c["model"], c["epochs"], c["seed"], c["patience"], c["frame_lags"]) == ("pilotnet", 1, 2, 3, [])
    repaired = repair_dropouts(load_labels(drive / "data" / "drivetest")).repaired_indices.size
    assert c["notes"] == {"drive": "drivetest", "protocol": "forward", "firewall": 20, "block": 100,
                          "repaired_labels": int(repaired)}
    assert h["device"] == "cpu" and h["code_version"] == code_version()


def _git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
                           "-c", "commit.gpgsign=false", *args],
                          cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def test_code_version_names_the_commit_and_marks_uncommitted_source(tmp_path):
    from sightline.train import code_version

    repo = tmp_path / "checkout"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "m.py").write_text("x = 1\n")
    (repo / "notes.txt").write_text("a\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "--no-verify", "-m", "start")
    head = _git(repo, "rev-parse", "HEAD")
    assert code_version(repo) == head
    (repo / "notes.txt").write_text("b\n")
    assert code_version(repo) == head  # a change outside src/ is not a code change
    (repo / "src" / "m.py").write_text("x = 2\n")
    assert code_version(repo) == head + "+dirty"
    (repo / "src" / "m.py").write_text("x = 1\n")
    assert code_version(repo) == head
    (repo / "src" / "new.py").write_text("")
    assert code_version(repo) == head + "+dirty"  # so does an untracked source file


def test_code_version_is_none_outside_a_git_checkout(tmp_path):
    from sightline.train import code_version

    assert code_version(tmp_path) is None


def test_code_version_defaults_to_the_checkout_holding_the_package():
    from sightline.train import code_version

    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.strip()
    assert code_version() in (head, head + "+dirty")


# --- clean failures -------------------------------------------------------------------

def test_every_registered_model_is_a_train_choice(monkeypatch):
    from sightline.cli import build_parser
    from sightline.models import MODEL_SPECS, ModelSpec

    monkeypatch.setitem(MODEL_SPECS, "toynet", ModelSpec("sightlinenet", (), "none", 0.0))
    for name in ("pilotnet", "sightlinenet", "sightlinenet-1f", "toynet"):
        assert build_parser().parse_args(["train", "--drive", "d", "--model", name]).model == name
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--drive", "d", "--model", "nonet"])


def _one_error_line(err):
    assert "Traceback" not in err
    lines = [line for line in err.splitlines() if line.startswith("error: ")]
    assert len(lines) == 1 and err.rstrip().endswith(lines[0]), err
    return lines[0]


@pytest.mark.parametrize("extra, message", [
    (["--protocol", "blocked", "--block", "1500"], "n=700 too small for the blocked split"),
    (["--protocol", "forward", "--firewall", "200"], "n=700 too small"),
    (["--protocol", "leaky", "--device", "cuda"], "device 'cuda' is not available"),
])
def test_unusable_arguments_are_one_line_errors_that_leave_no_run(short_drive, capsys, monkeypatch, extra, message):
    import torch

    drive = short_drive
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert main(ARGS(drive, "--model", "pilotnet", *extra)) == 2
    assert message in _one_error_line(capsys.readouterr().err)
    assert not (drive / "runs").exists() or not any(
        p.name.startswith("drivetest_") for p in (drive / "runs").iterdir())


@pytest.mark.parametrize("value", ["../drivetest", "data/drivetest", ".."])
def test_drive_must_be_a_name_not_a_path(tmp_path, capsys, value):
    argv = ARGS(tmp_path, "--model", "pilotnet")
    argv[argv.index("drivetest")] = value
    assert main(argv) == 2
    assert "drive name" in _one_error_line(capsys.readouterr().err)


@pytest.mark.parametrize("flag, value", [("--epochs", "0"), ("--patience", "-1"), ("--seed", "-1"),
                                         ("--firewall", "-1"), ("--block", "0")])
def test_out_of_range_numbers_are_rejected_by_the_parser(tmp_path, capsys, flag, value):
    with pytest.raises(SystemExit) as exc:
        main(ARGS(tmp_path, "--model", "pilotnet", flag, value))
    assert exc.value.code == 2
    assert f"argument {flag}: must be at least" in capsys.readouterr().err


def test_overwrite_discards_the_old_run_before_training(short_drive, monkeypatch):
    import sightline.train

    drive = short_drive
    run = drive / "runs" / "drivetest_pilotnet_leaky_s0"
    run.mkdir(parents=True)
    for name in ("history.json", "best.pt", "train_idx.npy", "val_idx.npy", "test_idx.npy"):
        (run / name).write_text("from the earlier run")

    def crash(*a, **k):
        raise RuntimeError("training crashed")

    monkeypatch.setattr(sightline.train, "train_model", crash)
    with pytest.raises(RuntimeError, match="training crashed"):
        main(ARGS(drive, "--model", "pilotnet", "--protocol", "leaky", "--overwrite"))
    assert not (run / "history.json").exists() and not (run / "best.pt").exists()
    assert (run / "test_idx.npy").read_bytes() != b"from the earlier run"


def test_a_run_that_never_scores_a_finite_validation_mae_is_not_finished(short_drive, monkeypatch):
    import sightline.train

    drive = short_drive
    monkeypatch.setattr(sightline.train, "predict",
                        lambda model, cache, indices, device, batch_size=256: np.full(len(indices), np.nan))
    with pytest.raises(RuntimeError, match="no finite validation MAE"):
        main(ARGS(drive, "--model", "pilotnet", "--protocol", "leaky"))
    run = drive / "runs" / "drivetest_pilotnet_leaky_s0"
    assert not (run / "history.json").exists() and not (run / "best.pt").exists()


# --- full cosine schedule ---------------------------------------------------------------

def test_train_defaults_are_the_full_twenty_epoch_schedule_without_early_stopping():
    from sightline.cli import build_parser
    from sightline.train import TrainConfig

    args = build_parser().parse_args(["train", "--drive", "d"])
    assert (args.epochs, args.patience) == (20, 0)
    assert TrainConfig().patience == 0


def test_patience_zero_runs_every_epoch(short_drive):
    argv = ARGS(short_drive, "--model", "pilotnet", "--protocol", "blocked", "--firewall", "20",
                "--epochs", "3", "--patience", "0")
    assert main(argv) == 0
    h = json.loads((short_drive / "runs" / "drivetest_pilotnet_blocked_s0" / "history.json").read_text())
    assert len(h["history"]) == 3 and h["config"]["epochs"] == 3 and h["config"]["patience"] == 0


def test_positive_patience_still_stops_early(short_drive, monkeypatch):
    import sightline.train

    calls = []

    def worsening(model, cache, indices, device, batch_size=256):
        calls.append(1)
        return cache.angles_deg[indices] + 5.0 * len(calls)

    monkeypatch.setattr(sightline.train, "predict", worsening)
    argv = ARGS(short_drive, "--model", "pilotnet", "--protocol", "blocked", "--firewall", "20",
                "--epochs", "5", "--patience", "1")
    assert main(argv) == 0
    h = json.loads((short_drive / "runs" / "drivetest_pilotnet_blocked_s0" / "history.json").read_text())
    assert len(h["history"]) == 2 and h["best_epoch"] == 0


# --- the training matrix --------------------------------------------------------------

MATRIX = [sys.executable, str(REPO / "scripts" / "train_matrix.py")]


def test_matrix_dry_run_lists_21_runs_seed_major(tmp_path):
    out = subprocess.run(MATRIX + ["--dry-run", "--runs-dir", str(tmp_path / "runs")],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    runs = [l for l in out if " train " in l]
    assert len(runs) == 21
    assert all("--seed 0" in l for l in runs[:7])
    assert not (tmp_path / "runs").exists()  # a dry run writes nothing


def test_every_matrix_command_is_a_valid_train_command(tmp_path):
    from sightline.cli import build_parser

    out = subprocess.run(MATRIX + ["--dry-run", "--runs-dir", str(tmp_path / "runs")],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    parsed = []
    for line in out:
        if " train " in line:
            argv = shlex.split(line)
            assert argv[1:4] == ["-m", "sightline.cli", "train"] and "--overwrite" not in argv
            parsed.append(build_parser().parse_args(argv[3:]))
    assert {(a.drive, a.epochs, a.patience, a.firewall, a.block, a.device) for a in parsed} == {
        ("drive2018", 20, 0, 250, 1500, "auto")}
    assert all("--epochs 20 --patience 0" in l for l in out if " train " in l)
    seeds = [a.seed for a in parsed]
    assert seeds == sorted(seeds)
    expected = {(m, p, s) for m in ("pilotnet", "sightlinenet") for p in ("leaky", "blocked", "forward")
                for s in (0, 1, 2)} | {("sightlinenet-1f", "blocked", s) for s in (0, 1, 2)}
    combos = [(a.model, a.protocol, a.seed) for a in parsed]
    assert len(combos) == len(set(combos)) and set(combos) == expected


def test_matrix_only_must_match_a_run(tmp_path):
    r = subprocess.run(MATRIX + ["--dry-run", "--runs-dir", str(tmp_path / "runs"), "--only", "nonet"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "matches no run" in r.stderr


def test_matrix_trains_a_real_run_then_skips_it(tmp_path):
    make_drive(tmp_path / "data", "drivetest", n=700)
    runs = tmp_path / "runs"
    cmd = MATRIX + ["--drive", "drivetest", "--data-dir", str(tmp_path / "data"), "--runs-dir", str(runs),
                    "--epochs", "1", "--block", "100", "--firewall", "20", "--device", "cpu",
                    "--parallel", "1", "--only", "sightlinenet-1f_blocked_s0$"]
    first = subprocess.run(cmd, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    run = runs / "drivetest_sightlinenet-1f_blocked_s0"
    finished = (run / "history.json").read_bytes()
    h = json.loads(finished)
    assert h["config"]["frame_lags"] == [] and h["config"]["notes"]["block"] == 100
    assert h["device"] == "cpu"
    assert "epoch 1/1" in (runs / "logs" / f"{run.name}.log").read_text()
    assert re.search(rf"done +{run.name} exit 0 in [0-9.]+s", first.stdout)

    second = subprocess.run(cmd, capture_output=True, text=True)
    assert second.returncode == 0, second.stdout + second.stderr
    assert re.search(rf"skip +{run.name}", second.stdout)
    assert (run / "history.json").read_bytes() == finished


FAKE_TRAIN = r"""
import json, os, pathlib, sys, time
run_dir, mode, history = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
print("fake training", run_dir.name, "pid", os.getpid(), " ".join(sys.argv[4:]), flush=True)
if mode == "fail":
    sys.exit(3)
if mode == "hang":
    time.sleep(120)
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "history.json").write_text(history)
"""


@pytest.fixture
def matrix(monkeypatch):
    spec = importlib.util.spec_from_file_location("train_matrix", REPO / "scripts" / "train_matrix.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "train_matrix", module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "POLL_SECONDS", 0.02)
    return module


def _history(job, epochs=20, firewall=250, block=1500, patience=0, **extra):
    return {"config": {"model": job.model, "epochs": epochs, "seed": job.seed, "patience": patience,
                       "notes": {"drive": job.drive, "protocol": job.protocol,
                                 "firewall": firewall, "block": block}}, **extra}


def _fake_children(matrix, monkeypatch, modes=None):
    """Stand-ins for the cache step and each training job: they only write history.json."""
    modes = modes or {}

    def train_command(job, opts, overwrite):
        history = json.dumps(_history(job, opts.epochs, opts.firewall, opts.block, by="matrix"))
        return [sys.executable, "-c", FAKE_TRAIN, str(Path(opts.runs_dir) / job.name),
                modes.get(job.name, "ok"), history, *(["--overwrite"] if overwrite else [])]

    monkeypatch.setattr(matrix, "train_command", train_command)
    monkeypatch.setattr(matrix, "cache_command", lambda opts: [sys.executable, "-c", "pass"])


def test_matrix_skips_finished_runs_and_never_passes_overwrite_to_new_ones(matrix, monkeypatch, tmp_path, capsys):
    _fake_children(matrix, monkeypatch)
    runs = tmp_path / "runs"
    jobs = matrix.matrix_jobs("drive2018")
    done = jobs[0]
    (runs / done.name).mkdir(parents=True)
    finished = json.dumps(_history(done, by="an earlier launch"))
    (runs / done.name / "history.json").write_text(finished)
    (runs / done.name / "best.pt").write_bytes(b"weights")

    assert matrix.main(["--runs-dir", str(runs), "--parallel", "3"]) == 0
    out = capsys.readouterr().out
    assert re.search(rf"skip +{done.name}", out)
    assert (runs / done.name / "history.json").read_text() == finished
    assert (runs / done.name / "best.pt").read_bytes() == b"weights"
    assert not (runs / "logs" / f"{done.name}.log").exists()
    for job in jobs[1:]:
        log = (runs / "logs" / f"{job.name}.log").read_text()
        assert "fake training" in log and "--overwrite" not in log
        assert json.loads((runs / job.name / "history.json").read_text())["by"] == "matrix"


def test_matrix_retrains_an_unfinished_run_from_scratch_and_says_so(matrix, monkeypatch, tmp_path, capsys):
    _fake_children(matrix, monkeypatch)
    runs = tmp_path / "runs"
    crashed = matrix.matrix_jobs("drive2018")[4]
    (runs / crashed.name).mkdir(parents=True)
    (runs / crashed.name / "best.pt").write_bytes(b"half-trained")

    assert matrix.main(["--runs-dir", str(runs), "--only", f"^{crashed.name}$"]) == 0
    out = capsys.readouterr().out
    log = (runs / "logs" / f"{crashed.name}.log").read_text()
    assert re.search(rf"start +{crashed.name} .*unfinished.*--overwrite", out)
    assert "unfinished" in log and log.count("--overwrite") >= 2  # header note and the command
    assert (runs / crashed.name / "history.json").is_file()


def test_matrix_leaves_a_finished_run_with_another_configuration_alone_and_fails(matrix, monkeypatch, tmp_path, capsys):
    _fake_children(matrix, monkeypatch)
    runs = tmp_path / "runs"
    job = matrix.matrix_jobs("drive2018")[0]
    (runs / job.name).mkdir(parents=True)
    smoke = json.dumps(_history(job, epochs=1))
    (runs / job.name / "history.json").write_text(smoke)

    assert matrix.main(["--runs-dir", str(runs), "--only", f"^{job.name}$"]) == 1
    out = capsys.readouterr().out
    assert re.search(rf"CONFLICT +{job.name} .*epochs=1", out)
    assert (runs / job.name / "history.json").read_text() == smoke
    assert not (runs / "logs" / f"{job.name}.log").exists()


def test_a_finished_run_trained_with_another_patience_is_a_conflict(matrix, tmp_path):
    job = matrix.matrix_jobs("drive2018")[0]
    opts = matrix.parse_args(["--runs-dir", str(tmp_path)])
    assert opts.patience == 0
    (tmp_path / job.name).mkdir()
    (tmp_path / job.name / "history.json").write_text(json.dumps(_history(job, patience=8)))
    state, note = matrix.run_state(job, opts)
    assert state == "conflict" and "patience=8" in note
    (tmp_path / job.name / "history.json").write_text(json.dumps(_history(job)))
    assert matrix.run_state(job, opts)[0] == "finished"


def test_matrix_survives_a_failing_job_and_exits_nonzero(matrix, monkeypatch, tmp_path, capsys):
    jobs = matrix.matrix_jobs("drive2018")
    _fake_children(matrix, monkeypatch, modes={jobs[1].name: "fail"})
    runs = tmp_path / "runs"

    assert matrix.main(["--runs-dir", str(runs), "--parallel", "2"]) == 1
    out = capsys.readouterr().out
    assert re.search(rf"FAILED +{jobs[1].name} exit 3 in [0-9.]+s", out)
    assert not (runs / jobs[1].name / "history.json").exists()
    assert all((runs / j.name / "history.json").is_file() for j in jobs if j != jobs[1])
    assert "20 ok, 1 failed" in out


def test_a_failed_cache_step_starts_no_training(matrix, monkeypatch, tmp_path, capsys):
    _fake_children(matrix, monkeypatch)
    monkeypatch.setattr(matrix, "cache_command", lambda opts: [sys.executable, "-c", "raise SystemExit(4)"])
    runs = tmp_path / "runs"

    assert matrix.main(["--runs-dir", str(runs)]) == 1
    assert re.search(r"FAILED +cache drive2018 exit 4", capsys.readouterr().out)
    assert sorted(p.name for p in (runs / "logs").iterdir()) == ["cache-drive2018.log"]


def test_matrix_records_its_pid_and_refuses_to_run_twice(matrix, monkeypatch, tmp_path, capsys):
    _fake_children(matrix, monkeypatch)
    runs = tmp_path / "runs"
    pid_file = runs / "logs" / "matrix.pid"
    seen = []
    fake = matrix.train_command

    def spy(job, opts, overwrite):
        seen.append(pid_file.read_text().strip())
        return fake(job, opts, overwrite)

    monkeypatch.setattr(matrix, "train_command", spy)
    assert matrix.main(["--runs-dir", str(runs), "--only", "pilotnet_leaky_s0"]) == 0
    assert seen == [str(os.getpid())]

    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "train_matrix"])
    try:
        pid_file.write_text(f"{other.pid}\n")
        assert matrix.main(["--runs-dir", str(runs)]) == 2
        assert f"already running (pid {other.pid})" in capsys.readouterr().err
        assert pid_file.read_text().strip() == str(other.pid)
    finally:
        other.kill()
        other.wait()
    # once that process is gone its pid file is stale and does not block a launch
    assert matrix.main(["--runs-dir", str(runs), "--only", "pilotnet_blocked_s0"]) == 0


def test_stopping_the_matrix_stops_its_jobs(matrix, monkeypatch, tmp_path, capsys):
    jobs = matrix.matrix_jobs("drive2018")
    _fake_children(matrix, monkeypatch, modes={j.name: "hang" for j in jobs})
    runs = tmp_path / "runs"
    before = signal.getsignal(signal.SIGTERM)
    timer = threading.Timer(1.5, os.kill, (os.getpid(), signal.SIGTERM))
    timer.start()
    t0 = time.monotonic()
    try:
        code = matrix.main(["--runs-dir", str(runs), "--parallel", "2"])
    finally:
        timer.cancel()
    out = capsys.readouterr().out
    assert code == 1 and time.monotonic() - t0 < 60
    assert signal.getsignal(signal.SIGTERM) == before
    stopped = [j for j in jobs if re.search(rf"stopped +{j.name}", out)]
    assert stopped == jobs[:2]
    for job in jobs[:2]:
        pid = int(re.search(r"pid (\d+)", (runs / "logs" / f"{job.name}.log").read_text()).group(1))
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert not any((runs / j.name / "history.json").exists() for j in jobs)
    assert not (runs / "logs" / f"{jobs[2].name}.log").exists()
