"""Per-run reports, the summary and its figures.

A module-scoped fixture trains seven one-epoch runs on a synthetic drive on CPU and
evaluates them with the real ``eval --all`` (device auto-picking disabled, so any attempt
to leave the CPU fails it); a second one runs ``report`` and ``figures`` on the reports.
Run-report tests use only the first, so a broken report fails its own test.
"""
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import REPO, make_drive
from sightline import train
from sightline.cli import build_parser, main
from sightline.data.cache import DriveCache
from sightline.data.splits import make_split, paired_frames

N, N_CROSS, BLOCK, FIREWALL = 1600, 300, 100, 20
RUNS = (
    ("pilotnet", "leaky", 0), ("pilotnet", "blocked", 0),
    ("sightlinenet", "leaky", 0), ("sightlinenet", "blocked", 0),
    ("sightlinenet", "leaky", 1), ("sightlinenet", "blocked", 1),
    ("sightlinenet", "forward", 0),
)
METRICS = {"n", "mae_deg", "rmse_deg", "medae_deg", "within_1p5_deg", "within_5_deg", "within_15_deg"}
SPLIT_SCORE = {"n", "model", "regimes", "baselines", "pearson_r", "sign_agreement", "skill_vs_zero",
               "smoothness"}
REPORT = {"run", "model", "protocol", "seed", "frame_lags", "n_params", "best_epoch", "epochs_run",
          "code_version", "versions", "device", "splits", "train_mean_deg", "val", "test", "cross_drive",
          "train_config"}
RECIPE = ("epochs", "patience", "lr", "batch_size", "weight_decay", "augment", "dropout")


def _name(model, protocol, seed):
    return f"drivetest_{model}_{protocol}_s{seed}"


def _no_device_picking():
    raise AssertionError("evaluation must not pick a device on its own")


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    root = tmp_path_factory.mktemp("reporting")
    make_drive(root / "data", "drivetest", n=N)
    make_drive(root / "data", "drivecross", n=N_CROSS, timestamps=False)
    loc = ["--data-dir", str(root / "data"), "--runs-dir", str(root / "runs")]
    for model, protocol, seed in RUNS:
        assert main(["train", "--drive", "drivetest", "--model", model, "--protocol", protocol,
                     "--seed", str(seed), "--epochs", "1", "--device", "cpu", "--block", str(BLOCK),
                     "--firewall", str(FIREWALL), *loc]) == 0
    reports_dir = root / "reports"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(train, "pick_device", _no_device_picking)
        assert main(["eval", "--all", "--cross-drive", "drivecross",
                     "--out-dir", str(reports_dir / "runs"), *loc]) == 0
    reports = {p.stem: json.loads(p.read_text()) for p in sorted((reports_dir / "runs").glob("*.json"))}
    return SimpleNamespace(root=root, loc=loc, reports_dir=reports_dir, reports=reports)


@pytest.fixture(scope="module")
def summary(ctx):
    """The summary of the fixture's reports, with its figures drawn."""
    assert main(["report", "--in-dir", str(ctx.reports_dir / "runs"),
                 "--out", str(ctx.reports_dir / "summary.json"), "--runs-dir", str(ctx.root / "runs")]) == 0
    assert main(["figures", "--summary", str(ctx.reports_dir / "summary.json"),
                 "--out-dir", str(ctx.reports_dir / "figures")]) == 0
    return json.loads((ctx.reports_dir / "summary.json").read_text())


# --- the per-run report ----------------------------------------------------------------

def test_every_trained_run_gets_one_report(ctx):
    assert sorted(ctx.reports) == sorted(_name(*r) for r in RUNS)


def test_run_report_has_the_contract_keys(ctx):
    for name, r in ctx.reports.items():
        assert REPORT <= r.keys(), name
        model, protocol, seed = name.split("_")[1], name.split("_")[2], int(name.rsplit("_s", 1)[1])
        assert (r["run"], r["model"], r["protocol"], r["seed"]) == (name, model, protocol, seed)
        assert r["frame_lags"] == ([2] if model == "sightlinenet" else [])
        assert r["epochs_run"] == 1 and r["best_epoch"] == 0 and r["device"] == "cpu"
        assert {"torch", "numpy", "python"} <= r["versions"].keys()
        assert {"train_n", "val_n", "test_n", "firewall"} <= r["splits"].keys()
        assert r["splits"]["firewall"] == FIREWALL
        assert r["cross_drive"]["drive"] == "drivecross"
        for split in ("val", "test", "cross_drive"):
            s = r[split]
            assert SPLIT_SCORE <= s.keys(), (name, split)
            assert s["model"].keys() == METRICS and s["model"]["n"] == s["n"]
            assert s["baselines"].keys() == {"zero", "mean", "zoh", "extrap"}
            assert all(m.keys() == METRICS for m in s["baselines"].values())
            assert s["regimes"].keys() == {"straight", "gentle", "sharp", "balanced_mae_deg"}
            assert all(s["regimes"][k].keys() == {"n", "mae_deg"} for k in ("straight", "gentle", "sharp"))
            assert s["skill_vs_zero"] == pytest.approx(
                1 - s["model"]["mae_deg"] / s["baselines"]["zero"]["mae_deg"], abs=1e-12)
            for who in ("model", "human"):
                assert "mean_abs_step_deg" in s["smoothness"][who]
                # the synthetic test drive is timestamped, the cross drive is not
                assert ("mean_abs_rate_deg_per_s" in s["smoothness"][who]) == (split != "cross_drive")
        assert r["cross_drive"]["n"] == N_CROSS - 2
        assert ("paired" in r) == (protocol in ("leaky", "blocked"))
        if "paired" in r:
            assert {"seed", "n", "model"} <= r["paired"].keys() and r["paired"]["model"].keys() == METRICS


def test_run_reports_record_the_training_recipe_from_the_run(ctx):
    from sightline.models import MODEL_SPECS

    for name, r in ctx.reports.items():
        config = json.loads((ctx.root / "runs" / name / "history.json").read_text())["config"]
        assert r["train_config"] == {key: config[key] for key in RECIPE}, name
        spec = MODEL_SPECS[r["model"]]
        assert r["train_config"] == {"epochs": 1, "patience": 0, "lr": 3e-4, "batch_size": 128,
                                     "weight_decay": 1e-4, "augment": spec.augment, "dropout": spec.dropout}, name


def test_paired_scores_use_identical_frames_for_both_protocols(ctx):
    for model, seeds in (("pilotnet", (0,)), ("sightlinenet", (0, 1))):
        for seed in seeds:
            expected = paired_frames(make_split("blocked", N, firewall=FIREWALL, block=BLOCK),
                                     make_split("leaky", N, seed=seed))
            digest = hashlib.sha256(expected.astype("<i8").tobytes()).hexdigest()
            leaky = ctx.reports[_name(model, "leaky", seed)]["paired"]
            blocked = ctx.reports[_name(model, "blocked", seed)]["paired"]
            assert leaky["seed"] == blocked["seed"] == seed
            assert leaky["n"] == blocked["n"] == expected.size > 0
            assert leaky["frames_sha256"] == blocked["frames_sha256"] == digest


def _fresh_cpu_predictions(ctx, run_name, indices):
    from sightline.reporting import load_run_model

    model, _ = load_run_model(ctx.root / "runs" / run_name)
    cache = DriveCache(ctx.root / "runs" / "cache" / "drivetest")
    pred = train.predict(model, cache, indices, torch.device("cpu")).astype(np.float64)
    return pred, cache.angles_deg[indices].astype(np.float64)


def test_reported_mae_recomputes_from_predictions(ctx):
    run = _name("sightlinenet", "blocked", 0)
    idx = np.load(ctx.root / "runs" / run / "test_idx.npy")
    pred, target = _fresh_cpu_predictions(ctx, run, idx)
    assert ctx.reports[run]["test"]["model"]["mae_deg"] == pytest.approx(np.abs(pred - target).mean(), abs=1e-9)

    leaky = _name("sightlinenet", "leaky", 0)
    paired = paired_frames(make_split("blocked", N, firewall=FIREWALL, block=BLOCK), make_split("leaky", N, seed=0))
    pred, target = _fresh_cpu_predictions(ctx, leaky, paired)
    assert ctx.reports[leaky]["paired"]["model"]["mae_deg"] == pytest.approx(
        np.abs(pred - target).mean(), abs=1e-9)


def test_reports_are_stable_plain_json(ctx, summary):
    def walk(value):
        if isinstance(value, dict):
            assert all(isinstance(k, str) for k in value)
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        else:
            assert type(value) in (int, float, str, bool, type(None)), (type(value), value)

    for path in [*sorted((ctx.reports_dir / "runs").glob("*.json")), ctx.reports_dir / "summary.json"]:
        text = path.read_text()
        data = json.loads(text)
        walk(data)
        assert text == json.dumps(data, indent=2, sort_keys=True) + "\n", path.name


def test_evaluation_defaults_to_the_cpu():
    assert build_parser().parse_args(["eval", "--run", "x"]).device == "cpu"
    assert build_parser().parse_args(["eval", "--run", "x"]).cross_drive == "drive2017"


def _copy_run(ctx, tmp_path, run_name, skip=()):
    src = ctx.root / "runs" / run_name
    dst = tmp_path / run_name
    dst.mkdir()
    for f in src.iterdir():
        if f.name not in skip:
            shutil.copy2(f, dst / f.name)
    return dst


def _error_line(err: str) -> str:
    assert "Traceback" not in err
    lines = [line for line in err.splitlines() if line.startswith("error: ")]
    assert len(lines) == 1, err
    return lines[0]


def test_a_run_missing_a_split_file_is_a_clear_error(ctx, tmp_path, capsys):
    run = _copy_run(ctx, tmp_path, _name("sightlinenet", "blocked", 0), skip=("test_idx.npy",))
    code = main(["eval", "--run", str(run), "--cross-drive", "none", *ctx.loc])
    line = _error_line(capsys.readouterr().err)
    assert code == 2 and "test_idx.npy" in line


def test_a_missing_cross_drive_is_a_clear_error_before_any_scoring(ctx, tmp_path, capsys):
    out = tmp_path / "r.json"
    code = main(["eval", "--run", _name("pilotnet", "blocked", 0), "--cross-drive", "drivenowhere",
                 "--out", str(out), *ctx.loc])
    line = _error_line(capsys.readouterr().err)
    assert code == 2 and "drive 'drivenowhere' not found" in line and not out.exists()


def test_a_run_whose_saved_split_is_not_its_protocol_split_is_refused(ctx, tmp_path, capsys):
    run = _copy_run(ctx, tmp_path, _name("sightlinenet", "leaky", 0))
    test_idx = np.load(run / "test_idx.npy")
    np.save(run / "test_idx.npy", test_idx[:-1])
    code = main(["eval", "--run", str(run), "--cross-drive", "none", *ctx.loc])
    line = _error_line(capsys.readouterr().err)
    assert code == 2 and "test_idx.npy" in line and "leaky" in line


def test_a_run_name_resolves_under_the_runs_dir(ctx, tmp_path, capsys):
    out = tmp_path / "one.json"
    name = _name("sightlinenet", "forward", 0)
    assert main(["eval", "--run", name, "--cross-drive", "drivecross", "--out", str(out), *ctx.loc]) == 0
    assert out.read_text() == (ctx.reports_dir / "runs" / f"{name}.json").read_text()
    printed = json.loads(capsys.readouterr().out)
    assert printed["test_mae_deg"] == ctx.reports[name]["test"]["model"]["mae_deg"]


def test_eval_all_skips_existing_reports(ctx, tmp_path, capsys):
    # two of the runs and the shared caches, so the test evaluates two runs, not seven
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "cache").symlink_to(ctx.root / "runs" / "cache")
    kept, missing = _name("pilotnet", "leaky", 0), _name("sightlinenet", "forward", 0)
    for name in (kept, missing):
        shutil.copytree(ctx.root / "runs" / name, runs / name)
    out = tmp_path / "reports"
    out.mkdir()
    marked = {**ctx.reports[kept], "marker": "kept"}  # still a valid report of the same checkpoint
    (out / f"{kept}.json").write_text(json.dumps(marked))
    capsys.readouterr()

    args = ["eval", "--all", "--cross-drive", "drivecross", "--out-dir", str(out),
            "--data-dir", str(ctx.root / "data"), "--runs-dir", str(runs)]
    assert main(args) == 0
    printed = capsys.readouterr().out.splitlines()
    assert json.loads((out / f"{kept}.json").read_text()) == marked
    assert printed[0].startswith(f"skipped {kept}: ") and printed[1].startswith(f"evaluated {missing}: ")
    # evaluating again reproduces the report byte for byte
    assert (out / f"{missing}.json").read_text() == (ctx.reports_dir / "runs" / f"{missing}.json").read_text()

    assert main([*args, "--overwrite"]) == 0
    assert (out / f"{kept}.json").read_text() == (ctx.reports_dir / "runs" / f"{kept}.json").read_text()


def test_eval_takes_a_run_or_all_but_not_both():
    with pytest.raises(SystemExit):
        main(["eval", "--run", "x", "--all"])
    with pytest.raises(SystemExit):
        main(["eval"])


# --- the summary ------------------------------------------------------------------------

def test_summary_sd_is_sample_sd_over_seeds(ctx, summary):
    from sightline.reporting import seed_stats

    assert seed_stats([1.0, 2.0, 3.0]) == {"mean": 2.0, "sd": 1.0, "values": [1.0, 2.0, 3.0]}
    assert seed_stats([4.0])["sd"] is None  # one seed has no spread to report

    group = summary["groups"]["sightlinenet/blocked"]
    values = [ctx.reports[_name("sightlinenet", "blocked", s)]["test"]["model"]["mae_deg"] for s in (0, 1)]
    assert group["seeds"] == [0, 1]
    assert group["test_mae_deg"]["values"] == values
    assert group["test_mae_deg"]["sd"] == pytest.approx(np.std(values, ddof=1))


def test_groups_carry_every_contract_statistic(ctx, summary):
    stats = ("val_mae_deg", "test_mae_deg", "cross_mae_deg", "test_r", "cross_r", "test_skill", "cross_skill")
    assert set(summary["groups"]) == {f"{m}/{p}" for m, p, _ in RUNS}
    for key, group in summary["groups"].items():
        for stat in stats:
            assert group[stat].keys() == {"mean", "sd", "values"}, (key, stat)
        assert isinstance(group["test_zero_mae_deg"], float) and isinstance(group["cross_zero_mae_deg"], float)
    blocked = ctx.reports[_name("sightlinenet", "blocked", 0)]
    assert summary["groups"]["sightlinenet/blocked"]["cross_zero_mae_deg"] == \
        blocked["cross_drive"]["baselines"]["zero"]["mae_deg"]


def test_shipped_model_is_chosen_on_validation_not_test():
    from sightline.reporting import seed_stats, select_shipped

    groups = {
        "sightlinenet/blocked": {
            "seeds": [0, 1], "runs": ["d_sightlinenet_blocked_s0", "d_sightlinenet_blocked_s1"],
            "val_mae_deg": seed_stats([5.0, 4.0]), "test_mae_deg": seed_stats([3.0, 6.0]),
        },
        "pilotnet/blocked": {  # better on validation, but not a candidate
            "seeds": [0], "runs": ["d_pilotnet_blocked_s0"],
            "val_mae_deg": seed_stats([1.0]), "test_mae_deg": seed_stats([1.0]),
        },
    }
    assert select_shipped(groups) == {"model": "sightlinenet", "protocol": "blocked", "seed": 1,
                                      "run": "d_sightlinenet_blocked_s1", "selected_by": "val_mae_deg"}


def test_summary_ships_the_best_validation_seed_and_names_its_shuffled_twin(ctx, summary):
    vals = {s: ctx.reports[_name("sightlinenet", "blocked", s)]["val"]["model"]["mae_deg"] for s in (0, 1)}
    best = min(vals, key=lambda s: (vals[s], s))
    assert summary["shipped"] == {"model": "sightlinenet", "protocol": "blocked", "seed": best,
                                      "run": _name("sightlinenet", "blocked", best),
                                      "selected_by": "val_mae_deg"}
    assert summary["twin"]["run"] == _name("sightlinenet", "leaky", best)
    assert summary["runs"] == sorted(ctx.reports)


def test_ladder_has_four_rungs_in_order(ctx, summary):
    ladder = summary["ladder"]["sightlinenet"]
    assert [r["rung"] for r in ladder] == ["shuffled", "blocked", "later in the drive", "another day"]
    assert [(r["protocol"], r["split"]) for r in ladder] == [
        ("leaky", "test"), ("blocked", "test"), ("forward", "test"), ("blocked", "cross_drive")]
    blocked = [ctx.reports[_name("sightlinenet", "blocked", s)] for s in (0, 1)]
    assert ladder[3]["mae_deg"]["values"] == [r["cross_drive"]["model"]["mae_deg"] for r in blocked]
    assert ladder[3]["zero_mae_deg"] == blocked[0]["cross_drive"]["baselines"]["zero"]["mae_deg"]
    assert ladder[1]["mae_deg"]["values"] == [r["test"]["model"]["mae_deg"] for r in blocked]
    # a model with no forward runs simply has no such rung
    assert [r["rung"] for r in summary["ladder"]["pilotnet"]] == ["shuffled", "blocked", "another day"]


def test_paired_leakage_compares_each_seed_on_its_own_paired_frames(ctx, summary):
    entry = summary["paired_leakage"]["sightlinenet"]
    assert [p["seed"] for p in entry["per_seed"]] == [0, 1]
    for p in entry["per_seed"]:
        leaky = ctx.reports[_name("sightlinenet", "leaky", p["seed"])]["paired"]
        blocked = ctx.reports[_name("sightlinenet", "blocked", p["seed"])]["paired"]
        assert p["n"] == leaky["n"]
        assert (p["shuffled_mae_deg"], p["blocked_mae_deg"]) == (leaky["model"]["mae_deg"], blocked["model"]["mae_deg"])
        assert p["ratio"] == pytest.approx(p["blocked_mae_deg"] / p["shuffled_mae_deg"])
    assert entry["ratio"]["values"] == [p["ratio"] for p in entry["per_seed"]]
    assert set(summary["paired_leakage"]) == {"pilotnet", "sightlinenet"}


def test_summary_refuses_paired_scores_on_different_frames(ctx):
    from sightline.reporting import aggregate

    reports = copy.deepcopy(list(ctx.reports.values()))
    for r in reports:
        if r["run"] == _name("sightlinenet", "leaky", 1):
            r["paired"]["frames_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="paired"):
        aggregate(reports)


def test_summary_refuses_two_reports_of_one_run(ctx):
    from sightline.reporting import aggregate

    reports = list(ctx.reports.values())
    with pytest.raises(ValueError, match="twice"):
        aggregate([*reports, reports[0]])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _with_change(ctx, run, change):
    reports = copy.deepcopy(list(ctx.reports.values()))
    for r in reports:
        if r["run"] == run:
            change(r)
    return reports


@pytest.mark.parametrize("what, run, change", [
    ("split settings", _name("sightlinenet", "blocked", 1), lambda r: r["splits"].update(block=99)),
    ("split settings", _name("pilotnet", "leaky", 0), lambda r: r["splits"].update(firewall=7)),
    ("evaluation device", _name("pilotnet", "leaky", 0), lambda r: r.update(device="mps")),
    ("frame_lags of sightlinenet", _name("sightlinenet", "leaky", 1), lambda r: r.update(frame_lags=[])),
    ("zero baseline on the sightlinenet/blocked test split", _name("sightlinenet", "blocked", 1),
     lambda r: r["test"]["baselines"]["zero"].update(mae_deg=1.0)),
    ("zero baseline on the cross drive", _name("pilotnet", "blocked", 0),
     lambda r: r["cross_drive"]["baselines"]["zero"].update(mae_deg=1.0)),
    ("library versions", _name("pilotnet", "blocked", 0), lambda r: r["versions"].update(torch="0.0")),
    ("training drive", _name("sightlinenet", "forward", 0), lambda r: r.update(drive="driveother")),
    ("drive 'drivetest'", _name("sightlinenet", "forward", 0),
     lambda r: r["drives"]["drivetest"].update(label_sha256="0" * 64)),
    ("cross drive", _name("sightlinenet", "forward", 0), lambda r: r.pop("cross_drive")),
    ("training recipe", _name("pilotnet", "blocked", 0), lambda r: r["train_config"].update(epochs=2)),
    ("training recipe", _name("sightlinenet", "forward", 0), lambda r: r["train_config"].update(patience=8)),
    ("training recipe", _name("sightlinenet", "leaky", 1), lambda r: r["train_config"].update(lr=1e-3)),
    ("training recipe", _name("pilotnet", "leaky", 0), lambda r: r["train_config"].update(batch_size=64)),
    ("training recipe", _name("sightlinenet", "blocked", 0), lambda r: r["train_config"].update(weight_decay=0.0)),
    ("training recipe", _name("sightlinenet", "blocked", 1), lambda r: r.pop("train_config")),
    ("augmentation and dropout of sightlinenet", _name("sightlinenet", "blocked", 1),
     lambda r: r["train_config"].update(augment="standard")),
    ("augmentation and dropout of pilotnet", _name("pilotnet", "blocked", 0),
     lambda r: r["train_config"].update(dropout=0.5)),
    ("shares of frames within a tolerance", _name("sightlinenet", "forward", 0),
     lambda r: r["test"]["model"].update(within_2_deg=0.5)),
])
def test_summary_refuses_runs_made_or_scored_differently(ctx, what, run, change):
    from sightline.reporting import aggregate

    with pytest.raises(ValueError, match=re.escape(what)) as exc:
        aggregate(_with_change(ctx, run, change))
    message = str(exc.value)
    assert run in message and "\n" not in message


def test_summary_records_the_shared_recipe_and_each_models_own_augmentation(ctx, summary):
    from sightline.models import MODEL_SPECS

    # the two models train with different augmentation and dropout, and that is accepted
    assert MODEL_SPECS["pilotnet"].augment != MODEL_SPECS["sightlinenet"].augment
    assert MODEL_SPECS["pilotnet"].dropout != MODEL_SPECS["sightlinenet"].dropout
    assert summary["train_config"] == {
        "epochs": 1, "patience": 0, "lr": 3e-4, "batch_size": 128, "weight_decay": 1e-4,
        "models": {model: {"augment": MODEL_SPECS[model].augment, "dropout": MODEL_SPECS[model].dropout}
                   for model in ("pilotnet", "sightlinenet")},
    }


def test_summary_refuses_a_report_whose_run_has_another_checkpoint(ctx, tmp_path):
    from sightline.reporting import aggregate

    name = _name("sightlinenet", "blocked", 0)
    runs = tmp_path / "runs"
    shutil.copytree(ctx.root / "runs" / name, runs / name)
    reports = list(ctx.reports.values())
    aggregate(reports, runs_dir=runs)  # the same checkpoint: accepted
    # the run is trained again: another checkpoint of the same model replaces best.pt
    shutil.copy2(ctx.root / "runs" / _name("sightlinenet", "blocked", 1) / "best.pt", runs / name / "best.pt")
    with pytest.raises(ValueError, match="checkpoint") as exc:
        aggregate(reports, runs_dir=runs)
    assert name in str(exc.value)


def test_a_summary_without_run_folders_is_the_same_summary(ctx, summary, tmp_path, capsys):
    out = tmp_path / "summary.json"
    capsys.readouterr()
    assert main(["report", "--in-dir", str(ctx.reports_dir / "runs"), "--out", str(out),
                 "--runs-dir", str(tmp_path / "no-runs")]) == 0
    assert "checked the checkpoints of 0 of 7 runs" in capsys.readouterr().out
    assert out.read_text() == (ctx.reports_dir / "summary.json").read_text()


def test_every_report_names_its_checkpoint(ctx):
    for name, r in ctx.reports.items():
        assert r["checkpoint_sha256"] == _sha256(ctx.root / "runs" / name / "best.pt"), name


def test_eval_all_re_evaluates_a_report_made_from_another_checkpoint(ctx, tmp_path, capsys):
    name = _name("sightlinenet", "blocked", 0)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "cache").symlink_to(ctx.root / "runs" / "cache")
    shutil.copytree(ctx.root / "runs" / name, runs / name)
    out = tmp_path / "reports"
    out.mkdir()
    old = json.loads((ctx.reports_dir / "runs" / f"{name}.json").read_text())
    shutil.copy2(ctx.reports_dir / "runs" / f"{name}.json", out / f"{name}.json")
    # the run is trained again: another checkpoint of the same model replaces best.pt
    shutil.copy2(ctx.root / "runs" / _name("sightlinenet", "blocked", 1) / "best.pt", runs / name / "best.pt")
    capsys.readouterr()

    assert main(["eval", "--all", "--cross-drive", "drivecross", "--out-dir", str(out),
                 "--data-dir", str(ctx.root / "data"), "--runs-dir", str(runs)]) == 0
    printed = capsys.readouterr().out
    new = json.loads((out / f"{name}.json").read_text())
    assert f"re-evaluating {name}: its report was made from another checkpoint" in printed
    assert new["checkpoint_sha256"] == _sha256(runs / name / "best.pt") != old["checkpoint_sha256"]
    assert new["test"]["model"]["mae_deg"] != old["test"]["model"]["mae_deg"]


def test_summary_records_the_drives_and_the_shipped_regimes(ctx, summary):
    assert summary["device"] == "cpu"
    drives = summary["drives"]
    assert drives["drivetest"]["frames"] == N and drives["drivecross"]["frames"] == N_CROSS
    assert drives["drivetest"]["median_dt_s"] == pytest.approx(0.057, abs=1e-6)  # epoch-second timestamps
    assert drives["drivecross"]["median_dt_s"] is None
    shipped = ctx.reports[summary["shipped"]["run"]]
    regimes = summary["regimes"]
    assert regimes["models"]["sightlinenet"]["run"] == summary["shipped"]["run"]
    assert regimes["models"]["sightlinenet"]["regimes"] == shipped["test"]["regimes"]
    assert regimes["zero"] == shipped["test"]["zero_regimes"]


def test_summary_records_the_cutoffs_of_the_evaluation_code(summary):
    from sightline.evaluate import REGIME_BINS, SIGN_MIN_ABS_DEG, TOLERANCES_DEG
    from sightline.reporting import evaluation_cutoffs

    assert summary["cutoffs"] == evaluation_cutoffs() == {
        "regime_bins_deg": list(REGIME_BINS), "sign_min_abs_deg": SIGN_MIN_ABS_DEG,
        "tolerances_deg": list(TOLERANCES_DEG)}
    assert all(type(v) is float for v in [*summary["cutoffs"]["regime_bins_deg"], summary["cutoffs"]["sign_min_abs_deg"],
                                          *summary["cutoffs"]["tolerances_deg"]])


def test_summary_refuses_reports_scored_with_other_tolerances_than_the_code(ctx):
    from sightline.reporting import aggregate

    reports = copy.deepcopy(list(ctx.reports.values()))
    for r in reports:
        r["test"]["model"]["within_20_deg"] = r["test"]["model"].pop("within_15_deg")
    with pytest.raises(ValueError, match="within_15_deg; evaluate them again") as exc:
        aggregate(reports)
    assert "within_20_deg" in str(exc.value) and "\n" not in str(exc.value)


def test_the_committed_summary_records_the_cutoffs_of_the_evaluation_code():
    from sightline.reporting import evaluation_cutoffs

    summary = json.loads((REPO / "reports" / "summary.json").read_text())
    assert summary["cutoffs"] == evaluation_cutoffs()


def test_report_and_figures_name_a_missing_input(tmp_path, capsys):
    assert main(["report", "--in-dir", str(tmp_path / "none"), "--out", str(tmp_path / "s.json")]) == 2
    assert "none" in _error_line(capsys.readouterr().err)
    assert main(["figures", "--summary", str(tmp_path / "s.json"), "--out-dir", str(tmp_path / "f")]) == 2
    assert "s.json" in _error_line(capsys.readouterr().err)


# --- the figures ------------------------------------------------------------------------

def test_figures_are_written(ctx, summary):
    from PIL import Image

    from sightline.figures import FIGURE_SIZES, MODEL_COLOURS
    from sightline.models import MODEL_SPECS

    assert set(MODEL_COLOURS) == set(MODEL_SPECS)  # every model keeps one colour in every figure
    assert set(FIGURE_SIZES) == {"ladder.png", "paired-leakage.png", "regimes.png"}
    for name, size in FIGURE_SIZES.items():
        path = ctx.reports_dir / "figures" / name
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name
        with Image.open(path) as img:
            assert img.size == size, name


def _stat(*values):
    from sightline.reporting import seed_stats

    return seed_stats(values)


def test_each_ladder_bar_gets_the_zero_tick_of_its_own_frames():
    from sightline.figures import draw

    def rung(name, protocol, split, mae, zero):
        return {"rung": name, "protocol": protocol, "split": split, "mae_deg": _stat(*mae), "zero_mae_deg": zero}

    summary = {"ladder": {
        "sightlinenet": [rung("shuffled", "leaky", "test", (5.0, 6.0), 10.0),
                         rung("another day", "blocked", "cross_drive", (9.0, 9.5), 20.0)],
        "pilotnet": [rung("shuffled", "leaky", "test", (7.0,), 12.0)],  # other seeds, other frames
    }}
    ax = draw(summary, "ladder.png").axes[0]
    ticks = [c for c in ax.collections if c.get_gid() == "zero-baseline"]
    bars = sorted(ax.patches, key=lambda b: b.get_x())
    assert len(ticks) == len(bars) == 3
    spans = sorted((seg[0][0], seg[1][0], seg[0][1]) for c in ticks for seg in c.get_segments())
    for bar, (x0, x1, y) in zip(bars, spans):
        assert x0 <= bar.get_x() and bar.get_x() + bar.get_width() <= x1  # the tick sits on its bar
    assert [y for _, _, y in spans] == [10.0, 12.0, 20.0]
    subtitle = ax.get_title(loc="left")
    assert "test split" in subtitle and "another day" in subtitle  # every rung's frames are named
    # another day is scored from frame CROSS_FIRST_FRAME on, the drive's third frame: the subtitle says so
    from sightline.reporting import CROSS_FIRST_FRAME

    ordinals = {1: "second", 2: "third", 3: "fourth"}
    assert f"another day on the other drive from its {ordinals[CROSS_FIRST_FRAME]} frame" in subtitle
    assert "all of the other drive" not in subtitle


def test_the_regime_labels_state_the_edges_the_evaluation_cuts_at():
    from sightline.evaluate import REGIME_BINS
    from sightline.figures import REGIMES, draw

    assert [key for key, _ in REGIMES] == ["straight", "gentle", "sharp"]
    edges = [[float(x) for x in re.findall(r"\d+(?:\.\d+)?", label)] for _, label in REGIMES]
    assert edges == [[REGIME_BINS[0]], list(REGIME_BINS), [REGIME_BINS[1]]]
    assert ["<", "–", "≥"] == [next(c for c in label if c in "<–≥") for _, label in REGIMES]
    regimes = {k: {"n": 10, "mae_deg": 1.0} for k, _ in REGIMES}
    summary = {"shipped": {"model": "sightlinenet", "seed": 0},
               "regimes": {"protocol": "blocked", "split": "test", "zero": regimes,
                           "models": {"sightlinenet": {"regimes": regimes}}}}
    shown = [t.get_text() for t in draw(summary, "regimes.png").axes[0].get_xticklabels()]
    assert shown == [f"{label}\n10 frames" for _, label in REGIMES]


def test_the_paired_ratio_is_printed_to_two_decimals():
    from sightline.figures import draw

    per_seed = [{"seed": 0, "n": 10, "shuffled_mae_deg": 2.0, "blocked_mae_deg": 2.08, "ratio": 1.04}]
    summary = {"paired_leakage": {"sightlinenet": {"per_seed": per_seed, "ratio": _stat(1.04)}}}
    legend = draw(summary, "paired-leakage.png").axes[0].get_legend()
    assert any("1.04×" in t.get_text() for t in legend.get_texts())


def test_a_report_is_current_only_for_the_checkpoint_it_describes(ctx, tmp_path):
    from sightline.reporting import stale_reason

    name = _name("pilotnet", "blocked", 0)
    report = ctx.reports_dir / "runs" / f"{name}.json"
    assert stale_reason(report, ctx.root / "runs" / name) is None
    unreadable = tmp_path / "bad.json"
    unreadable.write_text("not json")
    assert stale_reason(unreadable, ctx.root / "runs" / name) == "its report cannot be read"
    no_weights = _copy_run(ctx, tmp_path, name, skip=("best.pt",))
    assert stale_reason(report, no_weights) == "its run has no best.pt"
    no_recipe = tmp_path / "no-recipe.json"
    no_recipe.write_text(json.dumps({k: v for k, v in ctx.reports[name].items() if k != "train_config"}))
    assert stale_reason(no_recipe, ctx.root / "runs" / name) == "its report records no training recipe"
