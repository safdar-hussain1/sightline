import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from sightline.cli import main
from sightline.data.verify import verify_drive
from conftest import make_drive

REPO = Path(__file__).resolve().parents[1]


def test_missing_drive_is_a_clear_error(tmp_path, capsys):
    code = main(["cache", "--drive", "drive2018", "--data-dir", str(tmp_path / "nowhere"),
                 "--runs-dir", str(tmp_path / "runs")])
    err = capsys.readouterr().err
    assert code == 2
    assert "drive 'drive2018' not found" in err and "sightline data verify" in err
    assert "Traceback" not in err


def test_verify_flags_an_unknown_or_altered_drive(tmp_path):
    drive = make_drive(tmp_path, "drive2018", n=5)
    report = verify_drive(drive, "drive2018")
    assert not report["ok"] and any("sha256" in p for p in report["problems"])


def test_verify_reports_missing_images(tmp_path):
    drive = make_drive(tmp_path, "drivex", n=5)
    (drive / "3.jpg").unlink()
    report = verify_drive(drive, "drivex")
    assert any("have no image" in p for p in report["problems"])


def test_cache_builds_on_a_synthetic_drive(tmp_path, capsys):
    make_drive(tmp_path / "data", "drivetest", n=40)
    code = main(["cache", "--drive", "drivetest", "--data-dir", str(tmp_path / "data"),
                 "--runs-dir", str(tmp_path / "runs")])
    assert code == 0 and "cached 40 frames" in capsys.readouterr().out
    assert (tmp_path / "runs" / "cache" / "drivetest" / "inputs.u8").is_file()


def test_data_verify_exits_nonzero_on_a_problem(tmp_path, capsys):
    make_drive(tmp_path / "data", "drive2018", n=5)
    code = main(["data", "verify", "--drive", "drive2018", "--data-dir", str(tmp_path / "data")])
    assert code == 1 and "drive2018: PROBLEM" in capsys.readouterr().out


def test_help_runs_as_a_module(tmp_path):
    out = subprocess.run([sys.executable, "-m", "sightline.cli", "--help"],
                         capture_output=True, text=True, check=True, cwd=tmp_path,
                         env={**os.environ, "PYTHONPATH": str(REPO / "src")}).stdout
    listed = re.search(r"usage: sightline \[-h\]\s+\{([^}]*)\}", out)  # argparse may wrap the line
    assert listed, out
    assert {"data", "cache", "train", "eval", "report", "figures", "leakage", "legacy", "export", "site-data",
            "demo"} <= set(listed.group(1).split(","))


@pytest.mark.parametrize("argv", [["data", "verify"], ["cache"], ["train"], ["eval"], ["report"], ["figures"],
                                  ["leakage"], ["legacy"], ["export"], ["site-data"], ["demo"]])
def test_every_subcommand_has_its_own_help(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main([*argv, "--help"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: sightline {' '.join(argv)} ")


def _run_commands(tmp_path, run):
    """eval, export and demo pointed at ``run``; demo gets a real (synthetic) drive."""
    make_drive(tmp_path / "data", "drivetest", n=5)
    locations = ["--data-dir", str(tmp_path / "data"), "--runs-dir", str(tmp_path / "runs")]
    return (
        ["eval", "--run", str(run), *locations],
        ["export", "--run", str(run), "--out", str(tmp_path / "x.onnx"), *locations],
        ["demo", "--drive", "drivetest", "--run", str(run), "--out", str(tmp_path / "d.mp4"), *locations],
    )


def _error_line(err: str) -> str:
    """The CLI's one error line, after any progress output on stderr."""
    assert "Traceback" not in err
    lines = [line for line in err.splitlines() if line.startswith("error: ")]
    assert len(lines) == 1, err
    assert err.rstrip().endswith(lines[0]), err
    return lines[0]


def test_eval_export_and_demo_of_a_missing_run_are_clear_errors(tmp_path, capsys):
    for argv in _run_commands(tmp_path, tmp_path / "runs" / "nope"):
        code = main(argv)
        line = _error_line(capsys.readouterr().err)
        assert code == 2 and "run 'nope' not found" in line, argv[0]


def test_a_run_without_frame_lags_is_a_one_line_error(tmp_path, capsys):
    import torch

    from sightline.models import build_model

    run = tmp_path / "runs" / "drivetest_sightlinenet_blocked_s0"
    run.mkdir(parents=True)
    (run / "history.json").write_text(json.dumps({"config": {
        "model": "sightlinenet", "augment": True, "notes": {"drive": "drivetest"}}}))
    torch.save(build_model("sightlinenet-1f").state_dict(), run / "best.pt")
    for argv in _run_commands(tmp_path, run):
        code = main(argv)
        line = _error_line(capsys.readouterr().err)
        assert code == 2 and f"run {run.name!r} records no frame_lags" in line, argv[0]
        assert "train it again" in line, argv[0]


def test_data_verify_of_a_missing_drive_points_at_the_download_steps(tmp_path, capsys):
    code = main(["data", "verify", "--drive", "drive2018", "--data-dir", str(tmp_path / "nowhere")])
    out = capsys.readouterr().out
    assert code == 1 and "drive2018: PROBLEM drive 'drive2018' not found" in out
    assert "Get the data" in out and "sightline data verify" not in out


def test_export_without_a_drive_cache_names_the_cache_command(tmp_path, capsys):
    import json

    import torch

    from sightline.models import MODEL_SPECS, build_model

    spec = MODEL_SPECS["sightlinenet"]
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    (run / "history.json").write_text(json.dumps({"config": {
        "model": "sightlinenet", "frame_lags": list(spec.frame_lags), "dropout": spec.dropout,
        "notes": {"drive": "drivetest"}}}))
    torch.save(build_model("sightlinenet").state_dict(), run / "best.pt")
    code = main(["export", "--run", str(run), "--out", str(tmp_path / "x.onnx"),
                 "--runs-dir", str(tmp_path / "runs")])
    err = capsys.readouterr().err
    assert code == 2 and "sightline cache --drive drivetest" in err and "Traceback" not in err
