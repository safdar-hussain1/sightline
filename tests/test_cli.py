import os
import subprocess
import sys
from pathlib import Path

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


def test_help_runs_as_a_module():
    out = subprocess.run([sys.executable, "-m", "sightline.cli", "--help"],
                         capture_output=True, text=True, check=True, cwd=REPO,
                         env={**os.environ, "PYTHONPATH": "src"}).stdout
    for sub in ("data", "cache", "train", "eval", "export"):
        assert sub in out


def test_eval_and_export_of_a_missing_run_are_clear_errors(tmp_path, capsys):
    runs = str(tmp_path / "runs")
    for argv in (
        ["eval", "--run", str(tmp_path / "runs" / "nope"), "--runs-dir", runs],
        ["export", "--run", str(tmp_path / "runs" / "nope"), "--out", str(tmp_path / "x.onnx"),
         "--runs-dir", runs],
        ["demo", "--drive", "drivetest", "--run", str(tmp_path / "runs" / "nope"),
         "--out", str(tmp_path / "d.mp4"), "--data-dir", str(tmp_path / "data"), "--runs-dir", runs],
    ):
        code = main(argv)
        err = capsys.readouterr().err
        assert code == 2 and "not found" in err and "Traceback" not in err


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
