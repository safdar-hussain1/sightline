"""ONNX export: one graph from uint8 frame stacks to steering degrees and saliency, checked
against PyTorch with onnxruntime before it is kept.

Seeded random-weight models of both architectures are exported to tmp files; only the run
export at the end uses a drive, a small synthetic one with its cache.
"""
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from conftest import make_drive
from sightline.cli import main
from sightline.export import onnx_export
from sightline.export.onnx_export import export_onnx
from sightline.models import MODEL_SPECS, build_model
from sightline.saliency import SteeringWithSaliency

ARCHITECTURES = ("sightlinenet", "pilotnet")


def _model(name: str, seed: int = 0):
    torch.manual_seed(seed)
    return build_model(name).eval()


def _inputs(model, n: int, seed: int = 1) -> np.ndarray:
    channels = 3 * (1 + len(model.frame_lags))
    return np.random.default_rng(seed).integers(0, 256, (n, 66, 200, channels), dtype=np.uint8)


def _session(path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _torch(model, frames: np.ndarray):
    with torch.no_grad():
        angle, saliency = SteeringWithSaliency(model).eval()(torch.from_numpy(frames))
    return angle.numpy(), saliency.numpy()


def _error_line(err: str) -> str:
    assert "Traceback" not in err
    lines = [line for line in err.splitlines() if line.startswith("error: ")]
    assert len(lines) == 1, err
    return lines[0]


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_onnxruntime_reproduces_the_pytorch_angle_and_saliency(tmp_path, name):
    model = _model(name)
    frames = _inputs(model, 4)
    result = export_onnx(model, tmp_path / "m.onnx", frames, model_name=name, run="r")
    angle, saliency = _session(tmp_path / "m.onnx").run(None, {"frames": frames})
    torch_angle, torch_saliency = _torch(model, frames)
    assert angle.shape == (4, 1) and saliency.shape == (4, 66, 200)
    assert np.abs(angle - torch_angle).max() <= 1e-3
    assert np.abs(saliency - torch_saliency).max() <= 1e-3
    # the parity the export reports is the one measured on the frames it was given
    assert result["check_frames"] == 4
    assert result["max_angle_diff_deg"] == pytest.approx(float(np.abs(angle - torch_angle).max()), abs=1e-6)
    assert result["max_saliency_diff"] == pytest.approx(float(np.abs(saliency - torch_saliency).max()), abs=1e-6)
    data = (tmp_path / "m.onnx").read_bytes()
    assert (result["bytes"], result["sha256"]) == (len(data), hashlib.sha256(data).hexdigest())


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_the_graph_takes_uint8_frame_stacks_in_batches_of_any_size(tmp_path, name):
    model = _model(name)
    export_onnx(model, tmp_path / "m.onnx", _inputs(model, 2), model_name=name)
    session = _session(tmp_path / "m.onnx")
    (frames,) = session.get_inputs()
    assert frames.name == "frames" and frames.type == "tensor(uint8)"
    assert frames.shape[1:] == [66, 200, 3 * (1 + len(model.frame_lags))]
    assert isinstance(frames.shape[0], str)  # a named, dynamic batch axis
    assert [o.name for o in session.get_outputs()] == ["steering_deg", "saliency"]
    for n in (1, 5):
        x = _inputs(model, n, seed=n)
        angle, saliency = session.run(None, {"frames": x})
        assert angle.shape == (n, 1) and saliency.shape == (n, 66, 200)
        assert np.abs(angle - _torch(model, x)[0]).max() <= 1e-3
    assert [(o.domain, o.version) for o in onnx.load(tmp_path / "m.onnx").opset_import] == [("", 17)]


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_the_graph_names_its_frame_lags_model_and_run(tmp_path, name):
    model = _model(name)
    assert model.frame_lags == MODEL_SPECS[name].frame_lags
    export_onnx(model, tmp_path / "m.onnx", _inputs(model, 2), model_name=name, run="drive_x_s0")
    meta = _session(tmp_path / "m.onnx").get_modelmeta().custom_metadata_map
    assert json.loads(meta["frame_lags"]) == list(model.frame_lags)
    assert (meta["model"], meta["run"]) == (name, "drive_x_s0")


def test_the_same_weights_export_to_the_same_bytes(tmp_path):
    model = _model("sightlinenet")
    first = export_onnx(model, tmp_path / "a.onnx", _inputs(model, 3, seed=1), model_name="sightlinenet", run="r")
    # the same weights loaded again, checked on other frames
    second = export_onnx(_model("sightlinenet"), tmp_path / "b.onnx", _inputs(model, 5, seed=2),
                         model_name="sightlinenet", run="r")
    assert (tmp_path / "a.onnx").read_bytes() == (tmp_path / "b.onnx").read_bytes()
    assert first["sha256"] == second["sha256"]
    other = export_onnx(_model("sightlinenet", seed=1), tmp_path / "c.onnx", _inputs(model, 3),
                        model_name="sightlinenet", run="r")
    assert other["sha256"] != first["sha256"]


@pytest.mark.parametrize("gate", ["ANGLE_TOLERANCE_DEG", "SALIENCY_TOLERANCE"])
def test_an_export_that_fails_its_parity_check_leaves_no_file(tmp_path, monkeypatch, gate):
    monkeypatch.setattr(onnx_export, gate, -1.0)  # any measured difference now exceeds it
    model = _model("sightlinenet")
    out = tmp_path / "models" / "m.onnx"
    with pytest.raises(RuntimeError, match="parity"):
        export_onnx(model, out, _inputs(model, 2))
    assert not out.exists() and list(out.parent.iterdir()) == []


def test_an_export_over_its_size_limit_leaves_no_file(tmp_path):
    model = _model("pilotnet")  # about 6.4 MB of float32 weights
    out = tmp_path / "models" / "m.onnx"
    with pytest.raises(ValueError, match="limit"):
        export_onnx(model, out, _inputs(model, 1), max_bytes=2_000_000)
    assert not out.exists() and list(out.parent.iterdir()) == []


def test_the_parity_gate_can_be_set_per_export(tmp_path):
    model = _model("sightlinenet")
    result = export_onnx(model, tmp_path / "m.onnx", _inputs(model, 2), angle_tolerance_deg=1e-4,
                         saliency_tolerance=1e-4)
    assert (result["angle_tolerance_deg"], result["saliency_tolerance"]) == (1e-4, 1e-4)
    assert result["max_angle_diff_deg"] <= 1e-4 and result["max_saliency_diff"] <= 1e-4
    with pytest.raises(RuntimeError, match="exceeds -1 deg"):
        export_onnx(model, tmp_path / "n.onnx", _inputs(model, 2), angle_tolerance_deg=-1.0)
    assert not (tmp_path / "n.onnx").exists()


class _CheckedRegressor(torch.nn.Module):
    """The regressor behind a data-dependent check, which the tracer warns about."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        if bool(x.abs().sum() >= 0):
            return self.inner(x)
        return self.inner(x)


def test_the_export_hides_the_adapters_channel_check_warning_and_no_other(tmp_path):
    model = _model("sightlinenet")
    model.regressor = _CheckedRegressor(model.regressor)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_onnx(model, tmp_path / "m.onnx", _inputs(model, 2))
    traced = [w for w in caught if issubclass(w.category, torch.jit.TracerWarning)]
    # the adapter's check is hidden; the check in this file still warns
    assert traced and {Path(w.filename).name for w in traced} == {"test_export.py"}


# --- a trained run ----------------------------------------------------------------------

def _hand_made_run(runs, drive: str, n: int):
    """A run folder as training leaves one, with seeded random weights."""
    spec = MODEL_SPECS["sightlinenet"]
    run = runs / f"{drive}_sightlinenet_blocked_s0"
    run.mkdir(parents=True)
    (run / "history.json").write_text(json.dumps({"config": {
        "model": "sightlinenet", "frame_lags": list(spec.frame_lags), "dropout": spec.dropout,
        "notes": {"drive": drive}}}))
    torch.save(_model("sightlinenet").state_dict(), run / "best.pt")
    np.save(run / "test_idx.npy", np.arange(n // 2, n, dtype=np.int64))
    return run


@pytest.fixture
def run_and_drive(tmp_path):
    drive = make_drive(tmp_path / "data", "drivetest", n=60)
    loc = ["--data-dir", str(tmp_path / "data"), "--runs-dir", str(tmp_path / "runs")]
    assert main(["cache", "--drive", "drivetest", *loc]) == 0
    return _hand_made_run(tmp_path / "runs", "drivetest", 60), drive, loc


def test_a_run_exports_by_name_with_its_name_in_the_graph(tmp_path, capsys, run_and_drive):
    run, _, loc = run_and_drive
    out = tmp_path / "web" / "m.onnx"
    capsys.readouterr()
    assert main(["export", "--run", run.name, "--out", str(out), "--check-frames", "8", *loc]) == 0
    printed = json.loads(capsys.readouterr().out)
    meta = _session(out).get_modelmeta().custom_metadata_map
    assert (meta["run"], meta["model"], json.loads(meta["frame_lags"])) == (run.name, "sightlinenet", [2])
    assert printed["run"] == run.name and printed["check_frames"] == 8
    assert printed["max_angle_diff_deg"] <= 1e-3 and printed["max_saliency_diff"] <= 1e-3


def test_a_run_is_never_exported_from_a_cache_of_other_labels(tmp_path, capsys, run_and_drive):
    run, drive, loc = run_and_drive
    # one label edited in place, the file keeping its size: the cache no longer describes the drive
    lines = (drive / "data.txt").read_text().splitlines()
    name, rest = lines[40].split(" ", 1)
    angle, stamp = rest.split(",", 1)
    lines[40] = f"{name} {angle[:-1]}{'1' if angle[-1] != '1' else '2'},{stamp}"
    (drive / "data.txt").write_text("\n".join(lines) + "\n")
    out = tmp_path / "web" / "m.onnx"
    capsys.readouterr()
    code = main(["export", "--run", run.name, "--out", str(out), *loc])
    line = _error_line(capsys.readouterr().err)
    assert code == 2 and "sightline cache --drive drivetest" in line and "drivetest" in line
    assert not out.exists()
