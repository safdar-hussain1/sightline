"""The cockpit demo video: what the model sees, where it looks and how it steers, frame by frame.

The drawing helpers are checked on their own (the display filter, the wheel and the saliency
overlay); the renderer runs on a small synthetic drive with a hand-made run of seeded random
weights, through the command line, and its video is read back with OpenCV.
"""
import argparse
import inspect
import json
import math

import cv2
import numpy as np
import pytest
import torch

from conftest import REPO, W, make_drive
from sightline.cli import build_parser, main
from sightline.data.cache import DriveCache
from sightline.data.preprocess import CROP_BOTTOM
from sightline.demo import (
    BATCH,
    DISPLAY_ALPHA,
    HUMAN_BGR,
    MODEL_BGR,
    PANEL,
    draw_wheel,
    render_demo,
    saliency_overlay,
    smooth_toward,
)
from sightline.models import MODEL_SPECS, build_model
from sightline.paths import Workspace
from sightline.reporting import load_run_model
from sightline.train import predict


# --- the display filter -----------------------------------------------------------------------

def test_the_display_filter_is_defined_at_zero_error():
    for value in (0.0, 5.0, -338.82, 501.78):
        assert smooth_toward(value, value) == value


@pytest.mark.parametrize("display, target", [(0.0, 10.0), (10.0, -10.0), (-338.82, 501.78), (2.5, 2.5 + 1e-9)])
def test_the_display_filter_closes_a_fixed_share_of_the_gap_and_never_overshoots(display, target):
    step = smooth_toward(display, target)
    assert step == pytest.approx(display + DISPLAY_ALPHA * (target - display))
    assert min(display, target) <= step <= max(display, target)
    shown = display
    for _ in range(200):
        nearer = smooth_toward(shown, target)
        assert abs(target - nearer) <= abs(target - shown)
        shown = nearer
    assert shown == pytest.approx(target, abs=1e-6)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5, float("nan")])
def test_the_display_filter_refuses_a_share_outside_0_to_1(alpha):
    with pytest.raises(ValueError, match="alpha"):
        smooth_toward(0.0, 1.0, alpha=alpha)


# --- the wheel --------------------------------------------------------------------------------

def _pixels(img: np.ndarray, colour) -> tuple[np.ndarray, np.ndarray]:
    """Rows and columns of every pixel drawn exactly in ``colour``."""
    rows, cols = np.nonzero(np.all(img == np.array(colour, dtype=np.uint8), axis=2))
    return rows, cols


def test_the_wheel_draws_at_zero_error():
    img = draw_wheel(256, 0.0, 0.0)
    assert img.shape == (256, 256, 3) and img.dtype == np.uint8
    # the model's spoke points straight up to the rim, and the human's ghost ring sits on it there
    rows, cols = _pixels(img, MODEL_BGR)
    assert rows.min() < 40 and rows.max() <= 140 and abs(cols.mean() - 128) < 8
    rows, cols = _pixels(img, HUMAN_BGR)
    assert rows.size and rows.max() < 64 and abs(cols.mean() - 128) < 8


@pytest.mark.parametrize("angle, side", [(90.0, 1), (-90.0, -1), (450.0, 1), (-450.0, -1)])
def test_a_positive_angle_turns_the_wheel_clockwise(angle, side):
    rows, cols = _pixels(draw_wheel(256, angle), MODEL_BGR)
    assert side * (cols.mean() - 128) > 40 and abs(rows.mean() - 128) < 8


def test_the_ghost_marks_the_human_angle_apart_from_the_model():
    img = draw_wheel(256, 0.0, 90.0)
    rows, cols = _pixels(img, HUMAN_BGR)
    assert rows.size and cols.min() > 200 and abs(rows.mean() - 128) < 16
    rows, cols = _pixels(img, MODEL_BGR)
    assert rows.min() < 40 and rows.max() <= 140
    assert not _pixels(draw_wheel(256, 0.0), HUMAN_BGR)[0].size  # no human angle, no ghost


def test_angles_past_half_a_turn_draw_without_wrapping_errors():
    assert draw_wheel(256, 501.78, -338.82).shape == (256, 256, 3)


# --- the saliency overlay ---------------------------------------------------------------------

def test_saliency_lands_on_the_rows_the_model_sees_and_nowhere_else():
    frame = np.full((256, 455, 3), 60, np.uint8)
    saliency = np.zeros((66, 200), np.float32)
    saliency[:, 150:] = 1.0  # the right quarter of the model's input
    out = saliency_overlay(frame, saliency)
    assert out.shape == frame.shape and out.dtype == np.uint8
    top = 256 - CROP_BOTTOM
    assert np.array_equal(out[:top], frame[:top])  # rows the model never sees are left alone
    crop = out[top:].astype(float)
    assert crop[:, 360:].mean() > crop[:, :300].mean() + 60  # heat on the right, where the map is hot
    assert (frame == 60).all()  # the camera frame itself is not drawn on


def test_saliency_needs_a_frame_tall_enough_for_the_crop():
    with pytest.raises(ValueError, match="crop"):
        saliency_overlay(np.zeros((CROP_BOTTOM - 1, 455, 3), np.uint8), np.zeros((66, 200), np.float32))


# --- the renderer -----------------------------------------------------------------------------

def _hand_made_run(runs, drive: str, model: str):
    """A run folder as training leaves one, with seeded random weights."""
    spec = MODEL_SPECS[model]
    run = runs / f"{drive}_{model}_blocked_s0"
    run.mkdir(parents=True)
    (run / "history.json").write_text(json.dumps({"config": {
        "model": model, "frame_lags": list(spec.frame_lags), "dropout": spec.dropout,
        "notes": {"drive": drive}}}))
    torch.manual_seed(0)
    torch.save(build_model(model).state_dict(), run / "best.pt")
    return run


def _make_workspace(tmp_path, frames: int):
    """A synthetic drive of ``frames`` frames under ``tmp_path/data``, cached under ``tmp_path/runs``."""
    make_drive(tmp_path / "data", "drivetest", n=frames)
    loc = ["--data-dir", str(tmp_path / "data"), "--runs-dir", str(tmp_path / "runs")]
    assert main(["cache", "--drive", "drivetest", *loc]) == 0
    return tmp_path, loc


@pytest.fixture
def workspace(tmp_path):
    return _make_workspace(tmp_path, 60)


def _video(path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    info = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return frames, info


def _error_line(err: str) -> str:
    assert "Traceback" not in err
    lines = [line for line in err.splitlines() if line.startswith("error: ")]
    assert len(lines) == 1, err
    return lines[0]


def _ghost_angle(frame: np.ndarray) -> float:
    """The human's angle as the video shows it: where the amber ghost ring sits on the wheel, in
    degrees clockwise from straight up. The ring is found by its colour, loosely, because the
    codec shifts colours a little."""
    left = frame.shape[1] // 2 - PANEL // 2
    wheel = frame[-PANEL:, left:left + PANEL].astype(int)  # the wheel, under the two camera views
    blue, green, red = wheel[..., 0], wheel[..., 1], wheel[..., 2]
    rows, cols = np.nonzero((red > 170) & (green > 110) & (green < 235) & (blue < 140))
    assert rows.size > 20, "no ghost ring on the wheel"
    return math.degrees(math.atan2(cols.mean() - PANEL / 2, PANEL / 2 - rows.mean()))


def _bar_column(frame: np.ndarray) -> float:
    """Where the synthetic drive's bright bar stands in a camera frame, as a column: the bar fills
    the rows under the horizon and its position follows the steering angle."""
    bright = frame[160:250, :W].astype(float).mean(axis=(0, 2)) > 140
    assert bright.any(), "no bar in the camera frame"
    return float(np.nonzero(bright)[0].mean())


def test_a_demo_renders_the_requested_frames_of_a_run_named_under_runs_dir(workspace, capsys):
    tmp_path, loc = workspace
    run = _hand_made_run(tmp_path / "runs", "drivetest", "sightlinenet")
    out = tmp_path / "videos" / "demo.mp4"
    capsys.readouterr()
    code = main(["demo", "--drive", "drivetest", "--run", run.name, "--out", str(out),
                 "--start", "5", "--frames", "12", "--fps", "17.5", *loc])
    printed = capsys.readouterr().out
    assert code == 0 and out.is_file()
    frames, (count, fps) = _video(out)
    assert len(frames) == 12 and count == 12 and fps == pytest.approx(17.5)
    assert frames[0].shape == (256 + PANEL, 2 * 455, 3)  # two camera frames over the instrument panel
    assert f"wrote {out}" in printed and "12 frames" in printed and "from frame 5" in printed
    assert [p.name for p in out.parent.iterdir()] == ["demo.mp4"]  # no partial file left behind


@pytest.mark.parametrize("model", ["sightlinenet", "pilotnet"])
def test_the_demo_shows_the_angles_the_evaluation_computes(workspace, model):
    tmp_path, _ = workspace
    run = _hand_made_run(tmp_path / "runs", "drivetest", model)
    ws = Workspace(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs")
    result = render_demo("drivetest", run, tmp_path / "d.mp4", ws, start=0, num_frames=9)
    cache = DriveCache(tmp_path / "runs" / "cache" / "drivetest")
    net, _ = load_run_model(run)
    expected = predict(net, cache, np.arange(9), torch.device("cpu"))  # history frames as in evaluation
    assert result.device == "cpu" and result.first_frame == 0 and result.frames == 9
    np.testing.assert_allclose(result.model_deg, expected, atol=1e-4)
    np.testing.assert_array_equal(result.human_deg, cache.angles_deg[:9])
    assert result.mae_deg == pytest.approx(float(np.mean(np.abs(expected - cache.angles_deg[:9]))), abs=1e-4)
    assert result.zero_mae_deg == pytest.approx(float(np.mean(np.abs(cache.angles_deg[:9]))))


def test_a_render_longer_than_one_batch_keeps_every_frame_in_step(tmp_path):
    start, count = 3, 2 * BATCH + 22  # three batches, none of them starting on a multiple of the batch size
    _make_workspace(tmp_path, start + count + 5)
    run = _hand_made_run(tmp_path / "runs", "drivetest", "sightlinenet")
    ws = Workspace(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs")
    out = tmp_path / "long.mp4"
    result = render_demo("drivetest", run, out, ws, start=start, num_frames=count)
    cache = DriveCache(tmp_path / "runs" / "cache" / "drivetest")
    net, _ = load_run_model(run)
    stretch = np.arange(start, start + count)
    # every frame is scored with its own history, whichever batch carried it
    np.testing.assert_allclose(result.model_deg, predict(net, cache, stretch, torch.device("cpu")), atol=1e-5)
    np.testing.assert_array_equal(result.human_deg, cache.angles_deg[stretch])
    # and drawn in step: video frame i shows drive frame start + i, with that frame's human angle on the wheel
    frames, (counted, _) = _video(out)
    assert len(frames) == count == counted == result.frames
    for i, frame in enumerate(frames):
        source = cv2.imread(str(tmp_path / "data" / "drivetest" / f"{start + i}.jpg"))
        assert abs(_bar_column(frame) - _bar_column(source)) <= 1.0, f"video frame {i} shows another camera frame"
        gap = (_ghost_angle(frame) - cache.angles_deg[start + i] + 180.0) % 360.0 - 180.0
        assert abs(gap) < 1.0, f"video frame {i} marks another human angle"


def test_a_segment_past_the_end_of_the_drive_is_a_one_line_error(workspace, capsys):
    tmp_path, loc = workspace
    run = _hand_made_run(tmp_path / "runs", "drivetest", "sightlinenet")
    out = tmp_path / "demo.mp4"
    code = main(["demo", "--drive", "drivetest", "--run", run.name, "--out", str(out),
                 "--start", "55", "--frames", "6", *loc])
    line = _error_line(capsys.readouterr().err)
    assert code == 2 and "60 frames" in line and "55..60" in line
    assert not out.exists() and not list(tmp_path.glob(".demo*"))


@pytest.mark.parametrize("damage", ["deleted", "not a picture"])
def test_an_unreadable_frame_mid_segment_is_a_one_line_error_and_leaves_no_file(workspace, capsys, monkeypatch, damage):
    tmp_path, loc = workspace
    run = _hand_made_run(tmp_path / "runs", "drivetest", "sightlinenet")
    bad = tmp_path / "data" / "drivetest" / "13.jpg"  # frame 13 of the segment 5..16, after eight frames are written
    if damage == "deleted":
        bad.unlink()
    else:
        bad.write_bytes(b"not a jpeg")
    out = tmp_path / "videos" / "demo.mp4"
    listed = []
    read_picture = cv2.imread

    def reading(path, *args):
        if str(path) == str(bad):
            listed.append(sorted(p.name for p in out.parent.iterdir()))  # what is on disk as the read fails
        return read_picture(path, *args)

    monkeypatch.setattr(cv2, "imread", reading)
    code = main(["demo", "--drive", "drivetest", "--run", run.name, "--out", str(out),
                 "--start", "5", "--frames", "12", *loc])
    line = _error_line(capsys.readouterr().err)
    assert code == 2 and "frame 13" in line and "cannot be read" in line
    # the render was under way, writing into a hidden temporary file; none of it is left
    assert len(listed) == 1 and len(listed[0]) == 1 and listed[0][0].startswith("."), listed
    assert not out.exists() and list(out.parent.iterdir()) == []


def test_the_demo_reads_a_cache_and_never_builds_one(tmp_path, capsys):
    make_drive(tmp_path / "data", "drivetest", n=30)
    run = _hand_made_run(tmp_path / "runs", "drivetest", "sightlinenet")
    code = main(["demo", "--drive", "drivetest", "--run", run.name, "--out", str(tmp_path / "d.mp4"),
                 "--frames", "4", "--data-dir", str(tmp_path / "data"), "--runs-dir", str(tmp_path / "runs")])
    line = _error_line(capsys.readouterr().err)
    assert code == 2 and "sightline cache --drive drivetest" in line
    assert not (tmp_path / "runs" / "cache").exists() and not (tmp_path / "d.mp4").exists()


@pytest.mark.parametrize("flag, value, message", [
    ("--drive", "data/drivetest", "drive name"),
    ("--out", "demo.avi", ".mp4"),
])
def test_the_demo_refuses_a_drive_path_and_other_containers(workspace, capsys, flag, value, message):
    tmp_path, loc = workspace
    run = _hand_made_run(tmp_path / "runs", "drivetest", "sightlinenet")
    args = {"--drive": "drivetest", "--run": run.name, "--out": str(tmp_path / "d.mp4"), "--frames": "4"}
    args[flag] = value
    code = main(["demo", *[x for pair in args.items() for x in pair], *loc])
    assert code == 2 and message in _error_line(capsys.readouterr().err)
    assert not (tmp_path / "d.mp4").exists()


@pytest.mark.parametrize("flag, value", [("--start", "-1"), ("--frames", "0"), ("--fps", "0"), ("--fps", "nan")])
def test_the_demo_refuses_impossible_segment_settings(flag, value, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["demo", "--drive", "drivetest", "--run", "r", flag, value])
    assert exc.value.code == 2 and flag in capsys.readouterr().err


def test_the_default_frame_rate_is_the_one_drive2018s_timestamps_give():
    summary = json.loads((REPO / "reports" / "summary.json").read_text())
    interval = summary["drives"]["drive2018"]["median_dt_s"]
    assert summary["drives"]["drive2017"]["median_dt_s"] is None  # the other drive has no timestamps
    demo = next(a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)).choices["demo"]
    default = demo.get_default("fps")
    assert default == round(1.0 / interval, 1) == inspect.signature(render_demo).parameters["fps"].default
    words = " ".join(demo.format_help().split())  # argparse wraps the help text
    assert f"{default:g}" in words and f"{interval * 1000:.0f} ms" in words
    assert "drive2017 has no timestamps" in words and "inferred" in words
