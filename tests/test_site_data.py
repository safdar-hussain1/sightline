"""The site data: everything the browser loads, written from the reports and the checkpoints.

A module-scoped fixture makes two synthetic drives named like the real ones (the 2017
stand-in without timestamps), trains the shipped run and its shuffled-split twin for one
epoch on the CPU, evaluates and summarises them, runs the leakage analysis and the
standard-recipe evaluation (random TF1 weights), and then runs ``sightline site-data``
twice: once as published, with the clip videos when ffmpeg is installed, and once without
videos into a second folder, which must hold the same bytes otherwise.

The 2018 stand-in keeps the image of frame ``i`` under the file name frame ``i + 7`` had:
a drive's frames are its label rows, so code that assumes ``<i>.jpg`` shows the wrong frame.
One of its labels is a 120 deg spike, so the clip window that would vary most is one a clip
may not show (every label within 90 deg) and the rule must pick another. Each stand-in also
logs one sensor dropout, a 0.0 between large angles of one sign, which the label repair
interpolates, so the labels the drive profile describes must be the repaired ones.

The tests at the end check the committed site data in ``web/public`` against the committed
reports; they skip while ``web/public/data/results.json`` is not tracked by git, and fail
when it is tracked but missing or unreadable.
"""
import base64
import copy
import errno
import hashlib
import inspect
import itertools
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
import torch
from PIL import Image, JpegImagePlugin

from conftest import REPO, make_drive
from sightline.cli import main
from sightline.data import splits
from sightline.data.cache import DriveCache
from sightline.data.labels import DriveLabels, load_labels, repair_dropouts
from sightline.data.splits import SplitIndices, frames_sha256, make_split, paired_frames
from sightline.demo import render_demo
from sightline.export import site_data
from sightline.export.onnx_export import ANGLE_TOLERANCE_DEG, SALIENCY_TOLERANCE
from sightline.legacy import TENSOR_SHAPES
from sightline.paths import MissingDataError, UnusableRunError, Workspace
from sightline.reporting import load_run_model

N, N_CROSS, BLOCK, FIREWALL, CLIP, SHIFT = 1600, 300, 200, 20, 100, 7
#: a label past the clips' 90 deg limit, inside the 2018 test stretch's most varied windows
SPIKE_ROW, SPIKE_DEG = 1150, 120.0
#: the row where each stand-in logs a sensor dropout; its neighbours are large angles of one sign
DROPOUT_ROW = {"drive2018": 300, "drive2017": 100}
#: the frames each protocol trains, validates and tests on in the 1,600-frame stand-in with 200-frame
#: blocks and a 20-frame firewall, counted by hand:
#:   shuffled: 70% of 1,600 train, the next 15% validate, the last 15% test, nothing is dropped;
#:   blocked: the blocks run TTVTTETT (T train, V validation, E test) and 40 frames are dropped at each of
#:     the four boundaries between a training block and another: 380 + 360 + 380 frames train, 160
#:     validate, 160 test;
#:   later in the drive: training ends at frame 1,120 and validation at frame 1,360, with 20 frames
#:     dropped on each side of both
HAND_COUNTS = {
    "leaky": {"train": 1120, "val": 240, "test": 240},
    "blocked": {"train": 1120, "val": 160, "test": 160},
    "forward": {"train": 1100, "val": 200, "test": 220},
}
FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None or FFPROBE is None,
                                  reason="ffmpeg and ffprobe are not installed, so no clip video was written")
TITLES = {"c2018": "2018: minutes it never trained on", "c2017": "2017: another day"}
PUBLISHED = REPO / "web" / "public"


# --- the fixture ------------------------------------------------------------------------

def _store_frames_under_other_names(drive: Path, shift: int) -> None:
    """Keep row ``i``'s image under the file name row ``i + shift`` had, and point row ``i`` at it."""
    lines = (drive / "data.txt").read_text().splitlines()
    n = len(lines)
    for i in range(n):
        os.replace(drive / f"{i}.jpg", drive / f"old-{i}.jpg")
    renamed = []
    for i, line in enumerate(lines):
        name, rest = line.split(" ", 1)
        assert name == f"{i}.jpg"
        os.replace(drive / f"old-{i}.jpg", drive / f"{(i + shift) % n}.jpg")
        renamed.append(f"{(i + shift) % n}.jpg {rest}")
    (drive / "data.txt").write_text("\n".join(renamed) + "\n")


def _set_angle(drive: Path, row: int, angle: float) -> None:
    lines = (drive / "data.txt").read_text().splitlines()
    name, rest = lines[row].split(" ", 1)
    lines[row] = f"{name} {angle:.6f},{rest.split(',', 1)[1]}"
    (drive / "data.txt").write_text("\n".join(lines) + "\n")


def _tf1_weights(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {name: (rng.normal(0.0, 0.1, shape) if len(shape) == 1
                   else rng.normal(0.0, np.sqrt(2.0 / np.prod(shape[:-1])), shape)).astype(np.float32)
            for name, shape in TENSOR_SHAPES.items()}


def _site_data_args(summary: Path, out_dir: Path, reports: Path, loc: list[str], video: bool) -> list[str]:
    return ["site-data", "--summary", str(summary), "--out-dir", str(out_dir), "--reports-dir", str(reports),
            "--clip-frames", str(CLIP), *([] if video else ["--no-video"]), *loc]


def _files(folder: Path) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("site")
    data, runs, reports = root / "data", root / "runs", root / "reports"
    _store_frames_under_other_names(
        make_drive(data, "drive2018", n=N, scenery=True, dropouts=(DROPOUT_ROW["drive2018"],)), SHIFT)
    _set_angle(data / "drive2018", SPIKE_ROW, SPIKE_DEG)
    make_drive(data, "drive2017", n=N_CROSS, timestamps=False, dropouts=(DROPOUT_ROW["drive2017"],))
    loc = ["--data-dir", str(data), "--runs-dir", str(runs)]
    split = ["--block", str(BLOCK), "--firewall", str(FIREWALL)]
    for protocol in ("blocked", "leaky"):
        assert main(["train", "--drive", "drive2018", "--model", "sightlinenet", "--protocol", protocol,
                     "--seed", "0", "--epochs", "1", "--device", "cpu", *split, *loc]) == 0
    assert main(["eval", "--all", "--cross-drive", "drive2017", "--out-dir", str(reports / "runs"), *loc]) == 0
    assert main(["report", "--in-dir", str(reports / "runs"), "--out", str(reports / "summary.json"),
                 "--runs-dir", str(runs)]) == 0
    assert main(["leakage", "--drive", "drive2018", "--seed", "0", *split,
                 "--out", str(reports / "leakage.json"), *loc]) == 0
    weights = root / "private" / "weights.npz"
    weights.parent.mkdir()
    np.savez(weights, **_tf1_weights())
    assert main(["legacy", "--weights", str(weights), *split, "--out", str(reports / "legacy.json"),
                 "--data-dir", str(data)]) == 0

    second = root / "second"
    shutil.copytree(reports, second / "reports")
    out = root / "web" / "public"
    video = FFMPEG is not None
    assert main(_site_data_args(reports / "summary.json", out, reports, loc, video)) == 0
    assert main(_site_data_args(second / "reports" / "summary.json", second / "public", second / "reports",
                                loc, video=False)) == 0

    def load(path: Path):
        return json.loads(path.read_text())

    ctx = SimpleNamespace(root=root, data=data, runs=runs, reports=reports, out=out, second=second, loc=loc,
                          video=video)
    ctx.summary = load(reports / "summary.json")
    ctx.run_reports = {p.stem: load(p) for p in (reports / "runs").glob("*.json")}
    ctx.leakage, ctx.legacy = load(reports / "leakage.json"), load(reports / "legacy.json")
    ctx.export = load(reports / "export.json")
    ctx.results = load(out / "data" / "results.json")
    ctx.clips = {cid: load(out / "data" / "clips" / cid / "clip.json") for cid in TITLES}
    ctx.pairs = load(out / "data" / "leakage" / "pairs.json")
    ctx.selftest = load(out / "data" / "selftest" / "expected.json")
    ctx.shipped, ctx.twin = ctx.summary["shipped"]["run"], ctx.summary["twin"]["run"]
    ctx.raw_labels = {d: load_labels(data / d) for d in ("drive2018", "drive2017")}
    ctx.labels = {d: repair_dropouts(labels) for d, labels in ctx.raw_labels.items()}
    ctx.caches = {d: DriveCache(runs / "cache" / d) for d in ("drive2018", "drive2017")}
    return ctx


def _clip_dir(site, cid: str) -> Path:
    return site.out / "data" / "clips" / cid


def _decode(paths) -> np.ndarray:
    """The frames of sprite sheets, decoded with Pillow, written independently of the builder."""
    frames = []
    for path in paths:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        frames.extend(pixels[k * 66:(k + 1) * 66] for k in range(pixels.shape[0] // 66))
    return np.stack(frames)


def _decoded_clip(site, cid: str) -> np.ndarray:
    clip = site.clips[cid]
    return _decode([_clip_dir(site, cid) / name for name in clip["sprite"]["files"]])


def _stacks(frames: np.ndarray, lead_in: int, n: int, lags) -> np.ndarray:
    """Model inputs for frames ``lead_in .. lead_in + n - 1``: each frame, then each lag's frame."""
    return np.stack([np.concatenate([frames[t]] + [frames[t - lag] for lag in lags], axis=2)
                     for t in range(lead_in, lead_in + n)])


def _predict(site, run: str, inputs: np.ndarray) -> np.ndarray:
    model, _ = load_run_model(site.runs / run)
    with torch.no_grad():
        return model(torch.from_numpy(inputs))[:, 0].double().numpy()


def _ort(path, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """onnxruntime's CPU provider on one thread, 64 stacks at a time, written apart from the
    builder: the run the expected outputs must come from, bit for bit."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    outputs = [session.run(None, {"frames": np.ascontiguousarray(inputs[s:s + 64])}) for s in range(0, len(inputs), 64)]
    return np.concatenate([a[:, 0] for a, _ in outputs]), np.concatenate([m for _, m in outputs])


def _camera(path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).astype(np.float64)


def _mad(a, b) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).mean())


# --- what is written --------------------------------------------------------------------

def test_every_file_the_page_loads_is_written_and_nothing_else(site):
    sheets = math.ceil((max(site.clips["c2018"]["frame_lags"]) + CLIP) / 60)
    pair_frames = sorted({f for p in site.leakage["pairs"]
                          for f in (p["frame"], p["leaky"]["nn_frame"], p["blocked"]["nn_frame"])})
    expected = sorted(
        ["models/sightline.onnx", "models/sightline-shuffled.onnx", "data/results.json",
         "data/selftest/frames.bin", "data/selftest/expected.json", "data/leakage/pairs.json"]
        + [f"data/leakage/frame-{f}.jpg" for f in pair_frames]
        + [f"data/clips/{cid}/{name}" for cid in TITLES
           for name in ["clip.json", *(["clip.mp4"] if site.video else []),
                        *(f"sprite-{k:02d}.jpg" for k in range(sheets))]])
    assert _files(site.out) == expected


def test_every_file_a_json_file_names_exists(site):
    named = [site.out / path for path in site.results["assets"]["models"].values()]
    named += [site.out / path for path in site.results["assets"]["clips"].values()]
    named += [site.out / site.results["assets"]["leakage_pairs"]]
    named += [site.out / path for path in site.results["assets"]["selftest"].values()]
    for cid, clip in site.clips.items():
        named += [_clip_dir(site, cid) / name for name in clip["sprite"]["files"]]
        if site.video:
            named.append(_clip_dir(site, cid) / clip["video"])
    leakage_dir = site.out / "data" / "leakage"
    for pair in site.pairs["pairs"]:
        named += [leakage_dir / pair["image"]] + [leakage_dir / pair[side]["image"] for side in ("leaky", "blocked")]
    assert len(named) > 10
    assert [str(p) for p in named if not p.is_file()] == []


def test_the_models_are_the_shipped_run_and_its_twin_each_under_2_mb(site):
    for key, run in (("sightline", site.shipped), ("shuffled", site.twin)):
        path = site.out / site.results["assets"]["models"][key]
        data = path.read_bytes()
        entry = site.export["models"][key]
        assert len(data) <= 2_000_000
        assert (entry["run"], entry["bytes"], entry["sha256"]) == (run, len(data), hashlib.sha256(data).hexdigest())
        assert entry["checkpoint_sha256"] == site.run_reports[run]["checkpoint_sha256"]
        meta = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).get_modelmeta()
        tags = meta.custom_metadata_map
        assert (tags["run"], tags["model"], json.loads(tags["frame_lags"])) == (run, "sightlinenet", [2])
        # parity is measured on every clip input and every self-test input, against the shipped gate
        parity = entry["parity"]
        assert parity["frames"] == 2 * CLIP + 8
        assert (parity["angle_tolerance_deg"], parity["saliency_tolerance"]) == (5e-4, 1e-4)
        assert 0 <= parity["max_angle_diff_deg"] <= parity["angle_tolerance_deg"]
        assert 0 <= parity["max_saliency_diff"] <= parity["saliency_tolerance"]


# --- the clips --------------------------------------------------------------------------

def test_each_clip_brings_the_history_frames_of_its_first_frame(site):
    lags = site.run_reports[site.shipped]["frame_lags"]
    for cid, clip in site.clips.items():
        assert (clip["id"], clip["title"], clip["fps"], clip["n"]) == (cid, TITLES[cid], 17.5, CLIP)
        assert clip["frame_lags"] == lags and clip["lead_in"] == max(lags) == 2
        assert len(clip["labels_deg"]) == len(clip["expected"]["sightline"]) == len(clip["expected"]["shuffled"]) == CLIP
        sprite = clip["sprite"]
        assert (sprite["frames_per_sheet"], sprite["frame_w"], sprite["frame_h"]) == (60, 200, 66)
        assert sprite["files"] == [f"sprite-{k:02d}.jpg" for k in range(math.ceil((clip["lead_in"] + CLIP) / 60))]
        assert len(sprite["decoded_sha256"]) == clip["lead_in"] + CLIP
        assert clip["rule"] == site.export["clips"][cid]["rule"]  # the page can say how it was chosen
        assert clip["tolerance"] == {"angle_deg": 1e-3}
        for key in ("twin_trained_frac", "twin_neighbour_frac"):
            assert clip[key] == site.export["clips"][cid][key]
    assert (site.clips["c2018"]["drive"], site.clips["c2017"]["drive"]) == ("drive2018", "drive2017")


def test_the_sprites_are_q95_444_sheets_of_up_to_60_frames_stacked_top_to_bottom(site):
    for cid, clip in site.clips.items():
        total = clip["lead_in"] + CLIP
        for k, name in enumerate(clip["sprite"]["files"]):
            with Image.open(_clip_dir(site, cid) / name) as image:
                assert image.format == "JPEG" and image.mode == "RGB"
                assert image.size == (200, 66 * min(60, total - 60 * k))
                assert JpegImagePlugin.get_sampling(image) == 0  # 4:4:4: no chroma subsampling


def test_decoding_the_sprites_gives_the_recorded_hashes(site):
    for cid, clip in site.clips.items():
        frames = _decoded_clip(site, cid)
        assert [hashlib.sha256(f.tobytes()).hexdigest() for f in frames] == clip["sprite"]["decoded_sha256"]


def test_the_sprites_hold_the_cache_frames_of_the_clip(site):
    for cid, clip in site.clips.items():
        cache = site.caches[clip["drive"]]
        frames = _decoded_clip(site, cid)
        start = clip["first_frame"] - clip["lead_in"]
        rows = np.asarray(cache.inputs[start:start + len(frames)])
        assert _mad(frames, rows) < 2.0  # JPEG q95 changes a pixel by a level or so
        assert not np.array_equal(frames, rows)  # ...but it does change them
        # each frame is far nearer its own row than the row five frames earlier
        assert _mad(frames[5:], rows[:-5]) > 2 * _mad(frames, rows)


def test_the_labels_are_the_cache_labels_of_the_clip_frames(site):
    for clip in site.clips.values():
        angles = site.caches[clip["drive"]].angles_deg
        first = clip["first_frame"]
        assert clip["labels_deg"] == [float(a) for a in angles[first:first + CLIP]]


def test_expected_outputs_come_from_the_written_onnx_files_on_the_decoded_sprites(site):
    models = {key: site.out / path for key, path in site.results["assets"]["models"].items()}
    for cid, clip in site.clips.items():
        decoded = _decoded_clip(site, cid)
        inputs = _stacks(decoded, clip["lead_in"], CLIP, clip["frame_lags"])
        for key, run in (("sightline", site.shipped), ("shuffled", site.twin)):
            expected = np.asarray(clip["expected"][key], dtype=np.float32)
            # the very bytes the page loads, run by onnxruntime as the builder must run it
            assert np.array_equal(expected, _ort(models[key], inputs)[0]), (cid, key)
            # and the PyTorch model agrees, the first output included (its history: the lead-in)
            fresh = _predict(site, run, inputs)
            np.testing.assert_allclose(expected, fresh, rtol=0, atol=1e-4, err_msg=f"{cid} {key}")
            assert abs(expected[0] - fresh[0]) <= 1e-4
        # JPEG changes the pixels enough for this tolerance to tell the sprites from the cache frames
        first = clip["first_frame"]
        cache = site.caches[clip["drive"]]
        rows = np.asarray(cache.inputs[first - clip["lead_in"]:first + CLIP])
        from_cache = _predict(site, site.shipped, _stacks(rows, clip["lead_in"], CLIP, clip["frame_lags"]))
        assert np.abs(np.asarray(clip["expected"]["sightline"]) - from_cache).max() > 1e-4


def test_each_clip_reports_its_own_mae_for_each_model_and_for_steering_straight(site):
    for cid, clip in site.clips.items():
        labels = np.asarray(clip["labels_deg"])
        mae = clip["clip_mae_deg"]
        for key in ("sightline", "shuffled"):
            assert mae[key] == pytest.approx(np.abs(np.asarray(clip["expected"][key]) - labels).mean(), abs=1e-9)
        assert mae["zero"] == pytest.approx(np.abs(labels).mean(), abs=1e-9)
        assert site.export["clips"][cid]["clip_mae_deg"] == mae


def test_the_2018_clip_lies_inside_one_blocked_test_stretch_far_from_training(site):
    clip = site.clips["c2018"]
    frames = np.arange(clip["first_frame"] - clip["lead_in"], clip["first_frame"] + CLIP)
    test = make_split("blocked", N, firewall=FIREWALL, block=BLOCK).test
    assert np.isin(frames, test).all()  # history frames included, and one stretch: no gap inside
    trained = np.load(site.runs / site.shipped / "train_idx.npy")
    gap = int(np.abs(frames[:, None] - trained[None, :]).min())
    assert gap >= 2 * FIREWALL + 1
    assert site.export["clips"]["c2018"]["min_frames_from_training"] == gap
    assert site.export["clips"]["c2017"]["min_frames_from_training"] is None  # another drive


def test_a_clip_closer_to_training_than_the_firewall_allows_is_refused():
    trained = np.r_[np.arange(0, 100), np.arange(300, 400)]
    # frames 140..259 sit 41 frames from training frame 99 and from training frame 300
    assert site_data.min_gap_to_training(np.arange(140, 260), trained, minimum=41) == 41
    with pytest.raises(ValueError, match="frame 139 is 40 frames from a training frame"):
        site_data.min_gap_to_training(np.arange(139, 250), trained, minimum=41)


def _rule(angles: np.ndarray, frames: np.ndarray, n: int, lead_in: int, cap: float | None = 90.0) -> int:
    """The first scored frame of the n-frame window, with its lead-in, inside one stretch of
    consecutive ``frames`` and with every one of those labels within ``cap`` deg either way,
    whose labels vary most (population sd); windows start every 10 frames from the first
    possible one in each stretch; ties go to the earliest."""
    frames = np.sort(frames)
    cuts = np.flatnonzero(np.diff(frames) > 1) + 1
    best = None
    for stretch in np.split(frames, cuts):
        for first in range(stretch[0] + lead_in, stretch[-1] - n + 2, 10):
            if cap is not None and np.abs(angles[first - lead_in:first + n].astype(np.float64)).max() > cap:
                continue
            sd = float(np.std(angles[first:first + n].astype(np.float64)))
            if best is None or sd > best[1]:
                best = (first, sd)
    return best[0]


def test_each_clip_is_the_window_its_rule_picks_from_the_labels(site):
    clip = site.clips["c2018"]
    test = make_split("blocked", N, firewall=FIREWALL, block=BLOCK).test
    angles = site.caches["drive2018"].angles_deg
    assert clip["first_frame"] == _rule(angles, test, CLIP, clip["lead_in"])
    # the spike sits in the windows that would vary most: the limit changes the choice
    assert _rule(angles, test, CLIP, clip["lead_in"], cap=None) != clip["first_frame"]
    shown = angles[clip["first_frame"] - clip["lead_in"]:clip["first_frame"] + CLIP]
    assert np.abs(shown).max() <= 90 < SPIKE_DEG == float(angles[SPIKE_ROW])
    clip = site.clips["c2017"]
    angles = site.caches["drive2017"].angles_deg
    assert clip["first_frame"] == _rule(angles, np.arange(N_CROSS), CLIP, clip["lead_in"])
    for cid, clip in site.clips.items():
        entry = site.export["clips"][cid]
        assert (entry["drive"], entry["first_frame"], entry["n"], entry["lead_in"]) == \
            (clip["drive"], clip["first_frame"], CLIP, clip["lead_in"])
        assert "standard deviation" in entry["rule"] and "every 10 frames" in entry["rule"]
        assert "within 90 degrees either way" in entry["rule"] and "road following" in entry["rule"]
    assert "blocked test" in site.export["clips"]["c2018"]["rule"]


def test_each_clip_says_how_much_of_it_the_twin_trained_on(site):
    trained = np.load(site.runs / site.twin / "train_idx.npy")
    clip = site.clips["c2018"]
    frames = np.arange(clip["first_frame"], clip["first_frame"] + CLIP)
    nearest = np.abs(frames[:, None] - trained[None, :]).min(axis=1)
    assert clip["twin_trained_frac"] == np.isin(frames, trained).mean() == pytest.approx(0.7, abs=0.15)
    assert clip["twin_neighbour_frac"] == (nearest <= 1).mean() > clip["twin_trained_frac"]
    # the twin trained on the 2018 drive only: none of the 2017 clip
    assert (site.clips["c2017"]["twin_trained_frac"], site.clips["c2017"]["twin_neighbour_frac"]) == (0.0, 0.0)


def test_twin_exposure_counts_frames_trained_on_and_their_neighbours():
    exposure = site_data.twin_exposure(np.arange(10, 20), np.array([10, 11, 15, 30]))
    # trained on 10, 11, 15; within one frame of one of them: 10, 11, 12, 14, 15, 16
    assert exposure == {"twin_trained_frac": 0.3, "twin_neighbour_frac": 0.6}
    assert site_data.twin_exposure(np.arange(5), np.empty(0, dtype=np.int64)) == \
        {"twin_trained_frac": 0.0, "twin_neighbour_frac": 0.0}


def test_each_clip_gives_the_times_of_its_first_and_last_scored_frames(site):
    clip = site.clips["c2018"]
    stamps = site.labels["drive2018"].timestamps_s
    first, last = clip["first_frame"], clip["first_frame"] + CLIP - 1
    assert (clip["first_s"], clip["last_s"]) == \
        (round(stamps[first] - stamps[0], 3), round(stamps[last] - stamps[0], 3))
    assert (clip["first_s"], clip["last_s"]) == (round(first * 0.057, 3), round(last * 0.057, 3))  # 57 ms apart
    # the 2017 stand-in, like the 2017 drive, has no timestamps
    assert (site.clips["c2017"]["first_s"], site.clips["c2017"]["last_s"]) == (None, None)


def test_clip_times_are_seconds_since_the_first_frame_to_the_millisecond():
    stamps = 1530445184.912 + np.array([0.0, 0.057, 0.115, 0.171, 0.228, 0.3])
    assert stamps[2] - stamps[0] != 0.115  # epoch seconds leave float noise in their differences
    labels = _labels(np.zeros(6), stamps)
    assert site_data.clip_times(labels, 2, 3) == {"first_s": 0.115, "last_s": 0.228}
    assert site_data.clip_times(labels, 0, 6) == {"first_s": 0.0, "last_s": 0.3}
    assert site_data.clip_times(_labels(np.zeros(6)), 2, 3) == {"first_s": None, "last_s": None}
    for first, n in ((4, 3), (-1, 2), (0, 0)):
        with pytest.raises(ValueError, match="does not fit in the 6 frames of drive 'drivex'"):
            site_data.clip_times(labels, first, n)


def test_the_clips_play_at_the_rate_drive2018_was_recorded_at():
    # sightline report records the median interval between the drive's timestamps
    interval = json.loads((REPO / "reports" / "summary.json").read_text())["drives"]["drive2018"]["median_dt_s"]
    assert site_data.CLIP_FPS == pytest.approx(1 / interval, rel=0.005)
    drives = committed_drives()
    if drives is not None:  # the drive itself is here: its own timestamps give that interval
        assert drives["drive2018"].median_dt_s() == interval
    assert site_data.CLIP_FPS == inspect.signature(render_demo).parameters["fps"].default  # as the demo plays


@needs_ffmpeg
def test_each_clip_video_is_h264_at_17_5_fps_of_its_scored_camera_frames(site):
    for cid, clip in site.clips.items():
        path = _clip_dir(site, cid) / clip["video"]
        probe = json.loads(subprocess.run(
            [FFPROBE, "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
            capture_output=True, text=True, check=True).stdout)
        (stream,) = probe["streams"]  # one video stream, no audio
        assert (stream["codec_name"], stream["pix_fmt"]) == ("h264", "yuv420p")
        assert (stream["width"], stream["height"]) == tuple(clip["video_size"]) == (456, 256)
        assert clip["camera_size"] == [455, 256]
        assert stream["r_frame_rate"] == "35/2" and int(stream["nb_read_frames"]) == CLIP
        data = path.read_bytes()
        assert data.find(b"moov") < data.find(b"mdat")  # fast start: the index comes first
        entry = site.export["clips"][cid]["video"]
        assert (entry["bytes"], entry["sha256"]) == (len(data), hashlib.sha256(data).hexdigest())
        raw = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        video = np.frombuffer(raw, np.uint8).reshape(CLIP, 256, 456, 3)
        labels = site.labels[clip["drive"]]
        # the first and last scored frames, read through the label rows, against frames 10 away
        for k, other in ((0, 10), (CLIP - 1, CLIP - 11)):
            right = _camera(labels.frame_path(clip["first_frame"] + k))
            wrong = _camera(labels.frame_path(clip["first_frame"] + other))
            assert _mad(video[k, :, :455], right) < 0.5 * _mad(video[k, :, :455], wrong)
            assert _mad(video[k, :, 455], video[k, :, 454]) < 3  # the padding column repeats the last one


# --- the self-test fixture ----------------------------------------------------------------

def test_the_self_test_inputs_are_raw_cache_frames_of_the_2018_clip(site):
    clip, expected = site.clips["c2018"], site.selftest
    frames = np.asarray(expected["frames"])
    assert expected["shape"] == [8, 66, 200, 6] and expected["clip"] == "c2018"
    assert len(set(frames.tolist())) == 8
    assert (frames >= clip["first_frame"]).all() and (frames < clip["first_frame"] + CLIP).all()
    raw = (site.out / "data" / "selftest" / "frames.bin").read_bytes()
    cache = site.caches["drive2018"].inputs
    stacked = np.stack([np.concatenate([cache[f], cache[f - 2]], axis=2) for f in frames])
    assert raw == stacked.tobytes()


def test_the_self_test_expects_the_outputs_of_the_written_onnx_files(site):
    from sightline.saliency import SteeringWithSaliency

    expected = site.selftest
    assert expected["tolerance"] == {"angle_deg": 1e-3, "saliency": 1e-3}
    inputs = np.frombuffer((site.out / "data" / "selftest" / "frames.bin").read_bytes(), np.uint8)
    inputs = inputs.reshape(expected["shape"])
    for key, run in (("sightline", site.shipped), ("shuffled", site.twin)):
        angles = np.asarray(expected["angles"][key], dtype=np.float32)
        maps = np.frombuffer(base64.b64decode(expected["saliency_f32_b64"][key]), "<f4").reshape(8, 66, 200)
        ort_angle, ort_saliency = _ort(site.out / site.results["assets"]["models"][key], inputs)
        assert np.array_equal(angles, ort_angle) and np.array_equal(maps, ort_saliency), key
        model, _ = load_run_model(site.runs / run)
        with torch.no_grad():
            angle, saliency = SteeringWithSaliency(model).eval()(torch.from_numpy(inputs.copy()))
        np.testing.assert_allclose(angles, angle[:, 0].numpy(), rtol=0, atol=1e-4)
        np.testing.assert_allclose(maps, saliency.numpy(), rtol=0, atol=1e-4)


# --- the leakage pairs ------------------------------------------------------------------

def test_the_leakage_pairs_keep_the_frames_and_round_their_gaps_to_the_millisecond(site):
    assert site.pairs["drive"] == site.leakage["drive"] == "drive2018"
    assert site.pairs["median_frame_interval_s"] == site.leakage["dt_s"]
    assert len(site.pairs["pairs"]) == len(site.leakage["pairs"]) > 0
    rounded = 0
    for pair, source in zip(site.pairs["pairs"], site.leakage["pairs"]):
        assert (pair["frame"], pair["angle_deg"], pair["image"]) == \
            (source["frame"], source["angle_deg"], f"frame-{source['frame']}.jpg")
        for side in ("leaky", "blocked"):
            got, src = pair[side], source[side]
            assert {k: got[k] for k in ("nn_frame", "gap_frames", "pixel_rmse", "nn_angle_deg")} == \
                {k: src[k] for k in ("nn_frame", "gap_frames", "pixel_rmse", "nn_angle_deg")}
            assert got["image"] == f"frame-{src['nn_frame']}.jpg"
            assert got["gap_s"] == round(src["gap_s"], 3) and abs(got["gap_s"] - src["gap_s"]) < 5e-4
            rounded += got["gap_s"] != src["gap_s"]
    assert rounded > 0  # epoch-second timestamps leave float noise that the rounding removes


def test_each_leakage_image_is_the_camera_frame_of_its_label_row(site):
    labels = site.labels["drive2018"]
    names = sorted((site.out / "data" / "leakage").glob("frame-*.jpg"))
    assert names
    for path in names:
        frame = int(path.stem.split("-")[1])
        with Image.open(path) as image:
            assert (image.format, image.size) == ("JPEG", (455, 256))
        written = _camera(path)
        right = _camera(labels.frame_path(frame))
        wrong = _camera(site.data / "drive2018" / f"{frame}.jpg")  # the image of another row
        assert _mad(written, right) < 0.5 * _mad(written, wrong)


# --- results.json -----------------------------------------------------------------------

#: (where in results.json, which report, where in that report): every value under the
#: report's subtree must be in results.json, unchanged
COVERAGE = [
    ("ladder", "summary", "ladder"),
    ("paired_leakage", "summary", "paired_leakage"),
    ("shipped", "summary", "shipped"),
    ("twin", "summary", "twin"),
    ("regimes", "summary", "regimes"),
    ("versions", "summary", "versions"),
    ("drives", "summary", "drives"),
    ("train_config", "summary", "train_config"),
    ("legacy.eval", "legacy", "eval"),
    ("legacy.atan_cap", "legacy", "atan_cap"),
    ("legacy.port_parity", "legacy", "port_parity"),
    ("legacy.recipe", "legacy", "recipe"),
    ("legacy.trained_on", "legacy", "trained_on"),
    ("legacy.seen_first_frame", "legacy", "seen_first_frame"),
    ("legacy.splits", "legacy", "splits"),
    ("leakage.memoriser", "leakage", "memoriser"),
    ("leakage.paired", "leakage", "paired"),
    ("leakage.splits", "leakage", "splits"),
    ("leakage.drive", "leakage", "drive"),
    ("leakage.seed", "leakage", "seed"),
    ("leakage.median_frame_interval_s", "leakage", "dt_s"),
    ("cutoffs", "summary", "cutoffs"),
    ("baselines.run", "shipped_report", "run"),
    ("baselines.test.n", "shipped_report", "test.n"),
    ("baselines.test.model", "shipped_report", "test.model"),
    ("baselines.test.baselines", "shipped_report", "test.baselines"),
    ("baselines.cross_drive.drive", "shipped_report", "cross_drive.drive"),
    ("baselines.cross_drive.n", "shipped_report", "cross_drive.n"),
    ("baselines.cross_drive.model", "shipped_report", "cross_drive.model"),
    ("baselines.cross_drive.baselines", "shipped_report", "cross_drive.baselines"),
    # the settings and counts the run reports record (the rest of these fields is worked out from
    # the splits module and the drive labels, and recomputed by the tests that follow)
    ("split_rules.blocked.block", "shipped_report", "splits.block"),
    ("split_rules.blocked.firewall", "shipped_report", "splits.firewall"),
    ("split_rules.forward.firewall", "shipped_report", "splits.firewall"),
    ("split_sizes.blocked.drive", "shipped_report", "drive"),
    ("split_sizes.blocked.frames", "shipped_report", "drives.drive2018.frames"),
    ("split_sizes.blocked.train", "shipped_report", "splits.train_n"),
    ("split_sizes.blocked.val", "shipped_report", "splits.val_n"),
    ("split_sizes.blocked.test", "shipped_report", "splits.test_n"),
    ("split_sizes.leaky.drive", "twin_report", "drive"),
    ("split_sizes.leaky.frames", "twin_report", "drives.drive2018.frames"),
    ("split_sizes.leaky.train", "twin_report", "splits.train_n"),
    ("split_sizes.leaky.val", "twin_report", "splits.val_n"),
    ("split_sizes.leaky.test", "twin_report", "splits.test_n"),
]


def _at(tree, keys):
    """The value at ``keys`` (dict keys and list positions) under ``tree``."""
    for key in keys:
        tree = tree[key]
    return tree


def _leaves(tree, prefix=()):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _leaves(value, (*prefix, key))
    elif isinstance(tree, list):
        for i, value in enumerate(tree):
            yield from _leaves(value, (*prefix, i))
    else:
        yield prefix, tree


def _sources(site) -> dict:
    return {"summary": site.summary, "legacy": site.legacy, "leakage": site.leakage,
            "shipped_report": site.run_reports[site.shipped], "twin_report": site.run_reports[site.twin]}


def _derived_fields(site) -> dict:
    """The split rules and the split sizes results.json holds, worked out here from the splits
    module and the fixture's settings (a blocked and a shuffled run of seed 0) rather than read
    from the run reports."""
    shares = {"train_frac": splits.TRAIN_FRAC, "val_frac": splits.VAL_FRAC, "test_frac": 0.15}
    rules = {
        "blocked": {"label": "blocked", "block": BLOCK, "firewall": FIREWALL, "pattern": splits.BLOCK_PATTERN},
        "forward": {"label": "later in the drive", "firewall": FIREWALL, **shares},
        "leaky": {"label": "shuffled", **shares},
    }
    sizes = {}
    for protocol, label in (("blocked", "blocked"), ("leaky", "shuffled")):
        counts = HAND_COUNTS[protocol]
        sizes[protocol] = {"label": label, "drive": "drive2018", "frames": N, **counts,
                           "dropped": N - sum(counts.values()), "runs": 1, "seeds": [0]}
    return {"split_rules": rules, "split_sizes": sizes}


def assert_profile_matches(profile: dict, labels: dict) -> None:
    """``profile`` (a results.json drive_profile) holds the repaired ``labels`` of each drive
    averaged over its buckets, recomputed here with numpy: angles within the rounding to 0.01,
    times within the millisecond, everything else exactly."""
    bucket = profile["bucket_frames"]
    assert sorted(profile["drives"]) == sorted(labels)
    for name, drive in labels.items():
        got, angles = profile["drives"][name], drive.angles_deg
        starts = list(range(0, len(angles), bucket))
        assert got["frames"] == len(angles) and got["start_frame"] == starts, name
        assert all(type(s) is int for s in got["start_frame"])
        assert len(got["mean_deg"]) == len(got["mean_abs_deg"]) == len(starts), name
        for k, start in enumerate(starts):
            chunk = angles[start:start + bucket]
            assert got["mean_deg"][k] == pytest.approx(chunk.mean(), abs=0.005 + 1e-9), (name, k)
            assert got["mean_abs_deg"][k] == pytest.approx(np.abs(chunk).mean(), abs=0.005 + 1e-9), (name, k)
        if drive.timestamps_s is None:
            assert got["start_s"] is None and got["duration_s"] is None, name
        else:
            stamps = drive.timestamps_s
            assert got["start_s"] == pytest.approx([stamps[s] - stamps[0] for s in starts], abs=0.0005 + 1e-9)
            assert got["duration_s"] == pytest.approx(stamps[-1] - stamps[0], abs=0.0005 + 1e-9)
            assert all(round(v, 3) == v for v in [*got["start_s"], got["duration_s"]]), name
        for key in ("mean_deg", "mean_abs_deg"):  # hundredths of a degree, and no negative zero
            for v in got[key]:
                assert type(v) is float and round(v, 2) == v and str(v) != "-0.0", (name, key, v)


def test_results_json_copies_the_reports_value_for_value(site):
    results, sources = site.results, _sources(site)
    covered = set()
    for where, report, path in COVERAGE:
        target = tuple(where.split("."))
        for leaf, value in _leaves(_at(sources[report], path.split("."))):
            got = _at(results, (*target, *leaf))
            assert got == value and type(got) is type(value), (where, leaf)
            covered.add((*target, *leaf))
    # the groups keep each statistic's mean and sd over seeds, and nothing is averaged again
    for key, group in site.summary["groups"].items():
        for name, value in group.items():
            target = ("groups", key, name)
            if isinstance(value, dict) and set(value) == {"mean", "sd", "values"}:
                assert results["groups"][key][name] == {"mean": value["mean"], "sd": value["sd"]}
                covered |= {(*target, "mean"), (*target, "sd")}
            else:
                assert results["groups"][key][name] == value
                covered |= {(*target, *leaf) for leaf, _ in _leaves(value)}
    # the temporal gaps keep everything but median_s, a frame count times the median interval
    for protocol, sides in site.leakage["gaps"].items():
        for side, gaps in sides.items():
            assert results["leakage"]["gaps"][protocol][side] == {k: v for k, v in gaps.items() if k != "median_s"}
            covered |= {("leakage", "gaps", protocol, side, k) for k in gaps if k != "median_s"}
    assert not any("median_s" in leaf for leaf, _ in _leaves(results))
    # the split rules and sizes are what the splits module makes from the recorded settings
    for key, tree in _derived_fields(site).items():
        assert results[key] == tree, key
        covered |= {(key, *leaf) for leaf, _ in _leaves(tree)}
    # the drive profile is the repaired labels averaged over buckets, recomputed here with numpy
    assert_profile_matches(results["drive_profile"], site.labels)
    covered |= {("drive_profile", *leaf) for leaf, _ in _leaves(results["drive_profile"])}
    # the timing is the drives' timestamps worked out again here, apart from the builder, with the
    # shipped run's frame lags and split settings
    shipped = site.run_reports[site.shipped]
    expected = expected_timing(site.labels, shipped["frame_lags"], shipped["splits"], trained_on=shipped["drive"])
    assert leaf_differences(results["timing"], expected) == []
    covered |= {("timing", *leaf) for leaf, _ in _leaves(results["timing"])}
    # the export parity is export.json's, value for value, under the page's names
    for key, run in (("sightline", site.shipped), ("shuffled", site.twin)):
        checked = site.export["models"][key]["parity"]
        assert results["parity"][key] == {
            "run": run, "stacks": checked["frames"], "max_abs_angle_deg": checked["max_angle_diff_deg"],
            "max_abs_saliency": checked["max_saliency_diff"], "angle_tolerance_deg": checked["angle_tolerance_deg"],
            "saliency_tolerance": checked["saliency_tolerance"],
            "onnxruntime": site.export["versions"]["onnxruntime"]} and site.export["models"][key]["run"] == run
        assert results["parity"][key]["onnxruntime"] == ort.__version__
        covered |= {("parity", key, field) for field in results["parity"][key]}
    numeric = {leaf for leaf, value in _leaves(results)
               if isinstance(value, (int, float)) and not isinstance(value, bool)}
    assert len(numeric) > 150
    assert numeric - covered == set()  # every number on the page comes from a report


def test_results_json_names_the_protocols_as_the_page_does(site):
    assert site.results["protocol_names"] == {"leaky": "shuffled", "blocked": "blocked",
                                              "forward": "later in the drive"}
    assert site.results["generated_from"] == [
        "reports/summary.json", f"reports/runs/{site.shipped}.json", f"reports/runs/{site.twin}.json",
        "reports/leakage.json", "reports/legacy.json"]


@pytest.mark.parametrize("tree, path, missing", [
    ({"a": {"b": 1}}, "a.b", None),
    ({"a": {"b": None}}, "a.b", None),  # a null value is there
    ({"a": {"b": 1}}, "a.c", "a.c"),
    ({"a": 1}, "a.b", "a.b"),  # not a mapping
    ({}, "a.b.c", "a"),
    ({"a": {"x": {"b": 1}, "y": {"c": 2}}}, "a.*.b", "a.y.b"),
    ({"a": [{"b": 1}, {"b": 2}, {}]}, "a.*.b", "a.2.b"),
    ({"a": []}, "a.*.b", "a.*"),
    ({"a": {}}, "a.*", "a.*"),
    ({"a": {"x": []}}, "a.*.*", "a.x.*"),
])
def test_missing_key_names_the_first_place_a_report_lacks_a_path(tree, path, missing):
    assert site_data.missing_key(tree, path) == missing


def _key_paths(tree, prefix=()):
    """The path of every key of every mapping under ``tree``; in a list, the first item's only."""
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield (*prefix, key)
            yield from _key_paths(value, (*prefix, key))
    elif isinstance(tree, list) and tree:
        yield from _key_paths(tree[0], (*prefix, 0))


def _an_entry_a_star_stands_for(path: tuple, patterns) -> bool:
    """Whether ``path`` names a whole entry of a mapping or list that some pattern of ``patterns``
    (``REQUIRED`` paths) walks with ``*``: a drive, a group, a model, rather than a field."""
    for pattern in patterns:
        parts = pattern.split(".")
        if len(path) <= len(parts) and parts[len(path) - 1] == "*" and \
                all(part in ("*", str(key)) for part, key in zip(parts, path)):
            return True
    return False


def test_every_key_the_site_reads_is_one_it_requires(site):
    """Every key of every report, taken away in turn. A field is either one ``REQUIRED`` lists, and
    the reports are refused for lacking it in one line, or one the site never reads, and every check
    and the results go through without it. A whole entry of a mapping the site walks (a drive, a
    group) may instead be refused in one line by the checks that match the reports to each other.
    Anything else (a KeyError, a TypeError, a refusal for another reason) is a key the site reads but
    ``REQUIRED`` does not list, and is named here."""
    reports = site_data.load_reports(site.reports / "summary.json", site.reports)
    sources = {"summary": reports.summary, "leakage": reports.leakage, "legacy": reports.legacy,
               "export": site.export, **{name: report for name, report in reports.runs.items()}}
    patterns = {"summary": site_data.REQUIRED["summary"][0], "leakage": site_data.REQUIRED["leakage"][0],
                "legacy": site_data.REQUIRED["legacy"][0], "export": site_data.REQUIRED["export"][0]}
    run_patterns = [*site_data.REQUIRED["run"][0], *site_data.REQUIRED["splits"][0], *site_data.REQUIRED["paired"][0]]
    unlisted, outcomes = [], {"refused": 0, "refused by a check": 0, "read nothing": 0}
    for source, report in sources.items():
        for path in _key_paths(report):
            changed = copy.deepcopy(report)
            _at(changed, path[:-1]).pop(path[-1])
            trial, export = reports, site.export
            if source == "export":
                export = changed
            elif source in ("summary", "leakage", "legacy"):
                trial = replace(reports, **{source: changed})
            else:
                trial = replace(reports, runs={**reports.runs, source: changed})
            where = f"{source}: {'.'.join(map(str, path))}"
            try:
                site_data.require_what_the_site_reads(trial)
                site_data._require(export, "export", "export.json")
            except ValueError as exc:
                if " lacks " in str(exc) and "\n" not in str(exc):
                    outcomes["refused"] += 1
                else:
                    unlisted.append(f"{where} ({exc})")
                continue
            try:
                site_data.site_results(trial, site.labels, export)
                site_data.leakage_pairs(trial.leakage)
                outcomes["read nothing"] += 1
            except ValueError as exc:
                entry = _an_entry_a_star_stands_for(path, patterns.get(source, run_patterns))
                if entry and "\n" not in str(exc):
                    outcomes["refused by a check"] += 1
                else:
                    unlisted.append(f"{where} (ValueError: {exc})")
            except Exception as exc:  # noqa: BLE001 - every other way of failing names an unlisted key
                unlisted.append(f"{where} ({type(exc).__name__}: {exc})")
    assert unlisted == [], "keys the site reads but REQUIRED does not list:\n" + "\n".join(unlisted)
    assert outcomes["refused"] > 300 and outcomes["refused by a check"] > 0 and outcomes["read nothing"] > 100, outcomes


def test_an_export_record_of_another_run_or_checkpoint_is_refused(site):
    reports = site_data.load_reports(site.reports / "summary.json", site.reports)
    for key, field, value in (("sightline", "run", site.twin), ("shuffled", "checkpoint_sha256", "0" * 64)):
        export = copy.deepcopy(site.export)
        export["models"][key][field] = value
        with pytest.raises(ValueError, match=f"export.json's {key} model is run .*; write the site data again with "
                                             "`sightline site-data`") as exc:
            site_data.site_results(reports, {}, export)
        assert "\n" not in str(exc.value)


#: characters no key or string of the site data may hold: the middle dot and the bullet, separators
#: the page never uses, and the thin and narrow no-break spaces, which the page's font subset lacks
BANNED_CHARACTERS = {"\u00b7": "middle dot", "\u2022": "bullet", "\u2009": "thin space",
                     "\u202f": "narrow no-break space"}


def banned_characters(tree, where: str = "") -> list[str]:
    """Each key or string of ``tree`` (a parsed JSON file) that holds one of ``BANNED_CHARACTERS``,
    as ``<where it is>: <which>``."""
    found = []
    if isinstance(tree, dict):
        for key, value in tree.items():
            found += [f"{where}.{key} (key): {name}" for char, name in BANNED_CHARACTERS.items() if char in key]
            found += banned_characters(value, f"{where}.{key}")
    elif isinstance(tree, list):
        for i, value in enumerate(tree):
            found += banned_characters(value, f"{where}[{i}]")
    elif isinstance(tree, str):
        found += [f"{where}: {name}" for char, name in BANNED_CHARACTERS.items() if char in tree]
    return found


def site_data_files(public: Path) -> list[Path]:
    """The JSON files of the site data under ``public``: results.json, every clip.json, the leakage
    pairs and the self-test's expected outputs."""
    data = public / "data"
    return [data / "results.json", *sorted((data / "clips").glob("*/clip.json")), data / "leakage" / "pairs.json",
            data / "selftest" / "expected.json"]


def test_the_character_check_finds_each_banned_character_in_a_key_or_a_string():
    assert set(BANNED_CHARACTERS) == {"\u00b7", "\u2022", "\u2009", "\u202f"}
    assert banned_characters({"a": ["x", {"b": "fine"}], "c": 1.5, "d": None}) == []
    for char, name in BANNED_CHARACTERS.items():
        assert banned_characters({"a": [f"x{char}y"]}) == [f".a[0]: {name}"]
        assert banned_characters({"a": {f"k{char}": 1}}) == [f".a.k{char} (key): {name}"]


def test_no_string_the_page_shows_holds_a_middle_dot_a_bullet_or_a_thin_space(site):
    # the protocols' labels and the block pattern the page prints are among the strings checked below
    def strings_of(tree):
        return {value for _, value in _leaves(tree) if isinstance(value, str)}

    assert strings_of(site.results["split_rules"]) == \
        {"blocked", "later in the drive", "shuffled", splits.BLOCK_PATTERN}
    assert strings_of(site.results["split_sizes"]) == {"blocked", "shuffled", "drive2018"}
    files = site_data_files(site.out)
    assert len(files) == 5 and all(path.is_file() for path in files)
    for path in files:
        assert banned_characters(json.loads(path.read_text())) == [], path


def _differing_leaves(a: dict, b: dict) -> list[str]:
    """The leaves (path to value, as ``_leaves`` gives them) at which ``a`` and ``b`` differ: in value,
    in type (an int is not a float) or in being there at all (an empty mapping is not a null)."""
    return sorted(".".join(map(str, leaf)) for leaf in a.keys() | b.keys()
                  if leaf not in a or leaf not in b or a[leaf] != b[leaf] or type(a[leaf]) is not type(b[leaf]))


def leaf_differences(got, want) -> list[str]:
    """The paths at which the parsed JSON trees ``got`` and ``want`` differ (``_differing_leaves``)."""
    return _differing_leaves(dict(_leaves(got)), dict(_leaves(want)))


def results_out_of_sync(results: dict, summary_path: Path, reports_dir: Path, drives: dict | None) -> list[str]:
    """The paths at which ``results`` differs from the results.json the reports, export.json and
    the drives' repaired labels (``drives``: a drive's name to its labels) give now. Where the
    drives are not on this machine ``drives`` is None: the drive profile and the timing, which only
    they can recompute, are then left out of the comparison, apart from the profile's bucket size
    (``timing_problems`` checks the timing against the rest of results.json instead)."""
    export = json.loads((reports_dir / "export.json").read_text())
    fresh = site_data.site_results(site_data.load_reports(summary_path, reports_dir), drives or {}, export)
    a, b = dict(_leaves(results)), dict(_leaves(fresh))
    if drives is None:
        a, b = ({leaf: v for leaf, v in tree.items()
                 if leaf[:2] != ("drive_profile", "drives") and leaf[:1] != ("timing",)} for tree in (a, b))
    return _differing_leaves(a, b)


def test_the_sync_check_finds_a_value_that_drifted_from_its_report(site):
    summary = site.reports / "summary.json"
    assert results_out_of_sync(site.results, summary, site.reports, site.labels) == []
    assert results_out_of_sync(site.results, summary, site.reports, None) == []
    drifted = copy.deepcopy(site.results)
    drifted["paired_leakage"]["sightlinenet"]["ratio"]["mean"] += 1e-9
    assert results_out_of_sync(drifted, summary, site.reports, site.labels) == \
        ["paired_leakage.sightlinenet.ratio.mean"]


def test_the_sync_check_finds_a_drifted_drive_profile_split_rule_or_split_size(site):
    summary = site.reports / "summary.json"
    drifts = {
        ("drive_profile", "bucket_frames"): 1,
        ("drive_profile", "drives", "drive2018", "mean_deg", 3): 0.01,
        ("drive_profile", "drives", "drive2018", "start_s", 2): 0.001,
        ("drive_profile", "drives", "drive2018", "duration_s"): 0.001,
        ("drive_profile", "drives", "drive2017", "mean_abs_deg", 0): 0.01,
        ("drive_profile", "drives", "drive2017", "start_frame", 1): 1,
        ("split_rules", "blocked", "block"): 1,
        ("split_rules", "leaky", "train_frac"): 0.01,
        ("split_sizes", "blocked", "train"): 1,
        ("split_sizes", "leaky", "dropped"): 1,
        ("split_sizes", "blocked", "seeds", 0): 1,
        ("parity", "sightline", "max_abs_angle_deg"): 1e-9,
        ("parity", "shuffled", "stacks"): 1,
        ("timing", "drive2018", "frame_interval_s", "p95"): 0.001,
        ("timing", "drive2018", "lag_gap_s", 0, "median"): 0.001,
        ("timing", "drive2018", "lag_gap_s", 0, "lag"): 1,
        ("timing", "drive2018", "closest_eval_s", "blocked", "val", "eval_frame"): 1,
        ("timing", "drive2018", "closest_eval_s", "forward", "test", "gap_s"): 0.001,
    }
    for path, delta in drifts.items():
        drifted = copy.deepcopy(site.results)
        _at(drifted, path[:-1])[path[-1]] += delta
        assert results_out_of_sync(drifted, summary, site.reports, site.labels) == [".".join(map(str, path))], path
    drifted = copy.deepcopy(site.results)
    drifted["split_rules"]["blocked"]["pattern"] = "TTVTTEV"
    drifted["split_sizes"]["leaky"]["label"] = "random"
    assert results_out_of_sync(drifted, summary, site.reports, site.labels) == \
        ["split_rules.blocked.pattern", "split_sizes.leaky.label"]
    # the drive without timestamps has a null timing, not an empty one
    drifted = copy.deepcopy(site.results)
    drifted["timing"]["drive2017"] = {}
    assert results_out_of_sync(drifted, summary, site.reports, site.labels) == ["timing.drive2017"]
    # a drive profile or a timing whose drives are not on this machine cannot be recomputed there,
    # but the profile's bucket size, the split rules and the split sizes still can
    for path, delta in drifts.items():
        drifted = copy.deepcopy(site.results)
        _at(drifted, path[:-1])[path[-1]] += delta
        found = results_out_of_sync(drifted, summary, site.reports, None)
        local = path[:2] == ("drive_profile", "drives") or path[0] == "timing"
        assert found == ([] if local else [".".join(map(str, path))]), path


def test_the_drive_profile_cannot_be_made_for_a_drive_of_another_length(site):
    reports = site_data.load_reports(site.reports / "summary.json", site.reports)
    short = replace(site.labels["drive2017"], filenames=site.labels["drive2017"].filenames[:-1],
                    angles_deg=site.labels["drive2017"].angles_deg[:-1])
    with pytest.raises(ValueError, match=f"drive 'drive2017' has {N_CROSS - 1} labelled frames, but summary.json "
                                         f"describes {N_CROSS}") as exc:
        site_data.site_results(reports, {"drive2017": short}, site.export)
    assert "\n" not in str(exc.value)
    with pytest.raises(ValueError, match="drive 'drive2019' has 1 labelled frames, but summary.json describes None"):
        site_data.site_results(reports, {"drive2019": DriveLabels(
            root=Path("drive2019"), filenames=["0.jpg"], angles_deg=np.zeros(1))}, site.export)


def sprite_hash_mismatches(clip_dir: Path) -> list[int]:
    """The sprite frames of a clip whose Pillow decode no longer has its recorded hash."""
    clip = json.loads((clip_dir / "clip.json").read_text())
    frames = _decode([clip_dir / name for name in clip["sprite"]["files"]])
    hashes = [hashlib.sha256(f.tobytes()).hexdigest() for f in frames]
    recorded = clip["sprite"]["decoded_sha256"]
    return [i for i in range(max(len(hashes), len(recorded)))
            if i >= len(hashes) or i >= len(recorded) or hashes[i] != recorded[i]]


def test_the_decode_check_finds_a_sprite_that_was_written_again(site, tmp_path):
    clip_dir = tmp_path / "c2018"
    shutil.copytree(_clip_dir(site, "c2018"), clip_dir)
    assert sprite_hash_mismatches(clip_dir) == []
    sheet = clip_dir / "sprite-01.jpg"
    with Image.open(sheet) as image:
        pixels = np.asarray(image.convert("RGB"))
    Image.fromarray(pixels).save(sheet, format="JPEG", quality=94, subsampling=0)
    assert sprite_hash_mismatches(clip_dir)[0] == 60  # the first frame of the second sheet


# --- the drive profile, the split rules and the split sizes -----------------------------------

def _labels(angles, stamps=None) -> DriveLabels:
    """The labels of a drive made of nothing but ``angles`` and, if given, ``stamps`` (epoch seconds)."""
    return DriveLabels(root=Path("drivex"), filenames=[f"{i}.jpg" for i in range(len(angles))],
                       angles_deg=np.asarray(angles, dtype=np.float64),
                       timestamps_s=None if stamps is None else np.asarray(stamps, dtype=np.float64))


def test_the_drive_profile_is_the_repaired_labels_averaged_over_buckets(site):
    profile = site.results["drive_profile"]
    assert profile["bucket_frames"] == site_data.BUCKET_FRAMES
    for name in ("drive2018", "drive2017"):
        row = DROPOUT_ROW[name]
        raw, repaired = site.raw_labels[name].angles_deg, site.labels[name].angles_deg
        assert raw[row] == 0.0 and abs(repaired[row]) > 15.0  # the drive logs a dropout the repair interpolates
        got = profile["drives"][name]
        assert got == site_data.drive_profile(site.labels[name])
        # the profile holds the repaired labels, not the raw ones: only the bucket of the dropout differs
        unrepaired = site_data.drive_profile(site.raw_labels[name])
        k = row // profile["bucket_frames"]
        assert got["mean_abs_deg"][k] > unrepaired["mean_abs_deg"][k]
        assert [i for i, (a, b) in enumerate(zip(got["mean_abs_deg"], unrepaired["mean_abs_deg"])) if a != b] == [k]


def test_only_the_timestamped_drive_has_a_time_axis(site):
    drives, bucket = site.results["drive_profile"]["drives"], site.results["drive_profile"]["bucket_frames"]
    stamped, unstamped = drives["drive2018"], drives["drive2017"]
    # the stand-in's frames are 57 ms apart
    assert stamped["start_s"] == [round(k * bucket * 0.057, 3) for k in range(len(stamped["start_frame"]))]
    assert stamped["duration_s"] == round((N - 1) * 0.057, 3) == 91.143
    # the 2017 stand-in has no timestamps, and its last bucket holds the frames that are left over
    assert unstamped["start_s"] is None and unstamped["duration_s"] is None
    assert unstamped["frames"] == N_CROSS and len(unstamped["start_frame"]) == math.ceil(N_CROSS / bucket)


def test_the_bucket_size_gives_about_a_thousand_points_for_the_2018_drive():
    frames = json.loads((REPO / "reports" / "summary.json").read_text())["drives"]["drive2018"]["frames"]
    assert 900 <= math.ceil(frames / site_data.BUCKET_FRAMES) <= 1100


def test_a_drive_profile_averages_each_bucket_and_the_last_one_holds_what_is_left():
    profile = site_data.drive_profile(_labels([4, -2, 6, 1, 1, -4, -10]), bucket_frames=3)
    assert profile == {"frames": 7, "start_frame": [0, 3, 6], "mean_deg": [2.67, -0.67, -10.0],
                       "mean_abs_deg": [4.0, 2.0, 10.0], "start_s": None, "duration_s": None}
    whole = site_data.drive_profile(_labels([1.0, 2.0, 3.0, 4.0]), bucket_frames=2)
    assert (whole["start_frame"], whole["mean_deg"]) == ([0, 2], [1.5, 3.5])  # a drive the buckets divide


def test_a_drive_profile_rounds_angles_to_hundredths_times_to_milliseconds_and_has_no_negative_zero():
    stamps = 1530445184.912 + 0.057 * np.arange(7)
    assert stamps[3] - stamps[0] != 0.171  # epoch seconds leave float noise in their differences
    profile = site_data.drive_profile(_labels([0.004, 0.004, 0.004, -0.004, -0.004, -0.004, 1.236], stamps), 3)
    assert profile["mean_deg"] == profile["mean_abs_deg"] == [0.0, 0.0, 1.24]
    assert [str(v) for v in profile["mean_deg"]] == ["0.0", "0.0", "1.24"]  # not -0.0 for the mean of -0.004
    assert profile["start_s"] == [0.0, 0.171, 0.342] and profile["duration_s"] == 0.342


def test_a_drive_profile_does_not_depend_on_the_order_a_bucket_is_added_in():
    values = [1e16, 1.0, -1e16]  # added left to right, the 1.0 is lost between the large values
    for order in itertools.permutations(values):
        assert site_data.drive_profile(_labels(order), 3)["mean_deg"] == [0.33]
    # the builtin sum of Python 3.12 on compensates its rounding and gets the bucket above right in
    # any order; this one defeats that compensation, and plain left-to-right addition too: only an
    # exactly rounded sum gives it one mean whatever the order
    hard = [3087808415424627.5, 3666406176012145.0, -4866849293776619.0, -3.2723348642660356e-17,
            5983383604037270.0]
    assert len({round(sum(order) / 5, 2) for order in itertools.permutations(hard)}) > 1
    means = {mean for order in itertools.permutations(hard)
             for mean in site_data.drive_profile(_labels(order), 5)["mean_deg"]}
    assert means == {round(math.fsum(hard) / 5, 2)}


def test_a_drive_profile_leaves_its_labels_alone_and_repeats_itself():
    labels = _labels([3.0, -1.0, 2.0, 5.0], [100.0, 101.0, 102.5, 103.0])
    angles, stamps = labels.angles_deg.copy(), labels.timestamps_s.copy()
    assert site_data.drive_profile(labels, 2) == site_data.drive_profile(labels, 2)
    assert np.array_equal(labels.angles_deg, angles) and np.array_equal(labels.timestamps_s, stamps)


def test_a_drive_profile_refuses_what_it_cannot_chart():
    stamps = [10.0, 10.057, 10.114, 10.1, 10.2]
    with pytest.raises(ValueError, match="drive 'drivex': frame 3 is timed before frame 2, so the drive has no "
                                         "time axis"):
        site_data.drive_profile(_labels(np.zeros(5), stamps), 2)
    equal = site_data.drive_profile(_labels(np.zeros(3), [10.0, 10.0, 10.1]), 2)
    assert equal["start_s"] == [0.0, 0.1]  # two frames at one instant are not backwards
    with pytest.raises(ValueError, match="a bucket needs at least 1 frame, got 0"):
        site_data.drive_profile(_labels([1.0]), 0)
    with pytest.raises(ValueError, match="a drive without frames has no profile"):
        site_data.drive_profile(_labels([]), 4)


def test_the_profile_is_refused_for_labels_the_reports_were_not_scored_on(site, tmp_path):
    reports = site_data.load_reports(site.reports / "summary.json", site.reports)
    assert site_data.site_results(reports, site.labels, site.export)["drive_profile"]["drives"].keys() == \
        {"drive2018", "drive2017"}
    # the label file as recorded, its dropouts left as zeros: the frames agree, the repairs do not
    with pytest.raises(ValueError, match="drive 'drive2017' has 0 repaired dropouts, but summary.json records 1; "
                                         "the profile is made from the labels as repair_dropouts leaves them") as exc:
        site_data.site_results(reports, {"drive2017": site.raw_labels["drive2017"]}, site.export)
    assert "\n" not in str(exc.value)
    # another label file with as many frames and repairs: one angle differs in its last digit
    edited = tmp_path / "drive2017"
    edited.mkdir()
    lines = (site.data / "drive2017" / "data.txt").read_text().splitlines()
    name, angle = lines[5].split()
    lines[5] = f"{name} {angle[:-1]}{'1' if angle[-1] != '1' else '2'}"
    (edited / "data.txt").write_text("\n".join(lines) + "\n")
    labels = repair_dropouts(load_labels(edited))
    assert (len(labels), labels.repaired_indices.size) == (N_CROSS, 1)
    with pytest.raises(ValueError, match=r"drive 'drive2017': its label file .*data\.txt \(sha256 [0-9a-f]{12}\) is "
                                         r"not the one the reports were made from \([0-9a-f]{12}\)"):
        site_data.site_results(reports, {"drive2017": labels}, site.export)


def test_the_split_rules_state_the_settings_given_and_the_splits_modules_pattern_and_fractions():
    rules = site_data.split_rules(block=123, firewall=45)
    shares = {"train_frac": splits.TRAIN_FRAC, "val_frac": splits.VAL_FRAC, "test_frac": 0.15}
    assert rules == {
        "blocked": {"label": "blocked", "block": 123, "firewall": 45, "pattern": splits.BLOCK_PATTERN},
        "forward": {"label": "later in the drive", "firewall": 45, **shares},
        "leaky": {"label": "shuffled", **shares},
    }
    # the rest of the drive is written as the 0.15 it is, not with the noise of 1 - 0.7 - 0.15
    assert [repr(rules[p]["test_frac"]) for p in ("forward", "leaky")] == ["0.15", "0.15"]
    assert {p: rules[p]["label"] for p in rules} == splits.PUBLIC_NAMES
    # and they are what a split is made with when nothing else is asked for
    n = 1000
    for protocol in ("leaky", "forward"):
        made = make_split(protocol, n, firewall=0)
        shown = rules[protocol]
        assert [made.train.size, made.val.size, made.test.size] == \
            [round(n * shown["train_frac"]), round(n * shown["val_frac"]), round(n * shown["test_frac"])]


def test_the_split_rules_of_the_fixture_are_its_recorded_settings(site):
    recorded = site.run_reports[site.shipped]["splits"]
    assert (recorded["block"], recorded["firewall"]) == (BLOCK, FIREWALL)  # not the splits module's defaults
    assert (BLOCK, FIREWALL) != (splits.BLOCK, splits.FIREWALL)
    assert site.results["split_rules"]["blocked"]["block"] == BLOCK
    assert site.results["split_rules"]["forward"]["firewall"] == FIREWALL


def test_the_split_rules_and_sizes_come_from_the_settings_every_run_shares(site):
    reports = site_data.load_reports(site.reports / "summary.json", site.reports)
    shipped = reports.runs[site.shipped]
    # later-in-the-drive runs made with a wider firewall, their counts the splits module's for it:
    # every run of the protocol agrees on its sizes, but not with the firewall the rules state
    wide = []
    for seed in (0, 1):
        report = _run_report(shipped, "forward", seed)
        made = make_split("forward", N, seed=seed, firewall=FIREWALL + 10, block=BLOCK)
        report["splits"].update(firewall=FIREWALL + 10, train_n=int(made.train.size), val_n=int(made.val.size),
                                test_n=int(made.test.size))
        wide.append(report)
    mixed = replace(reports, runs={**reports.runs, **{r["run"]: r for r in wide}})
    assert site_data.split_sizes(mixed)["forward"]["train"] == made.train.size != HAND_COUNTS["forward"]["train"]
    with pytest.raises(ValueError, match=re.escape(
            f"run 'drive2018_sightlinenet_forward_s0' made its splits with {{'firewall': {FIREWALL + 10}, "
            f"'block': {BLOCK}}}, but the shipped run with {{'firewall': {FIREWALL}, 'block': {BLOCK}}}")):
        site_data.site_results(mixed, {}, site.export)


def _blocks_swapped(protocol, n, seed=0, firewall=splits.FIREWALL, block=splits.BLOCK):
    """``make_split`` with the blocked split's validation and test blocks swapped (TTETTVT)."""
    if protocol == "blocked":
        return splits.blocked_split(n, block=block, firewall=firewall, pattern="TTETTVT")
    return make_split(protocol, n, seed=seed, firewall=firewall, block=block)


def test_a_split_pattern_that_keeps_the_sizes_is_found_by_the_paired_frames(site, monkeypatch):
    reports = site_data.load_reports(site.reports / "summary.json", site.reports)
    site_data._check_reports(reports)  # the splits the runs recorded: accepted

    def sizes(made):
        return made.train.size, made.val.size, made.test.size

    swapped = _blocks_swapped("blocked", N, firewall=FIREWALL, block=BLOCK)
    original = make_split("blocked", N, firewall=FIREWALL, block=BLOCK)
    assert sizes(swapped) == sizes(original) and not np.array_equal(swapped.test, original.test)
    monkeypatch.setattr(site_data, "make_split", _blocks_swapped)
    with pytest.raises(ValueError, match=rf"run '{site.shipped}' recorded \d+ paired frames \(sha256 [0-9a-f]{{12}}\), "
                                         rf"but the splits module pairs \d+ \(sha256 [0-9a-f]{{12}}\) of the same {N} "
                                         "frames for seed 0 with the same settings") as exc:
        site_data._check_reports(reports)
    assert "\n" not in str(exc.value)


def test_a_leakage_report_of_other_paired_frames_is_refused(site, tmp_path):
    reports = site_data.load_reports(site.reports / "summary.json", site.reports)
    leakage = copy.deepcopy(reports.leakage)
    leakage["paired"]["frames_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=r"leakage.json analyses \d+ paired frames \(sha256 000000000000\), but the "
                                         "splits module pairs .*; analyse the drive again with `sightline leakage`"):
        site_data._check_reports(replace(reports, leakage=leakage))


def _paired(seed: int, firewall: int = FIREWALL, block: int = BLOCK) -> dict:
    """The paired frames of the fixture's drive for ``seed``: their count and digest."""
    both = paired_frames(make_split("blocked", N, firewall=firewall, block=block), make_split("leaky", N, seed=seed))
    return {"n": int(both.size), "frames_sha256": frames_sha256(both)}


def _run_report(shipped: dict, protocol: str, seed: int, model: str = "sightlinenet") -> dict:
    """A copy of the shipped run's report for another run of the fixture's drive, with the split
    sizes the splits module makes for it and, for a blocked or shuffled run, its paired frames."""
    report = copy.deepcopy(shipped)
    made = make_split(protocol, N, seed=seed, firewall=FIREWALL, block=BLOCK)
    report.update(run=f"drive2018_{model}_{protocol}_s{seed}", model=model, protocol=protocol, seed=seed)
    report["splits"].update(train_n=int(made.train.size), val_n=int(made.val.size), test_n=int(made.test.size))
    if protocol == "forward":
        report.pop("paired")
    else:
        report["paired"].update(seed=seed, **_paired(seed))
    return report


def _many_runs(shipped: dict) -> list[dict]:
    return [_run_report(shipped, "blocked", 1), _run_report(shipped, "blocked", 2, model="pilotnet"),
            _run_report(shipped, "forward", 0), _run_report(shipped, "forward", 1),
            _run_report(shipped, "leaky", 1)]


def test_the_split_sizes_of_a_protocol_are_what_every_one_of_its_runs_recorded(site):
    reports = site_data.load_reports(site.reports / "summary.json", site.reports)
    extra = _many_runs(reports.runs[site.shipped])
    sizes = site_data.split_sizes(replace(reports, runs={**reports.runs, **{r["run"]: r for r in extra}}))
    assert list(sizes) == ["leaky", "blocked", "forward"]  # in the protocols' order
    for protocol, label, runs, seeds in (("leaky", "shuffled", 2, [0, 1]), ("blocked", "blocked", 3, [0, 1, 2]),
                                         ("forward", "later in the drive", 2, [0, 1])):
        counts = HAND_COUNTS[protocol]
        assert sizes[protocol] == {"label": label, "drive": "drive2018", "frames": N, **counts,
                                   "dropped": N - sum(counts.values()), "runs": runs, "seeds": seeds}, protocol
    assert [sizes[p]["dropped"] for p in ("leaky", "blocked", "forward")] == [0, 160, 80]


@pytest.mark.parametrize("protocol, label", [("leaky", "shuffled"), ("blocked", "blocked"),
                                             ("forward", "later in the drive")])
def test_runs_of_one_protocol_that_recorded_other_counts_are_refused_naming_both(site, protocol, label):
    reports = site_data.load_reports(site.reports / "summary.json", site.reports)
    shipped = reports.runs[site.shipped]
    agree = [_run_report(shipped, protocol, seed) for seed in (1, 2, 3, 4)]
    other = _run_report(shipped, protocol, 5)
    other["splits"]["val_n"] += 1
    runs = {r["run"]: r for r in [*agree, other]}
    with pytest.raises(ValueError) as exc:
        site_data.split_sizes(replace(reports, runs=runs))
    message = str(exc.value)
    assert f"the run reports disagree on the {label} split's train, validation and test frames" in message
    expected = HAND_COUNTS[protocol]
    # the rarest value comes first, and a long list of runs is cut short
    assert message.index(other["run"]) < message.index(agree[0]["run"])
    assert f"{expected['train']}/{expected['val'] + 1}/{expected['test']} of the {N} frames of drive2018 in " \
           f"{other['run']}; {expected['train']}/{expected['val']}/{expected['test']} of the {N} frames of " \
           f"drive2018 in {agree[0]['run']}, {agree[1]['run']}, {agree[2]['run']} and 1 more" in message
    assert "\n" not in message


def test_split_sizes_refuse_counts_that_do_not_fit_the_drive_and_a_protocol_that_is_none_of_the_three(site):
    reports = site_data.load_reports(site.reports / "summary.json", site.reports)
    shipped = copy.deepcopy(reports.runs[site.shipped])
    shipped["splits"]["train_n"] = N
    with pytest.raises(ValueError, match=f"the blocked split's {N}/160/160 train/validation/test frames do not "
                                         f"fit in the {N} frames of drive2018"):
        site_data.split_sizes(replace(reports, runs={**reports.runs, site.shipped: shipped}))
    honest = _run_report(reports.runs[site.shipped], "blocked", 1)
    honest["protocol"] = "honest"
    with pytest.raises(ValueError, match=f"run {honest['run']!r} follows the protocol 'honest', which is not one "
                                         "of leaky, blocked, forward"):
        site_data.split_sizes(replace(reports, runs={**reports.runs, honest["run"]: honest}))
    nothing = copy.deepcopy(reports.runs[site.shipped])
    nothing["drives"].pop("drive2018")
    with pytest.raises(ValueError, match="run .* does not describe its training drive 'drive2018'; evaluate it "
                                         "again with `sightline eval --all`"):
        site_data.split_sizes(replace(reports, runs={**reports.runs, site.shipped: nothing}))


def drive_profile_problems(results: dict) -> list[str]:
    """What does not fit between the drive profile of ``results`` (a results.json) and the rest of
    it, which needs no drive on disk: a profile for every drive and no other, each drive's frames
    covered by buckets of the profile's size, a time axis for a drive with a median frame interval
    and for none else (starting at 0 and increasing, never past the drive's duration), and every
    bucket's mean absolute angle at least the size of its mean angle."""
    profile, problems = results["drive_profile"], []
    bucket = profile["bucket_frames"]
    if sorted(profile["drives"]) != sorted(results["drives"]):
        problems.append(f"profiles of {sorted(profile['drives'])}, drives {sorted(results['drives'])}")
    for name, info in results["drives"].items():
        got = profile["drives"].get(name)
        if got is None:
            continue
        starts = list(range(0, info["frames"], bucket))
        if got["frames"] != info["frames"] or got["start_frame"] != starts:
            problems.append(f"{name}: buckets do not cover its {info['frames']} frames")
        for key in ("mean_deg", "mean_abs_deg"):
            if len(got[key]) != len(starts):
                problems.append(f"{name}: {len(got[key])} values of {key}, {len(starts)} buckets")
        if any(abs(m) > a + 0.01 for m, a in zip(got["mean_deg"], got["mean_abs_deg"])):
            problems.append(f"{name}: a mean absolute angle is smaller than its mean angle")
        timed = info["median_dt_s"] is not None
        if (got["start_s"] is not None) != timed or (got["duration_s"] is not None) != timed:
            problems.append(f"{name}: {'no' if timed else 'a'} time axis for a drive "
                            f"{'with' if timed else 'without'} timestamps")
        elif timed:
            times = got["start_s"]
            if len(times) != len(starts) or times[0] != 0.0 or any(b <= a for a, b in zip(times, times[1:])) \
                    or times[-1] > got["duration_s"]:
                problems.append(f"{name}: its time axis is not increasing from 0 to its duration")
    return problems


def test_the_drive_profile_fits_the_rest_of_results_json(site):
    assert drive_profile_problems(site.results) == []
    for change in (
        lambda r: r["drive_profile"]["drives"]["drive2018"]["mean_deg"].pop(),
        lambda r: r["drive_profile"]["drives"]["drive2017"]["start_frame"].insert(1, 5),
        lambda r: r["drive_profile"]["drives"]["drive2017"].update(frames=N_CROSS + 1),
        lambda r: r["drive_profile"]["drives"]["drive2017"].update(start_s=[0.0] * 5, duration_s=1.0),
        lambda r: r["drive_profile"]["drives"]["drive2018"].update(duration_s=None),
        lambda r: r["drive_profile"]["drives"]["drive2018"]["start_s"].reverse(),
        lambda r: r["drive_profile"]["drives"]["drive2018"].update(duration_s=1.0),
        lambda r: r["drive_profile"]["drives"]["drive2018"]["mean_abs_deg"].__setitem__(2, 0.0),
        lambda r: r["drive_profile"]["drives"].pop("drive2017"),
        lambda r: r["drive_profile"]["drives"].update(drive2019=r["drive_profile"]["drives"]["drive2017"]),
    ):
        broken = copy.deepcopy(site.results)
        change(broken)
        assert drive_profile_problems(broken), change


# --- the frame timing -------------------------------------------------------------------------

#: what a spread of frame times gives
SPREAD = ("max", "mean", "median", "min", "p05", "p95")


def _whole_ms(stamps) -> list[int]:
    """Timestamps in seconds as whole milliseconds since the first, the resolution the label files
    record them at."""
    first = float(stamps[0])
    return [round((float(stamp) - first) * 1000) for stamp in stamps]


def _exact_percentile(values: list[int], p: int) -> Fraction:
    """The ``p``-th percentile of ``values``, interpolated linearly between the two nearest values in
    order (Hyndman and Fan's type 7, numpy's "linear"), in exact arithmetic."""
    ordered = sorted(values)
    h = Fraction(len(ordered) - 1) * Fraction(p, 100)
    lo = math.floor(h)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


def _exact_spread(values: list[int]) -> dict:
    """What a spread of ``values`` (whole milliseconds) must hold: each statistic worked out exactly,
    then rounded to the millisecond with an exact half going to the even one, in seconds."""
    def seconds(ms) -> float:
        return round(ms) / 1000  # round() of an int or a Fraction: exact, a half to the even one

    return {"min": seconds(min(values)), "max": seconds(max(values)),
            "mean": seconds(Fraction(sum(values), len(values))), "p05": seconds(_exact_percentile(values, 5)),
            "median": seconds(_exact_percentile(values, 50)), "p95": seconds(_exact_percentile(values, 95))}


def _exact_closest(ms: list[int], held, trained) -> dict:
    """The pair of a frame of ``held`` and a frame of ``trained`` closest in time by ``ms``, ties going
    to the pair fewest frames apart, then to the earliest frame of ``held``, then to the earliest of
    ``trained``. Found by sorting the training frames by their times and gathering every pair at the
    least time apart, never assuming that the times increase with the frames."""
    times = np.asarray(ms, dtype=np.int64)
    held, trained = np.asarray(held, dtype=np.int64), np.asarray(trained, dtype=np.int64)
    order = np.argsort(times[trained], kind="stable")
    by_time, frames_by_time = times[trained][order], trained[order]
    pos = np.searchsorted(by_time, times[held])
    below, above = by_time[np.clip(pos - 1, 0, by_time.size - 1)], by_time[np.clip(pos, 0, by_time.size - 1)]
    nearest = np.minimum(np.abs(times[held] - below), np.abs(above - times[held]))
    gap = int(nearest.min())
    pairs = []
    for frame in held[nearest == gap]:
        for target in {int(times[frame]) - gap, int(times[frame]) + gap}:
            lo, hi = np.searchsorted(by_time, target, "left"), np.searchsorted(by_time, target, "right")
            pairs += [(gap, abs(int(frame) - int(other)), int(frame), int(other)) for other in frames_by_time[lo:hi]]
    gap, _, frame, other = min(pairs)
    return {"eval_frame": frame, "gap_s": gap / 1000, "train_frame": other}


def expected_timing(drives: dict, lags, settings: dict, trained_on: str) -> dict:
    """results.json's ``timing``, worked out here apart from the builder: for each drive of ``drives``
    (a drive's name to its labels) with timestamps, the spread of its frame intervals and of the gaps
    of each of ``lags``, and, on the drive ``trained_on``, the closest validation and test frames of the
    blocked and later-in-the-drive splits made with ``settings`` (a ``firewall`` and a ``block``)."""
    timing = {}
    for name, labels in drives.items():
        if labels.timestamps_s is None:
            timing[name] = None
            continue
        ms = _whole_ms(labels.timestamps_s)
        closest = None
        if name == trained_on:
            closest = {}
            for protocol in ("blocked", "forward"):
                split = make_split(protocol, len(ms), firewall=settings["firewall"], block=settings["block"])
                closest[protocol] = {which: _exact_closest(ms, getattr(split, which), split.train)
                                     for which in ("val", "test")}
        timing[name] = {
            "closest_eval_s": closest,
            "frame_interval_s": _exact_spread([b - a for a, b in zip(ms, ms[1:])]),
            "lag_gap_s": [{"lag": lag, **_exact_spread([ms[i] - ms[i - lag] for i in range(lag, len(ms))])}
                          for lag in lags],
        }
    return timing


def _whole_ms_of(seconds) -> int | None:
    """``seconds`` as whole milliseconds when it is a float of 0 or more on the millisecond grid and not
    a negative zero, else None."""
    if type(seconds) is not float or seconds < 0 or math.copysign(1.0, seconds) < 0 or round(seconds, 3) != seconds:
        return None
    return round(seconds * 1000)


def _spread_problems(where: str, spread) -> list[str]:
    """What is wrong with a spread of frame times: its keys, a value that is not seconds of whole
    milliseconds, or values out of order."""
    if not isinstance(spread, dict) or sorted(spread) != sorted(SPREAD):
        return [f"{where}: not a spread of {', '.join(SPREAD)}"]
    ms = {key: _whole_ms_of(value) for key, value in spread.items()}
    if None in ms.values():
        return [f"{where}: {', '.join(k for k, v in sorted(ms.items()) if v is None)} not seconds to the millisecond"]
    if not (ms["min"] <= ms["p05"] <= ms["median"] <= ms["p95"] <= ms["max"] and ms["min"] <= ms["mean"] <= ms["max"]):
        return [f"{where}: its values are out of order"]
    return []


def _closest_problems(where: str, entry, split: SplitIndices, which: str, closest_frames: int,
                      interval_ms: tuple[int, int]) -> list[str]:
    """What is wrong with a closest evaluation frame, ``entry``, of ``split``'s ``which`` frames: it
    must pair one of them with a training frame, with neither a training frame nor another of them in
    between (either would make a closer pair), no nearer in frames than ``closest_frames`` (the leakage
    report's), and a time that ``interval_ms`` (the drive's least and largest frame intervals) allows
    for that many frames."""
    if not isinstance(entry, dict) or sorted(entry) != ["eval_frame", "gap_s", "train_frame"] or \
            type(entry["eval_frame"]) is not int or type(entry["train_frame"]) is not int or \
            _whole_ms_of(entry["gap_s"]) is None:
        return [f"{where}: not an evaluation frame, a training frame and seconds to the millisecond"]
    frame, trained, held = entry["eval_frame"], entry["train_frame"], getattr(split, which)
    if not (np.isin(frame, held) and np.isin(trained, split.train)):
        return [f"{where}: frame {frame} is not a {which} frame, or frame {trained} not a training frame"]
    problems = []
    between = np.arange(min(frame, trained) + 1, max(frame, trained))
    if np.isin(between, split.train).any() or np.isin(between, held).any():
        problems.append(f"{where}: a training frame or a {which} frame lies between frames {frame} and {trained}")
    apart = abs(frame - trained)
    if apart < closest_frames:
        problems.append(f"{where}: {apart} frames from training, nearer than the leakage report's {closest_frames}")
    least, largest = interval_ms
    if not apart * least <= _whole_ms_of(entry["gap_s"]) <= apart * largest:
        problems.append(f"{where}: {entry['gap_s']} s for {apart} frames, whose intervals run from {least} to "
                        f"{largest} ms")
    return problems


def timing_problems(results: dict, lags) -> list[str]:
    """What does not fit between the timing of ``results`` (a results.json) and the rest of it, which
    needs no drive on disk: a timing for every drive and no other; null for a drive without timestamps
    and only for one; spreads of seconds to the millisecond in order; a median interval the summary's,
    and a mean the drive profile's duration over its intervals; a gap for each of ``lags`` (the shipped
    run's frame lags) in turn, within ``lag`` times the least and the largest interval, with the mean
    the drive's duration allows for it; and, on the drive the runs trained on alone, the closest
    validation and test frames of the blocked and later-in-the-drive splits, each a pair of frames
    those splits make (``_closest_problems``)."""
    timing, problems = results["timing"], []
    if sorted(timing) != sorted(results["drives"]):
        problems.append(f"timing of {sorted(timing)}, drives {sorted(results['drives'])}")
    trained_on, rules = results["leakage"]["drive"], results["split_rules"]
    for name, info in results["drives"].items():
        if name not in timing:
            continue
        got, timed = timing[name], info["median_dt_s"] is not None
        if (got is not None) != timed:
            problems.append(f"{name}: {'no' if timed else 'a'} timing for a drive {'with' if timed else 'without'} "
                            "timestamps")
            continue
        if got is None:
            continue
        if not isinstance(got, dict) or sorted(got) != ["closest_eval_s", "frame_interval_s", "lag_gap_s"]:
            problems.append(f"{name}: not a timing")
            continue
        interval = got["frame_interval_s"]
        found = _spread_problems(f"{name}: frame_interval_s", interval)
        problems += found
        if found:
            continue
        least, largest = _whole_ms_of(interval["min"]), _whole_ms_of(interval["max"])
        if least > 0 and abs(interval["median"] - info["median_dt_s"]) > 0.0005 + 1e-9:
            problems.append(f"{name}: its median frame interval is not the summary's {info['median_dt_s']}")
        duration, frames = round(results["drive_profile"]["drives"][name]["duration_s"] * 1000), info["frames"]
        if round(Fraction(duration, frames - 1)) != _whole_ms_of(interval["mean"]):
            problems.append(f"{name}: its mean frame interval is not the drive's duration over its intervals")
        entries = got["lag_gap_s"]
        if not isinstance(entries, list) or [e.get("lag") if isinstance(e, dict) else None for e in entries] != list(lags):
            problems.append(f"{name}: lag gaps for other lags than the shipped run's {list(lags)}")
        else:
            for entry in entries:
                where, lag = f"{name}: lag_gap_s of lag {entry['lag']}", entry["lag"]
                spread = {key: value for key, value in entry.items() if key != "lag"}
                found = _spread_problems(where, spread)
                problems += found
                if found:
                    continue
                if not lag * least <= _whole_ms_of(spread["min"]) <= _whole_ms_of(spread["max"]) <= lag * largest:
                    problems.append(f"{where}: outside {lag} times the least and the largest frame interval")
                # the gaps add up to the times between the drive's first lag frames and its last lag frames:
                # the duration lag times over, less at most lag - 1 of the largest interval each time
                top = Fraction(lag * duration, frames - lag)
                bottom = Fraction(lag * duration - lag * (lag - 1) * largest, frames - lag)
                if not bottom - Fraction(1, 2) <= _whole_ms_of(spread["mean"]) <= top + Fraction(1, 2):
                    problems.append(f"{where}: its mean is not the drive's duration over its gaps")
        closest = got["closest_eval_s"]
        if name != trained_on:
            if closest is not None:
                problems.append(f"{name}: closest evaluation frames on a drive no run trained on")
            continue
        if not isinstance(closest, dict) or sorted(closest) != ["blocked", "forward"] or \
                any(not isinstance(sides, dict) or sorted(sides) != ["test", "val"] for sides in closest.values()):
            problems.append(f"{name}: not the closest validation and test frames of the blocked and "
                            "later-in-the-drive splits")
            continue
        for protocol, sides in closest.items():
            split = make_split(protocol, info["frames"], firewall=rules[protocol]["firewall"],
                               block=rules["blocked"]["block"])
            for which, entry in sides.items():
                problems += _closest_problems(f"{name}: closest {protocol} {which} frame", entry, split, which,
                                              results["leakage"]["gaps"][protocol][which]["min_frames"],
                                              (least, largest))
    return problems


def test_the_timing_gives_the_stand_ins_frame_times(site):
    timing = site.results["timing"]
    assert list(timing) == ["drive2017", "drive2018"]
    assert timing["drive2017"] is None  # the 2017 stand-in, like the 2017 drive, has no timestamps
    drive = timing["drive2018"]
    # the 2018 stand-in's frames are 57 ms apart: so is every pair of neighbours, and every pair the
    # shipped run's lag of 2 frames apart is 114 ms apart
    assert site.run_reports[site.shipped]["frame_lags"] == [2]
    assert drive["frame_interval_s"] == dict.fromkeys(SPREAD, 0.057)
    assert drive["lag_gap_s"] == [{"lag": 2, **dict.fromkeys(SPREAD, 0.114)}]
    # the 20-frame firewall leaves 41 frames between training and evaluation: 2.337 s. The blocked
    # validation and test blocks each have such a frame at both ends, and the earlier is given; the
    # later-in-the-drive test frames start 281 frames after the last training frame
    assert drive["closest_eval_s"] == {
        "blocked": {"val": {"eval_frame": 420, "gap_s": 2.337, "train_frame": 379},
                    "test": {"eval_frame": 1020, "gap_s": 2.337, "train_frame": 979}},
        "forward": {"val": {"eval_frame": 1140, "gap_s": 2.337, "train_frame": 1099},
                    "test": {"eval_frame": 1380, "gap_s": 16.017, "train_frame": 1099}},
    }
    shipped = site.run_reports[site.shipped]
    assert timing == expected_timing(site.labels, shipped["frame_lags"], shipped["splits"], shipped["drive"])
    # from the label file's timestamps through the loader, as the drive profile reads them
    assert site_data.drive_timing(load_labels(site.data / "drive2018"), [2]) == {**drive, "closest_eval_s": None}


def test_the_timing_fits_the_rest_of_results_json(site):
    lags = site.run_reports[site.shipped]["frame_lags"]
    assert timing_problems(site.results, lags) == []
    for change in (
        lambda t: t.update(drive2017={}),  # the drive without timestamps
        lambda t: t.update(drive2017=copy.deepcopy(t["drive2018"])),
        lambda t: t.update(drive2018=None),
        lambda t: t.pop("drive2017"),
        lambda t: t["drive2018"]["closest_eval_s"].update(blocked={
            "val": t["drive2018"]["closest_eval_s"]["blocked"]["test"],
            "test": t["drive2018"]["closest_eval_s"]["blocked"]["val"]}),
        lambda t: t["drive2018"]["closest_eval_s"]["forward"]["val"].update(eval_frame=1139),  # a dropped frame
        lambda t: t["drive2018"]["closest_eval_s"]["blocked"]["test"].update(train_frame=978),  # 979 is nearer
        lambda t: t["drive2018"]["closest_eval_s"]["forward"]["test"].update(eval_frame=1381),  # 1380 is nearer
        lambda t: t["drive2018"]["closest_eval_s"]["blocked"]["val"].update(gap_s=2.3),  # under 41 intervals
        lambda t: t["drive2018"]["closest_eval_s"]["blocked"]["val"].update(gap_s=2337),  # milliseconds
        lambda t: t["drive2018"]["closest_eval_s"]["blocked"]["val"].update(eval_frame=420.0),
        lambda t: t["drive2018"]["closest_eval_s"].pop("forward"),
        lambda t: t["drive2018"]["closest_eval_s"]["blocked"].pop("test"),
        lambda t: t["drive2018"]["frame_interval_s"].update(p05=0.058),  # above the median
        lambda t: t["drive2018"]["frame_interval_s"].update(median=0.0571),  # finer than a millisecond
        lambda t: t["drive2018"]["frame_interval_s"].update(min=-0.0),
        lambda t: t["drive2018"]["frame_interval_s"].pop("max"),
        lambda t: t["drive2018"]["lag_gap_s"][0].update(lag=1),  # another lag than the shipped run's
        lambda t: t["drive2018"]["lag_gap_s"].clear(),
        lambda t: t["drive2018"]["lag_gap_s"][0].update(min=0.057),  # under twice the least interval
        lambda t: t["drive2018"]["lag_gap_s"][0].update(max=0.2),  # over twice the largest
    ):
        broken = copy.deepcopy(site.results)
        change(broken["timing"])
        assert timing_problems(broken, lags), change
    # a mean or a median that no longer matches the drive profile or the summary, the spread kept in order
    for key, value in (("mean", 0.056), ("median", 0.056)):
        broken = copy.deepcopy(site.results)
        broken["timing"]["drive2018"]["frame_interval_s"].update({"min": 0.056, "p05": 0.056, key: value})
        assert timing_problems(broken, lags), key


def test_the_timing_measures_the_lags_the_shipped_run_records(site):
    """Never a lag typed in: the shipped run's report names them, and its twin's must agree."""
    reports = site_data.load_reports(site.reports / "summary.json", site.reports)
    runs = copy.deepcopy(reports.runs)
    for run in (site.shipped, site.twin):
        runs[run]["frame_lags"] = [1, 3]
    timing = site_data.site_results(replace(reports, runs=runs), site.labels, site.export)["timing"]["drive2018"]
    assert [(entry["lag"], entry["median"]) for entry in timing["lag_gap_s"]] == [(1, 0.057), (3, 0.171)]
    runs[site.shipped]["frame_lags"] = []  # a model of one frame: no lag to time
    runs[site.twin]["frame_lags"] = []
    assert site_data.site_results(replace(reports, runs=runs), site.labels, site.export)["timing"]["drive2018"][
        "lag_gap_s"] == []


def test_a_spread_is_measured_from_every_interval_and_every_lag_gap():
    # 14 frames whose 13 intervals are uneven and out of order in time; in order they run 20 30 31 38
    # 40 41 57 58 60 66 70 91 121 ms: the 5th percentile lies 0.6 of the way from the first to the
    # second (26), the median is the 7th, the 95th percentile lies 0.4 of the way from the 12th to the
    # 13th (103), and the mean is 723 / 13 = 55.6
    intervals = [57, 20, 91, 31, 40, 121, 38, 58, 30, 66, 41, 70, 60]
    stamps = 1530445184.912 + np.cumsum([0, *intervals]) / 1000
    labels = _labels(np.zeros(len(stamps)), stamps)
    timing = site_data.drive_timing(labels, [1, 3])
    assert timing["frame_interval_s"] == {"min": 0.02, "p05": 0.026, "median": 0.057, "mean": 0.056, "p95": 0.103,
                                          "max": 0.121}
    assert timing["lag_gap_s"][0] == {"lag": 1, **timing["frame_interval_s"]}
    # each gap of 3 frames sums 3 intervals in a row; in order they run 126 137 142 154 162 168 171 177
    # 192 199 217 ms, so the 5th and 95th percentiles fall halfway between two of them: 131.5 ms, whose
    # half goes to the even millisecond, and 208 ms; the mean is 1845 / 11 = 167.7
    assert timing["lag_gap_s"][1] == {"lag": 3, "min": 0.126, "p05": 0.132, "median": 0.168, "mean": 0.168,
                                      "p95": 0.208, "max": 0.217}
    assert timing["lag_gap_s"][1] == {"lag": 3, **_exact_spread([sum(intervals[i:i + 3]) for i in range(11)])}
    assert timing["closest_eval_s"] is None  # no splits given: a drive no run trained on
    # the epoch seconds' floating point noise never reaches the times: the same intervals from 0 s
    assert site_data.drive_timing(_labels(np.zeros(len(stamps)), np.cumsum([0, *intervals]) / 1000), [1, 3]) == timing


def test_a_spread_rounds_once_to_the_millisecond_an_exact_half_to_the_even_one():
    def spread(intervals):
        stamps = 1530445184.912 + np.cumsum([0, *intervals]) / 1000
        return site_data.drive_timing(_labels(np.zeros(len(stamps)), stamps), [])["frame_interval_s"]

    # 11 intervals: the 5th percentile is 20.5 ms, the 95th 91.5 ms, the mean 639 / 11 = 58.1 ms
    assert spread([20, 21, 50, 55, 57, 60, 61, 62, 70, 90, 93]) == \
        {"min": 0.02, "p05": 0.02, "median": 0.06, "mean": 0.058, "p95": 0.092, "max": 0.093}
    # two intervals: the median and the mean are 57.5 ms, then 56.5 ms
    assert (spread([57, 58])["median"], spread([57, 58])["mean"]) == (0.058, 0.058)
    assert (spread([56, 57])["median"], spread([56, 57])["mean"]) == (0.056, 0.056)
    # a value numpy leaves a hair to one side of a half is put back on it first
    assert [site_data._ms_to_s(v) for v in (66.49999999999999, 66.50000000000001, 67.49999999999999)] == \
        [0.066, 0.066, 0.068]
    # frames at one instant: intervals of 0.0, never a negative zero
    flat = site_data.drive_timing(_labels(np.zeros(4), np.full(4, 1530445184.912)), [2])
    assert json.dumps(flat).count("-0.0") == 0 and flat["frame_interval_s"] == dict.fromkeys(SPREAD, 0.0)


def _split(train, val, test) -> SplitIndices:
    return SplitIndices(train=np.asarray(train, dtype=np.int64), val=np.asarray(val, dtype=np.int64),
                        test=np.asarray(test, dtype=np.int64))


def _closest(ms: list[int], split: SplitIndices) -> dict:
    labels = _labels(np.zeros(len(ms)), 1530445184.912 + np.asarray(ms) / 1000)
    return site_data.drive_timing(labels, [], {"split": split})["closest_eval_s"]["split"]


def test_the_closest_evaluation_frame_is_the_nearest_in_time_not_in_frames():
    ms = [0, 100, 200, 300, 310, 320, 330, 400, 420, 450]
    split = _split(train=[0, 1, 9], val=[3, 6], test=[8])
    # validation frame 3 is 2 frames and 200 ms from training frame 1; frame 6 is 3 frames but only
    # 120 ms from training frame 9
    assert _closest(ms, split) == {"val": {"eval_frame": 6, "gap_s": 0.12, "train_frame": 9},
                                   "test": {"eval_frame": 8, "gap_s": 0.03, "train_frame": 9}}
    for which in ("val", "test"):
        assert _closest(ms, split)[which] == _exact_closest(ms, getattr(split, which), split.train)


def test_ties_go_to_the_fewest_frames_apart_then_the_earliest_evaluation_frame_then_the_earliest_training_frame():
    # frame 3 is 100 ms from training frame 1, 2 frames away, and from training frame 4, 1 frame away;
    # frame 6 is 100 ms and 1 frame from training frame 7, and 200 ms from training frame 4
    ms = [0, 100, 150, 200, 300, 400, 500, 600]
    split = _split(train=[0, 1, 4, 7], val=[3], test=[6])
    assert _closest(ms, split) == {"val": {"eval_frame": 3, "gap_s": 0.1, "train_frame": 4},
                                   "test": {"eval_frame": 6, "gap_s": 0.1, "train_frame": 7}}
    # evenly timed: validation frames 1 and 5 each sit 1 frame and 100 ms from two training frames, as
    # does test frame 3; the earliest validation frame, and each one's earliest training frame, are given
    ms = [0, 100, 200, 300, 400, 500, 600]
    split = _split(train=[0, 2, 4, 6], val=[1, 5], test=[3])
    assert _closest(ms, split) == {"val": {"eval_frame": 1, "gap_s": 0.1, "train_frame": 0},
                                   "test": {"eval_frame": 3, "gap_s": 0.1, "train_frame": 2}}
    for ms, split in (([0, 100, 150, 200, 300, 400, 500, 600], _split([0, 1, 4, 7], [3], [6])),
                      ([0, 100, 200, 300, 400, 500, 600], _split([0, 2, 4, 6], [1, 5], [3]))):
        for which in ("val", "test"):
            assert _closest(ms, split)[which] == _exact_closest(ms, getattr(split, which), split.train)


def test_the_timing_of_a_drive_without_timestamps_is_null_and_it_refuses_what_it_cannot_time():
    assert site_data.drive_timing(_labels(np.zeros(5)), [2], {"blocked": _split([0], [2], [4])}) is None
    with pytest.raises(ValueError, match="drive 'drivex': frame 3 is timed before frame 2, so its frames have no "
                                         "intervals in time"):
        site_data.drive_timing(_labels(np.zeros(5), [10.0, 10.057, 10.114, 10.1, 10.2]), [2])
    with pytest.raises(ValueError, match=r"drive 'drivex' has too few timed frames \(3\) to time frames 3 apart"):
        site_data.drive_timing(_labels(np.zeros(3), [10.0, 10.057, 10.1]), [3])
    with pytest.raises(ValueError, match=r"has too few timed frames \(1\) to time frames 1 apart"):
        site_data.drive_timing(_labels(np.zeros(1), [10.0]), [])
    for lag in (0, -1, True, 1.5, "2"):
        with pytest.raises(ValueError, match="a frame lag is a whole number of frames, 1 or more"):
            site_data.drive_timing(_labels(np.zeros(5), 10.0 + 0.057 * np.arange(5)), [lag])
    # and it leaves the labels alone and repeats itself
    labels = _labels(np.zeros(6), 10.0 + np.array([0.0, 0.03, 0.09, 0.1, 0.16, 0.2]))
    stamps = labels.timestamps_s.copy()
    assert site_data.drive_timing(labels, [2]) == site_data.drive_timing(labels, [2])
    assert np.array_equal(labels.timestamps_s, stamps)


# --- the same bytes every time --------------------------------------------------------------

def test_a_jpeg_of_the_same_pixels_is_the_same_bytes():
    pixels = np.random.default_rng(0).integers(0, 256, (132, 200, 3), dtype=np.uint8)
    first = site_data.jpeg_bytes(pixels, quality=95, subsampling=0)
    assert first == site_data.jpeg_bytes(pixels.copy(), quality=95, subsampling=0)
    assert first[:2] == b"\xff\xd8"


def test_a_second_build_writes_the_same_bytes(site):
    first, second = site.out, site.second / "public"
    assert _files(second) == [f for f in _files(first) if not f.endswith(".mp4")]
    for name in _files(second):
        a, b = (first / name).read_bytes(), (second / name).read_bytes()
        if name.endswith("clip.json"):
            a, b = json.loads(a), json.loads(b)
            assert b["video"] is None and b["video_size"] is None and b["camera_size"] == [455, 256]
            for d in (a, b):
                d.pop("video")
                d.pop("video_size")
        assert a == b, name
    one = json.loads((site.reports / "export.json").read_text())
    two = json.loads((site.second / "reports" / "export.json").read_text())
    for clip in two["clips"].values():
        assert clip.pop("video") is None
    for clip in one["clips"].values():
        clip.pop("video")
    one["versions"].pop("ffmpeg")
    two["versions"].pop("ffmpeg")
    assert one == two


# --- refusals -------------------------------------------------------------------------------

def _workspace_with(site, tmp_path, change) -> Workspace:
    """The fixture's runs and drives, copied where ``change`` may alter them."""
    runs = tmp_path / "runs"
    for run in (site.shipped, site.twin):
        shutil.copytree(site.runs / run, runs / run)
    (runs / "cache").symlink_to(site.runs / "cache")
    data = tmp_path / "data"
    data.mkdir()
    for drive in ("drive2018", "drive2017"):
        (data / drive).symlink_to(site.data / drive)
    change(runs, data)
    return Workspace(data_dir=data, runs_dir=runs)


def _build(site, tmp_path, workspace, reports=None, **kwargs):
    reports = reports or site.reports
    return site_data.build_site_data(reports / "summary.json", workspace, out_dir=tmp_path / "public",
                                     reports_dir=reports, clip_frames=CLIP, video=False, **kwargs)


def test_a_checkpoint_other_than_the_reported_one_is_refused(site, tmp_path):
    def retrain(runs, data):
        shutil.copy2(runs / site.twin / "best.pt", runs / site.shipped / "best.pt")

    with pytest.raises(UnusableRunError, match="checkpoint") as exc:
        _build(site, tmp_path, _workspace_with(site, tmp_path, retrain))
    assert site.shipped in str(exc.value)
    assert not (tmp_path / "public").exists()


def test_a_cache_of_other_labels_is_refused(site, tmp_path):
    def edit_labels(runs, data):
        (data / "drive2018").unlink()
        (data / "drive2018").mkdir()
        lines = (site.data / "drive2018" / "data.txt").read_text().splitlines()
        name, rest = lines[100].split(" ", 1)
        angle, stamp = rest.split(",", 1)
        lines[100] = f"{name} {angle[:-1]}{'1' if angle[-1] != '1' else '2'},{stamp}"
        (data / "drive2018" / "data.txt").write_text("\n".join(lines) + "\n")

    with pytest.raises(MissingDataError, match="sightline cache --drive drive2018"):
        _build(site, tmp_path, _workspace_with(site, tmp_path, edit_labels))
    assert not (tmp_path / "public").exists()


def test_a_leakage_report_of_other_splits_is_refused(site, tmp_path):
    reports = tmp_path / "reports"
    shutil.copytree(site.reports, reports)
    leakage = json.loads((reports / "leakage.json").read_text())
    leakage["splits"]["firewall"] = FIREWALL + 1
    (reports / "leakage.json").write_text(json.dumps(leakage))
    with pytest.raises(ValueError, match="leakage.json"):
        _build(site, tmp_path, _workspace_with(site, tmp_path, lambda runs, data: None), reports=reports)
    assert not (tmp_path / "public").exists()


def _reports_with(site, tmp_path, change) -> Path:
    """A copy of the fixture's reports in which ``change`` alters the summary (``"summary"``),
    the shipped run's or the twin's report (``"shipped"``, ``"twin"``), or the leakage or the
    standard-recipe report (``"leakage"``, ``"legacy"``)."""
    reports = tmp_path / "reports"
    shutil.copytree(site.reports, reports)
    files = {"summary": reports / "summary.json", "shipped": reports / "runs" / f"{site.shipped}.json",
             "twin": reports / "runs" / f"{site.twin}.json", "leakage": reports / "leakage.json",
             "legacy": reports / "legacy.json"}
    loaded = {key: json.loads(path.read_text()) for key, path in files.items()}
    change(loaded)
    for key, path in files.items():
        path.write_text(json.dumps(loaded[key]))
    return reports


def _untouched(runs, data):
    pass


@pytest.mark.parametrize("change, message", [
    (lambda r: r["shipped"].update(protocol="forward"), "trained on the later in the drive split, not the blocked one"),
    (lambda r: r["twin"].update(protocol="blocked"), "trained on the blocked split, not the shuffled one"),
    (lambda r: r["twin"].update(seed=1), "has seed 1, the shipped run seed 0"),
])
def test_a_twin_that_is_not_the_shipped_runs_shuffled_twin_is_refused(site, tmp_path, change, message):
    reports = _reports_with(site, tmp_path, change)
    with pytest.raises(ValueError, match=re.escape(message)) as exc:
        _build(site, tmp_path, _workspace_with(site, tmp_path, _untouched), reports=reports)
    assert "\n" not in str(exc.value)
    assert not (tmp_path / "public").exists()


def _add_one(tree, key):
    tree[key] += 1.0


@pytest.mark.parametrize("change, named", [
    (lambda r: _add_one(r["summary"]["groups"]["sightlinenet/blocked"]["test_mae_deg"]["values"], 0), "test_mae_deg"),
    (lambda r: _add_one(r["summary"]["groups"]["sightlinenet/leaky"]["val_mae_deg"]["values"], 0), "val_mae_deg"),
    (lambda r: _add_one(r["summary"]["paired_leakage"]["sightlinenet"]["per_seed"][0], "blocked_mae_deg"),
     "blocked paired MAE"),
    (lambda r: r["summary"]["regimes"]["models"]["sightlinenet"].update(run="drive2018_sightlinenet_blocked_s9"),
     "regimes"),
])
def test_a_summary_that_disagrees_with_the_run_reports_is_refused(site, tmp_path, change, named):
    reports = _reports_with(site, tmp_path, change)
    with pytest.raises(ValueError, match="sightline report") as exc:
        _build(site, tmp_path, _workspace_with(site, tmp_path, _untouched), reports=reports)
    assert named in str(exc.value) and "\n" not in str(exc.value)
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("change, named, command", [
    (lambda r: r["summary"].pop("regimes"), "'regimes'", "sightline report"),
    (lambda r: r["summary"].pop("train_config"), "'train_config'", "sightline report"),
    (lambda r: r["summary"].pop("runs"), "'runs'", "sightline report"),
    (lambda r: r["shipped"]["splits"].pop("block"), "'splits.block'", "sightline eval --all"),
    (lambda r: r["shipped"]["splits"].pop("train_n"), "'splits.train_n'", "sightline eval --all"),
    (lambda r: r["twin"].pop("drives"), "'drives'", "sightline eval --all"),
    (lambda r: r["summary"].pop("cutoffs"), "'cutoffs'", "sightline report"),
    (lambda r: r["summary"]["groups"]["sightlinenet/blocked"].pop("seeds"), "'groups.sightlinenet/blocked.seeds'",
     "sightline report"),
    (lambda r: r["summary"]["ladder"]["sightlinenet"][2].pop("zero_mae_deg"),
     "'ladder.sightlinenet.2.zero_mae_deg'", "sightline report"),
    (lambda r: r["summary"]["paired_leakage"]["sightlinenet"]["per_seed"].clear(),
     "'paired_leakage.sightlinenet.per_seed.*'", "sightline report"),
    (lambda r: r["shipped"]["test"]["zero_regimes"]["sharp"].pop("n"), "'test.zero_regimes.sharp.n'",
     "sightline eval --all"),
    (lambda r: r["twin"]["paired"].pop("frames_sha256"), "'paired.frames_sha256'", "sightline eval --all"),
    (lambda r: r["leakage"]["pairs"][1]["blocked"].pop("gap_s"), "'pairs.1.blocked.gap_s'", "sightline leakage"),
    (lambda r: r["legacy"]["drives"]["drive2017"].pop("label_sha256"), "'drives.drive2017.label_sha256'",
     "sightline legacy"),
    # values results.json copies whole, which only the page reads
    (lambda r: r["summary"]["paired_leakage"]["sightlinenet"]["ratio"].pop("values"),
     "'paired_leakage.sightlinenet.ratio.values'", "sightline report"),
    (lambda r: r["leakage"]["gaps"]["leaky"]["test"].pop("frac_within_1"), "'gaps.leaky.test.frac_within_1'",
     "sightline leakage"),
    (lambda r: r["leakage"]["gaps"]["blocked"]["test"].pop("min_frames"), "'gaps.blocked.test.min_frames'",
     "sightline leakage"),
    (lambda r: r["shipped"]["test"]["baselines"]["zero"].pop("n"), "'test.baselines.zero.n'", "sightline eval --all"),
    (lambda r: r["twin"]["cross_drive"]["baselines"]["zero"].pop("n"), "'cross_drive.baselines.zero.n'",
     "sightline eval --all"),
])
def test_a_report_lacking_a_key_the_site_reads_is_a_one_line_error(site, tmp_path, capsys, change, named, command):
    reports = _reports_with(site, tmp_path, change)
    capsys.readouterr()
    code = main(_site_data_args(reports / "summary.json", tmp_path / "public", reports, site.loc, video=False))
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if line.startswith("error: ")]
    assert code == 2 and "Traceback" not in err and len(lines) == 1, err
    assert named in lines[0] and command in lines[0]
    assert not (tmp_path / "public").exists()


def _snapshot(folder: Path) -> dict:
    """Every path under ``folder``, with each file's bytes (None for a folder)."""
    return {p.relative_to(folder).as_posix(): (p.read_bytes() if p.is_file() else None) for p in folder.rglob("*")}


def _previous_build(site, tmp_path) -> tuple[Path, Path]:
    out, reports = tmp_path / "public", tmp_path / "reports"
    shutil.copytree(site.out, out)
    shutil.copytree(site.reports, reports)
    return out, reports


def test_a_build_that_fails_before_its_swap_leaves_the_previous_one(site, tmp_path, monkeypatch):
    out, reports = _previous_build(site, tmp_path)
    before, export = _snapshot(out), (reports / "export.json").read_bytes()

    def refuse(*args, **kwargs):
        raise RuntimeError("ONNX angle parity 1 deg exceeds 0.0005 deg on 208 frame stacks")

    monkeypatch.setattr(site_data, "export_onnx", refuse)
    with pytest.raises(RuntimeError, match="parity"):
        _build(site, tmp_path, _workspace_with(site, tmp_path, _untouched), reports=reports)
    assert _snapshot(out) == before and (reports / "export.json").read_bytes() == export
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".public-")] == []
    assert sorted(p.name for p in reports.iterdir()) == sorted(p.name for p in site.reports.iterdir())


def test_a_build_that_fails_while_swapping_in_puts_the_previous_one_back(site, tmp_path):
    out, reports = _previous_build(site, tmp_path)
    # a file where the models folder belongs: the swap fails after the data files were replaced
    shutil.rmtree(out / "models")
    (out / "models").write_bytes(b"not a folder")
    # files and a folder the previous build lacked: the failed swap must take the new ones away
    shutil.rmtree(out / "data" / "selftest")
    next((out / "data" / "leakage").glob("frame-*.jpg")).unlink()
    before, export = _snapshot(out), (reports / "export.json").read_bytes()
    with pytest.raises(OSError):
        _build(site, tmp_path, _workspace_with(site, tmp_path, _untouched), reports=reports)
    assert _snapshot(out) == before and (reports / "export.json").read_bytes() == export
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".public-")] == []
    assert sorted(p.name for p in reports.iterdir()) == sorted(p.name for p in site.reports.iterdir())


def test_results_json_carries_the_export_parity_of_its_own_build_not_the_previous_ones(site, tmp_path):
    out, reports = _previous_build(site, tmp_path)
    previous = json.loads((reports / "export.json").read_text())
    # a build of shorter clips checks fewer frame stacks than the build whose export.json is on disk
    export = site_data.build_site_data(reports / "summary.json", _workspace_with(site, tmp_path, _untouched),
                                       out_dir=out, reports_dir=reports, clip_frames=CLIP - 10, video=False)
    parity = json.loads((out / "data" / "results.json").read_text())["parity"]
    assert json.loads((reports / "export.json").read_text()) == export
    for key in ("sightline", "shuffled"):
        checked = export["models"][key]["parity"]
        assert checked["frames"] == 2 * (CLIP - 10) + 8 != previous["models"][key]["parity"]["frames"]
        assert (parity[key]["stacks"], parity[key]["max_abs_angle_deg"], parity[key]["max_abs_saliency"]) == \
            (checked["frames"], checked["max_angle_diff_deg"], checked["max_saliency_diff"]), key


def test_export_json_is_staged_beside_itself_so_the_reports_can_sit_on_another_volume(site, tmp_path, monkeypatch):
    out, reports = tmp_path / "volume-a" / "public", tmp_path / "volume-b" / "reports"
    shutil.copytree(site.out, out)
    shutil.copytree(site.reports, reports)
    other_volume = reports.parent.resolve()
    replace_file = os.replace

    def renames_within_one_volume(src, dst):
        # volume b holds the reports; a rename between it and anywhere else fails as it would across volumes
        if Path(src).resolve().is_relative_to(other_volume) != Path(dst).resolve().is_relative_to(other_volume):
            raise OSError(errno.EXDEV, os.strerror(errno.EXDEV), str(src), None, str(dst))
        replace_file(src, dst)

    monkeypatch.setattr(os, "replace", renames_within_one_volume)
    export = site_data.build_site_data(reports / "summary.json", _workspace_with(site, tmp_path, _untouched),
                                       out_dir=out, reports_dir=reports, clip_frames=CLIP, video=False)
    assert json.loads((reports / "export.json").read_text()) == export
    assert json.loads((out / "data" / "results.json").read_text())["parity"]["sightline"]["max_abs_angle_deg"] == \
        export["models"]["sightline"]["parity"]["max_angle_diff_deg"]
    # nothing staged or set aside is left beside the reports or the public folder
    assert sorted(p.name for p in reports.iterdir()) == sorted(p.name for p in site.reports.iterdir())
    assert [p.name for p in out.parent.iterdir()] == ["public"]


def test_a_missing_ffmpeg_is_named_before_anything_is_written(site, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "no-tools"))
    with pytest.raises(MissingDataError, match="ffmpeg"):
        site_data.build_site_data(site.reports / "summary.json", _workspace_with(site, tmp_path, lambda r, d: None),
                                  out_dir=tmp_path / "public", reports_dir=site.reports, clip_frames=CLIP)
    assert not (tmp_path / "public").exists()


def test_site_data_without_a_summary_is_a_one_line_error(tmp_path, capsys):
    code = main(["site-data", "--summary", str(tmp_path / "none.json"), "--out-dir", str(tmp_path / "public")])
    err = capsys.readouterr().err
    assert code == 2 and "Traceback" not in err and "sightline report" in err and "none.json" in err


# --- reports of many runs -------------------------------------------------------------------

def _reports_with_runs(site, tmp_path, runs: list[dict], files: bool = True) -> Path:
    """A copy of the fixture's reports in which the summary lists ``runs`` (reports of other runs)
    too, and with ``files`` their run reports are there."""
    reports = _reports_with(site, tmp_path, lambda loaded: None)
    summary = json.loads((reports / "summary.json").read_text())
    summary["runs"] = sorted([*summary["runs"], *(r["run"] for r in runs)])
    (reports / "summary.json").write_text(json.dumps(summary))
    for report in runs if files else []:
        (reports / "runs" / f"{report['run']}.json").write_text(json.dumps(report))
    return reports


def test_reports_of_many_runs_are_checked_and_listed_and_give_every_protocol_its_sizes(site, tmp_path):
    shipped = json.loads((site.reports / "runs" / f"{site.shipped}.json").read_text())
    extra = _many_runs(shipped)
    reports_dir = _reports_with_runs(site, tmp_path, extra)
    reports = site_data.load_reports(reports_dir / "summary.json", reports_dir)
    # the shipped run and its twin first, as before, then the others in the summary's order
    assert reports.files == ["reports/summary.json", f"reports/runs/{site.shipped}.json",
                             f"reports/runs/{site.twin}.json",
                             *(f"reports/runs/{name}.json" for name in sorted(r["run"] for r in extra)),
                             "reports/leakage.json", "reports/legacy.json"]
    assert list(reports.runs) == [site.shipped, site.twin, *sorted(r["run"] for r in extra)]
    site_data._check_reports(reports)  # the runs agree with the shipped run and with the splits module
    results = site_data.site_results(reports, site.labels, site.export)
    assert {p: (s["runs"], s["seeds"]) for p, s in results["split_sizes"].items()} == \
        {"leaky": (2, [0, 1]), "blocked": (3, [0, 1, 2]), "forward": (2, [0, 1])}
    assert results["generated_from"] == reports.files


def test_a_build_from_the_reports_of_many_runs_writes_the_sizes_of_every_protocol(site, tmp_path):
    shipped = json.loads((site.reports / "runs" / f"{site.shipped}.json").read_text())
    reports_dir = _reports_with_runs(site, tmp_path, _many_runs(shipped))
    _build(site, tmp_path, _workspace_with(site, tmp_path, _untouched), reports=reports_dir)
    written = json.loads((tmp_path / "public" / "data" / "results.json").read_text())
    assert sorted(written["split_sizes"]) == ["blocked", "forward", "leaky"]
    assert {p: {k: s[k] for k in ("train", "val", "test")} for p, s in written["split_sizes"].items()} == HAND_COUNTS
    assert len(written["generated_from"]) == 10  # the summary, seven run reports, the leakage and legacy reports
    assert written["drive_profile"] == site.results["drive_profile"]


@pytest.mark.parametrize("change, message", [
    (lambda r: r["splits"].update(firewall=FIREWALL + 1),
     "made its splits with {'firewall': 21, 'block': 200}, but the shipped run with {'firewall': 20, 'block': 200}"),
    (lambda r: r["splits"].update(block=BLOCK + 1),
     "made its splits with {'firewall': 20, 'block': 201}, but the shipped run with {'firewall': 20, 'block': 200}"),
    (lambda r: r.update(drive="drive2017"), "trained on drive 'drive2017', the shipped run on 'drive2018'"),
    (lambda r: r["drives"]["drive2018"].update(frames=N + 1),
     f"describes drive 'drive2018' with {N + 1} frames, but summary.json with {N}"),
    (lambda r: r["splits"].update(train_n=r["splits"]["train_n"] + 1),
     "recorded 1121/160/160 train/validation/test frames for its blocked split, but the splits module makes "
     "1120/160/160 of the same 1600 frames with the same settings; the split rules are not the ones it was "
     "trained with"),
    (lambda r: r["splits"].update(test_n=r["splits"]["test_n"] - 1),
     "recorded 1120/160/159 train/validation/test frames for its blocked split"),
    (lambda r: r.update(protocol="honest"),
     "the splits module cannot make its honest split of 1600 frames: unknown protocol 'honest'"),
])
def test_a_run_report_that_disagrees_with_the_shipped_run_or_the_splits_module_is_refused(
        site, tmp_path, change, message):
    shipped = json.loads((site.reports / "runs" / f"{site.shipped}.json").read_text())
    report = _run_report(shipped, "blocked", 1)
    change(report)
    reports_dir = _reports_with_runs(site, tmp_path, [report])
    with pytest.raises(ValueError) as exc:
        _build(site, tmp_path, _workspace_with(site, tmp_path, _untouched), reports=reports_dir)
    assert message in str(exc.value) and report["run"] in str(exc.value) and "\n" not in str(exc.value)
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("whose", ["a blocked run", "a shuffled run", "the twin"])
@pytest.mark.parametrize("change", [
    lambda paired: paired.update(frames_sha256="0" * 64),
    lambda paired: paired.update(n=paired["n"] + 1),
], ids=["digest", "count"])
def test_paired_frames_that_are_not_the_ones_the_splits_module_pairs_are_refused_for_either_protocol(
        site, tmp_path, whose, change):
    """Both protocols scored on the paired frames have them checked: an extra blocked or shuffled run,
    and the shuffled twin itself."""
    if whose == "the twin":
        run, reports_dir = site.twin, _reports_with(site, tmp_path, lambda loaded: change(loaded["twin"]["paired"]))
    else:
        shipped = json.loads((site.reports / "runs" / f"{site.shipped}.json").read_text())
        report = _run_report(shipped, {"a blocked run": "blocked", "a shuffled run": "leaky"}[whose], 1)
        change(report["paired"])
        run, reports_dir = report["run"], _reports_with_runs(site, tmp_path, [report])
    with pytest.raises(ValueError) as exc:
        _build(site, tmp_path, _workspace_with(site, tmp_path, _untouched), reports=reports_dir)
    message = str(exc.value)
    assert f"run {run!r} recorded " in message and " paired frames (sha256 " in message, message
    assert "but the splits module pairs" in message and "\n" not in message
    assert not (tmp_path / "public").exists()


def _validation_moved(protocol, n, seed=0, firewall=splits.FIREWALL, block=splits.BLOCK):
    """``make_split`` with the blocked split's validation block one block earlier (TVTTTET instead of
    TTVTTET): the same sizes and the same test frames, other validation and training frames."""
    if protocol == "blocked":
        return splits.blocked_split(n, block=block, firewall=firewall, pattern="TVTTTET")
    return make_split(protocol, n, seed=seed, firewall=firewall, block=block)


def test_a_split_that_moves_validation_frames_alone_is_found_by_the_saved_splits(site, tmp_path, monkeypatch):
    moved, made = _validation_moved("blocked", N, firewall=FIREWALL, block=BLOCK), \
        make_split("blocked", N, firewall=FIREWALL, block=BLOCK)
    assert (moved.train.size, moved.val.size, moved.test.size) == (made.train.size, made.val.size, made.test.size)
    assert np.array_equal(moved.test, made.test) and not np.array_equal(moved.val, made.val)
    monkeypatch.setattr(site_data, "make_split", _validation_moved)
    # the reports cannot tell: the split sizes and the paired test frames are the same
    reports = site_data.load_reports(site.reports / "summary.json", site.reports)
    site_data._check_reports(reports)
    with pytest.raises(ValueError, match=rf"^run '{site.shipped}': its saved (train|val) frames \((train|val)_idx\.npy\) "
                                         rf"are not the blocked split the splits module makes of the {N} frames of "
                                         rf"drive2018 with seed 0, firewall {FIREWALL} and block {BLOCK}; the split "
                                         "rules are not the ones it was trained with$"):
        _build(site, tmp_path, _workspace_with(site, tmp_path, _untouched))
    assert not (tmp_path / "public").exists()


def test_a_twin_whose_saved_split_is_not_the_one_its_report_describes_is_refused(site, tmp_path):
    def trade_a_frame(runs, data):
        # one validation frame and one training frame change places: the same sizes, other frames
        train, val = (np.load(runs / site.twin / f"{split}_idx.npy") for split in ("train", "val"))
        train[0], val[0] = val[0], train[0]
        np.save(runs / site.twin / "train_idx.npy", train)
        np.save(runs / site.twin / "val_idx.npy", val)

    with pytest.raises(ValueError, match=rf"^run '{site.twin}': its saved train frames \(train_idx\.npy\) are not "
                                         rf"the shuffled split the splits module makes of the {N} frames of drive2018 "
                                         "with seed 0"):
        _build(site, tmp_path, _workspace_with(site, tmp_path, trade_a_frame))
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("saved_seed", [3, 0])
def test_a_twins_saved_split_is_the_shuffled_split_of_the_seed_its_report_records(site, tmp_path, saved_seed):
    """With seed 3 recorded for the shipped run and its twin (the blocked split takes no seed), the twin's
    saved split must be the shuffled split of seed 3: that one passes, and the one of seed 0 is refused."""
    reports = site_data.load_reports(site.reports / "summary.json", site.reports)
    reseeded = replace(reports, runs={**reports.runs, **{name: {**reports.runs[name], "seed": 3}
                                                         for name in (site.shipped, site.twin)}})

    def save_the_split_of_the_seed(runs, data):
        made = make_split("leaky", N, seed=saved_seed, firewall=FIREWALL, block=BLOCK)
        for split in ("train", "val", "test"):
            np.save(runs / site.twin / f"{split}_idx.npy", getattr(made, split))

    workspace = _workspace_with(site, tmp_path, save_the_split_of_the_seed)
    if saved_seed == 3:
        site_data._check_saved_splits(workspace, reseeded)
        return
    with pytest.raises(ValueError, match=rf"^run '{site.twin}': its saved train frames \(train_idx\.npy\) are not "
                                         rf"the shuffled split the splits module makes of the {N} frames of drive2018 "
                                         "with seed 3, "):
        site_data._check_saved_splits(workspace, reseeded)


def test_a_run_report_filed_under_another_runs_name_is_refused(site, tmp_path):
    shipped = json.loads((site.reports / "runs" / f"{site.shipped}.json").read_text())
    report = _run_report(shipped, "blocked", 1)
    reports_dir = _reports_with_runs(site, tmp_path, [report])
    (reports_dir / "runs" / f"{report['run']}.json").write_text(
        json.dumps({**report, "run": "drive2018_sightlinenet_blocked_s9"}))
    with pytest.raises(ValueError, match=f"runs/{report['run']}.json is the report of run "
                                         "'drive2018_sightlinenet_blocked_s9'"):
        _build(site, tmp_path, _workspace_with(site, tmp_path, _untouched), reports=reports_dir)
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("change, files, message", [
    (lambda r: r["splits"].pop("val_n"), True, "lacks 'splits.val_n'; write it again with `sightline eval --all`"),
    (lambda r: r.pop("drives"), True, "lacks 'drives'; write it again with `sightline eval --all`"),
    (lambda r: None, False, "no report at "),
])
def test_a_run_report_that_is_missing_or_lacks_a_key_is_a_one_line_error(
        site, tmp_path, capsys, change, files, message):
    shipped = json.loads((site.reports / "runs" / f"{site.shipped}.json").read_text())
    report = _run_report(shipped, "forward", 0)
    change(report)
    reports_dir = _reports_with_runs(site, tmp_path, [report], files)
    capsys.readouterr()
    code = main(_site_data_args(reports_dir / "summary.json", tmp_path / "public", reports_dir, site.loc,
                                video=False))
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if line.startswith("error: ")]
    assert code == 2 and "Traceback" not in err and len(lines) == 1, err
    assert message in lines[0] and report["run"] in lines[0]
    assert not (tmp_path / "public").exists()


# --- the clip rule on its own ---------------------------------------------------------------

def test_a_window_must_fit_its_history_frames_inside_the_stretch():
    angles = np.zeros(100)
    angles[20:30] = np.arange(10)  # the most varied labels sit at the very start of the stretch
    chosen = site_data.choose_window(angles, [(20, 60)], n=10, lead_in=2, step=10)
    # a window starting at 20 would need frames 18 and 19, outside the stretch
    assert chosen == {"first_frame": 22, "label_sd_deg": pytest.approx(np.std(angles[22:32])), "windows": 3}


def test_ties_go_to_the_earliest_window_and_a_stretch_too_short_is_an_error():
    flat = np.ones(200)
    assert site_data.choose_window(flat, [(50, 120), (130, 200)], n=20, lead_in=2, step=10)["first_frame"] == 52
    with pytest.raises(ValueError, match="longest"):
        site_data.choose_window(flat, [(0, 21), (40, 50)], n=20, lead_in=2, step=10)


def test_a_clip_window_keeps_every_label_within_90_degrees_lead_in_included():
    angles = np.zeros(80)
    angles[10:20] = np.linspace(-80, 80, 10)  # varied, within the limit
    angles[40:50] = np.linspace(-150, 150, 10)  # more varied, past it
    spans = [(0, 80)]
    assert site_data.choose_window(angles, spans, n=10, lead_in=2, step=10, max_abs_deg=1e9)["first_frame"] == 42
    capped = site_data.choose_window(angles, spans, n=10, lead_in=2, step=10)
    # 32 and 42 hold labels past 90 deg, in their windows or their lead-in (frames 40 and 41)
    assert capped["first_frame"] == 12 and capped["windows"] == 5
    # frame 11 past the limit: the window at 12 keeps its scored frames 12..21 within it, but
    # its lead-in (frames 10 and 11) is not, so it goes too (and the window at 2, which holds
    # frame 11); the flat windows that are left tie, and the earliest wins
    angles[11] = 91.0
    assert site_data.choose_window(angles, spans, n=10, lead_in=2, step=10)["first_frame"] == 22


# --- the committed site data --------------------------------------------------------------

def published_skip_reason(repo: Path) -> str | None:
    """Why the committed-site-data tests skip, or None when they run: they run once
    web/public/data/results.json is tracked by git, and then fail if it is missing or
    unreadable."""
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "web/public/data/results.json"],
                             cwd=repo, capture_output=True).returncode == 0
    return None if tracked else "no site data is committed in web/public yet"


needs_published = pytest.mark.skipif(published_skip_reason(REPO) is not None,
                                     reason="no site data is committed in web/public yet")


def test_the_committed_site_data_tests_run_once_results_json_is_tracked(tmp_path):
    repo = tmp_path / "clone"
    results = repo / "web" / "public" / "data" / "results.json"
    results.parent.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    results.write_text("{}")
    assert published_skip_reason(repo) == "no site data is committed in web/public yet"  # there, not tracked
    subprocess.run(["git", "add", "web/public/data/results.json"], cwd=repo, check=True, capture_output=True)
    results.unlink()
    assert published_skip_reason(repo) is None  # tracked: the tests run, and fail on the missing file


def clip_prediction_gaps(clip_dir: Path, models: dict) -> dict:
    """For each model, the largest |onnxruntime - expected| over a clip's scored frames, with
    onnxruntime's CPU provider running the model file on the clip's sprites as Pillow decodes
    them, stacked with the clip's frame lags from its lead-in."""
    clip = json.loads((clip_dir / "clip.json").read_text())
    frames = _decode([clip_dir / name for name in clip["sprite"]["files"]])
    inputs = _stacks(frames, clip["lead_in"], clip["n"], clip["frame_lags"])
    return {key: float(np.abs(_ort(path, inputs)[0] - np.asarray(clip["expected"][key], dtype=np.float32)).max())
            for key, path in models.items()}


def test_the_clip_check_runs_each_model_file_on_the_sprites(site):
    models = {key: site.out / path for key, path in site.results["assets"]["models"].items()}
    for cid in TITLES:
        assert clip_prediction_gaps(_clip_dir(site, cid), models) == {"sightline": 0.0, "shuffled": 0.0}
    swapped = {"sightline": models["shuffled"], "shuffled": models["sightline"]}
    assert min(clip_prediction_gaps(_clip_dir(site, "c2018"), swapped).values()) > 1e-3


def committed_drives() -> dict | None:
    """The real drives' repaired labels where the dataset is on this machine (it is not part of
    a clone), else None."""
    names = ("drive2018", "drive2017")
    if not all((REPO / "data" / name / "data.txt").is_file() for name in names):
        return None
    return {name: repair_dropouts(load_labels(REPO / "data" / name)) for name in names}


@needs_published
def test_the_committed_results_json_matches_the_committed_reports_and_the_drives_when_they_are_here():
    results = json.loads((PUBLISHED / "data" / "results.json").read_text())
    assert results_out_of_sync(results, REPO / "reports" / "summary.json", REPO / "reports",
                               committed_drives()) == []


@needs_published
def test_the_committed_drive_profile_fits_the_committed_results():
    results = json.loads((PUBLISHED / "data" / "results.json").read_text())
    assert drive_profile_problems(results) == []
    profile = results["drive_profile"]["drives"]
    assert 900 <= len(profile["drive2018"]["start_frame"]) <= 1100  # about a thousand points
    # a bucket holds the same number of frames but the last, so the frame-weighted mean of the
    # buckets' mean absolute angles is the drive's mean absolute angle: the other day's
    # always-straight error in the reports (taken from its third frame on), to the rounding
    bucket, frames = results["drive_profile"]["bucket_frames"], profile["drive2017"]["frames"]
    weights = [min(bucket, frames - start) for start in profile["drive2017"]["start_frame"]]
    mean_abs = sum(w * m for w, m in zip(weights, profile["drive2017"]["mean_abs_deg"])) / frames
    straight = results["baselines"]["cross_drive"]["baselines"]["zero"]["mae_deg"]
    assert results["baselines"]["cross_drive"]["drive"] == "drive2017"
    assert mean_abs == pytest.approx(straight, abs=0.01)


@needs_published
def test_the_committed_timing_fits_the_committed_results_and_is_the_drives_measured_times_when_they_are_here():
    results = json.loads((PUBLISHED / "data" / "results.json").read_text())
    summary = json.loads((REPO / "reports" / "summary.json").read_text())
    shipped = json.loads((REPO / "reports" / "runs" / f"{summary['shipped']['run']}.json").read_text())
    assert timing_problems(results, shipped["frame_lags"]) == []
    assert [name for name, timing in results["timing"].items() if timing is not None] == [shipped["drive"]]
    drives = committed_drives()
    if drives is not None:  # the drives are here: every value is worked out again, apart from the builder
        expected = expected_timing(drives, shipped["frame_lags"], shipped["splits"], trained_on=shipped["drive"])
        assert leaf_differences(results["timing"], expected) == []


@needs_published
def test_the_committed_split_rules_and_sizes_are_the_splits_modules_and_the_run_reports():
    results = json.loads((PUBLISHED / "data" / "results.json").read_text())
    rules, sizes = results["split_rules"], results["split_sizes"]
    assert rules["blocked"] == {"label": "blocked", "block": splits.BLOCK, "firewall": splits.FIREWALL,
                                "pattern": splits.BLOCK_PATTERN}
    for protocol in ("leaky", "forward"):
        shares = {"train_frac": splits.TRAIN_FRAC, "val_frac": splits.VAL_FRAC, "test_frac": 0.15}
        assert rules[protocol] == {"label": splits.PUBLIC_NAMES[protocol], **shares,
                                   **({"firewall": splits.FIREWALL} if protocol == "forward" else {})}
    frames = results["drives"]["drive2018"]["frames"]
    assert sorted(sizes) == sorted(splits.PROTOCOLS)
    for protocol, found in sizes.items():
        made = make_split(protocol, frames)  # with the splits module's own settings
        counts = {"train": made.train.size, "val": made.val.size, "test": made.test.size}
        assert found == {"label": splits.PUBLIC_NAMES[protocol], "drive": "drive2018", "frames": frames, **counts,
                         "dropped": frames - sum(counts.values()), "runs": found["runs"], "seeds": [0, 1, 2]}, protocol
    # every run report of the summary is in, and every one of them was counted
    summary = json.loads((REPO / "reports" / "summary.json").read_text())
    assert sum(found["runs"] for found in sizes.values()) == len(summary["runs"])


@needs_published
def test_no_string_of_the_committed_site_data_holds_a_middle_dot_a_bullet_or_a_thin_space():
    files = site_data_files(PUBLISHED)
    assert len(files) == 5 and all(path.is_file() for path in files)
    for path in files:
        assert banned_characters(json.loads(path.read_text())) == [], path.relative_to(REPO)


@needs_published
def test_the_committed_2018_profile_is_of_the_labels_the_forward_runs_were_scored_on():
    results = json.loads((PUBLISHED / "data" / "results.json").read_text())
    rules, sizes = results["split_rules"]["forward"], results["split_sizes"]["forward"]
    frames, test = sizes["frames"], sizes["test"]
    # the later-in-the-drive split tests on the end of the drive, from a firewall after its validation frames
    start = frames - test
    assert start == int(frames * (rules["train_frac"] + rules["val_frac"])) + rules["firewall"]
    profile, bucket = results["drive_profile"]["drives"][sizes["drive"]], results["drive_profile"]["bucket_frames"]

    def mean_abs_over_the_test_frames(drive: dict) -> tuple[float, float]:
        """The frame-weighted mean |angle| of the buckets over the test frames, and how far that can
        be from the frames' own mean: a bucket's mean rounded to 0.01 deg moves it by 0.005 deg at
        most, and a bucket only partly inside the stretch can average there anything from 0 up to
        its size over the part inside times its mean."""
        weighted, bound = 0.0, 0.005
        for first, mean_abs in zip(drive["start_frame"], drive["mean_abs_deg"]):
            size = min(bucket, frames - first)
            inside = max(0, first + size - max(first, start))
            weighted += inside * mean_abs
            if 0 < inside < size:
                bound += inside / test * (mean_abs + 0.005) * max(1.0, size / inside - 1.0)
        return weighted / test, bound

    weighted, bound = mean_abs_over_the_test_frames(profile)
    (straight,) = {g["test_zero_mae_deg"] for key, g in results["groups"].items() if key.endswith("/forward")}
    # always steering straight ahead errs by the mean |angle| of the forward test frames, so the two agree
    # within the bound (0.011 deg here: 9.4049 against 9.4064), plus 1e-4 deg for the cache's float32
    # labels; a profile of the labels as recorded, dropouts left as zeros, gives 9.3038 and is found
    assert abs(weighted - straight) <= bound + 1e-4 and bound < 0.02
    if (REPO / "data" / sizes["drive"] / "data.txt").is_file():
        raw = site_data.drive_profile(load_labels(REPO / "data" / sizes["drive"]))
        assert abs(mean_abs_over_the_test_frames(raw)[0] - straight) > 5 * (bound + 1e-4)


@needs_published
def test_the_committed_clips_give_their_times_on_the_drive_profiles_time_axis():
    results = json.loads((PUBLISHED / "data" / "results.json").read_text())
    profile, drives = results["drive_profile"], committed_drives()
    timed = {}
    for cid, path in results["assets"]["clips"].items():
        clip = json.loads((PUBLISHED / path).read_text())
        starts = profile["drives"][clip["drive"]]["start_s"]
        timed[cid] = clip["first_s"] is not None
        if starts is None:  # a drive without timestamps
            assert (clip["first_s"], clip["last_s"]) == (None, None), cid
            continue
        # each time lies between the start of its frame's bucket and the start of the next one
        last = clip["first_frame"] + clip["n"] - 1
        for frame, time in ((clip["first_frame"], clip["first_s"]), (last, clip["last_s"])):
            k = frame // profile["bucket_frames"]
            assert starts[k] <= time and (k + 1 == len(starts) or time <= starts[k + 1]), (cid, frame, time)
        if drives is not None:  # the drive is here: the times are its own timestamps'
            assert site_data.clip_times(drives[clip["drive"]], clip["first_frame"], clip["n"]) == \
                {"first_s": clip["first_s"], "last_s": clip["last_s"]}
    assert timed == {"c2018": True, "c2017": False}


@needs_published
def test_the_committed_sprites_decode_to_their_recorded_hashes():
    clips = sorted((PUBLISHED / "data" / "clips").glob("*/clip.json"))
    assert [p.parent.name for p in clips] == ["c2017", "c2018"]
    for path in clips:
        assert sprite_hash_mismatches(path.parent) == [], path.parent.name


@needs_published
def test_the_committed_2018_clip_is_as_far_from_training_as_the_firewall_promises():
    export = json.loads((REPO / "reports" / "export.json").read_text())
    shipped = json.loads((REPO / "reports" / "summary.json").read_text())["shipped"]["run"]
    firewall = json.loads((REPO / "reports" / "runs" / f"{shipped}.json").read_text())["splits"]["firewall"]
    assert export["models"]["sightline"]["run"] == shipped
    assert export["clips"]["c2018"]["min_frames_from_training"] >= 2 * firewall + 1


@needs_published
def test_the_committed_models_reproduce_each_committed_clip_from_its_sprites():
    results = json.loads((PUBLISHED / "data" / "results.json").read_text())
    models = {key: PUBLISHED / path for key, path in results["assets"]["models"].items()}
    assert sorted(results["assets"]["clips"]) == ["c2017", "c2018"]
    for cid, path in results["assets"]["clips"].items():
        clip_dir = (PUBLISHED / path).parent
        # the tolerance the page holds a browser to is the export's, and the check uses the export's own
        assert json.loads((clip_dir / "clip.json").read_text())["tolerance"] == {"angle_deg": ANGLE_TOLERANCE_DEG}
        gaps = clip_prediction_gaps(clip_dir, models)
        assert max(gaps.values()) <= ANGLE_TOLERANCE_DEG, (cid, gaps)


@needs_published
def test_the_committed_models_are_the_exported_ones_and_reproduce_the_self_test():
    export = json.loads((REPO / "reports" / "export.json").read_text())
    expected = json.loads((PUBLISHED / "data" / "selftest" / "expected.json").read_text())
    inputs = np.frombuffer((PUBLISHED / "data" / "selftest" / "frames.bin").read_bytes(), np.uint8)
    inputs = inputs.reshape(expected["shape"])
    assert expected["tolerance"] == {"angle_deg": ANGLE_TOLERANCE_DEG, "saliency": SALIENCY_TOLERANCE}
    for key, entry in export["models"].items():
        data = (PUBLISHED / entry["file"]).read_bytes()
        assert len(data) == entry["bytes"] <= 2_000_000 and hashlib.sha256(data).hexdigest() == entry["sha256"]
        session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
        angle, saliency = session.run(None, {"frames": inputs})
        maps = np.frombuffer(base64.b64decode(expected["saliency_f32_b64"][key]), "<f4").reshape(saliency.shape)
        assert np.abs(angle[:, 0] - np.asarray(expected["angles"][key])).max() <= ANGLE_TOLERANCE_DEG
        assert np.abs(saliency - maps).max() <= SALIENCY_TOLERANCE
