"""The site data: everything the browser loads, written from the reports and the checkpoints.

A module-scoped fixture makes two synthetic drives named like the real ones (the 2017
stand-in without timestamps), trains the shipped run and its shuffled-split twin for one
epoch on the CPU, evaluates and summarises them, runs the leakage analysis and the
standard-recipe evaluation (random TF1 weights), and then runs ``sightline site-data``
twice: once as published, with the clip videos when ffmpeg is installed, and once without
videos into a second folder, which must hold the same bytes otherwise.

The 2018 stand-in keeps the image of frame ``i`` under the file name frame ``i + 7`` had:
a drive's frames are its label rows, so code that assumes ``<i>.jpg`` shows the wrong frame.

The tests at the end check the committed site data in ``web/public`` against the committed
reports; they skip while none is committed.
"""
import base64
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
import torch
from PIL import Image, JpegImagePlugin

from conftest import REPO, make_drive
from sightline.cli import main
from sightline.data.cache import DriveCache
from sightline.data.labels import load_labels, repair_dropouts
from sightline.data.splits import make_split
from sightline.export import site_data
from sightline.legacy import TENSOR_SHAPES
from sightline.paths import MissingDataError, UnusableRunError, Workspace
from sightline.reporting import load_run_model

N, N_CROSS, BLOCK, FIREWALL, CLIP, SHIFT = 1600, 300, 200, 20, 100, 7
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
    _store_frames_under_other_names(make_drive(data, "drive2018", n=N, scenery=True), SHIFT)
    make_drive(data, "drive2017", n=N_CROSS, timestamps=False)
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
    ctx.labels = {d: repair_dropouts(load_labels(data / d)) for d in ("drive2018", "drive2017")}
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
        # parity is measured on every clip input and every self-test input
        assert entry["parity"]["frames"] == 2 * CLIP + 8
        assert 0 <= entry["parity"]["max_angle_diff_deg"] <= 1e-3
        assert 0 <= entry["parity"]["max_saliency_diff"] <= 1e-3


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


def test_expected_outputs_are_cpu_predictions_on_the_decoded_sprites(site):
    for cid, clip in site.clips.items():
        decoded = _decoded_clip(site, cid)
        inputs = _stacks(decoded, clip["lead_in"], CLIP, clip["frame_lags"])
        for key, run in (("sightline", site.shipped), ("shuffled", site.twin)):
            fresh = _predict(site, run, inputs)
            np.testing.assert_allclose(clip["expected"][key], fresh, rtol=0, atol=1e-4, err_msg=f"{cid} {key}")
            # the first expected output uses the lead-in frames as its history
            assert abs(clip["expected"][key][0] - fresh[0]) <= 1e-4
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


def _rule(angles: np.ndarray, frames: np.ndarray, n: int, lead_in: int) -> int:
    """The first scored frame of the n-frame window, with its lead-in, inside one stretch of
    consecutive ``frames``, whose labels vary most (population sd); windows start every 10
    frames from the first possible one in each stretch; ties go to the earliest."""
    frames = np.sort(frames)
    cuts = np.flatnonzero(np.diff(frames) > 1) + 1
    best = None
    for stretch in np.split(frames, cuts):
        for first in range(stretch[0] + lead_in, stretch[-1] - n + 2, 10):
            sd = float(np.std(angles[first:first + n].astype(np.float64)))
            if best is None or sd > best[1]:
                best = (first, sd)
    return best[0]


def test_each_clip_is_the_window_its_rule_picks_from_the_labels(site):
    clip = site.clips["c2018"]
    test = make_split("blocked", N, firewall=FIREWALL, block=BLOCK).test
    angles = site.caches["drive2018"].angles_deg
    assert clip["first_frame"] == _rule(angles, test, CLIP, clip["lead_in"])
    clip = site.clips["c2017"]
    angles = site.caches["drive2017"].angles_deg
    assert clip["first_frame"] == _rule(angles, np.arange(N_CROSS), CLIP, clip["lead_in"])
    for cid, clip in site.clips.items():
        entry = site.export["clips"][cid]
        assert (entry["drive"], entry["first_frame"], entry["n"], entry["lead_in"]) == \
            (clip["drive"], clip["first_frame"], CLIP, clip["lead_in"])
        assert "standard deviation" in entry["rule"] and "every 10 frames" in entry["rule"]
    assert "blocked test" in site.export["clips"]["c2018"]["rule"]


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


def test_the_self_test_expects_cpu_outputs_that_the_exported_models_reproduce(site):
    expected = site.selftest
    assert expected["tolerance"] == {"angle_deg": 1e-3, "saliency": 1e-3}
    inputs = np.frombuffer((site.out / "data" / "selftest" / "frames.bin").read_bytes(), np.uint8)
    inputs = inputs.reshape(expected["shape"])
    for key, run in (("sightline", site.shipped), ("shuffled", site.twin)):
        model, _ = load_run_model(site.runs / run)
        from sightline.saliency import SteeringWithSaliency

        with torch.no_grad():
            angle, saliency = SteeringWithSaliency(model).eval()(torch.from_numpy(inputs.copy()))
        maps = np.frombuffer(base64.b64decode(expected["saliency_f32_b64"][key]), "<f4").reshape(8, 66, 200)
        np.testing.assert_allclose(expected["angles"][key], angle[:, 0].numpy(), rtol=0, atol=1e-4)
        np.testing.assert_allclose(maps, saliency.numpy(), rtol=0, atol=1e-4)
        session = ort.InferenceSession(str(site.out / site.results["assets"]["models"][key]),
                                       providers=["CPUExecutionProvider"])
        ort_angle, ort_saliency = session.run(None, {"frames": inputs})
        assert np.abs(ort_angle[:, 0] - np.asarray(expected["angles"][key])).max() <= 1e-3
        assert np.abs(ort_saliency - maps).max() <= 1e-3


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
    ("legacy.trained_on", "legacy", "trained_on"),
    ("legacy.seen_first_frame", "legacy", "seen_first_frame"),
    ("legacy.splits", "legacy", "splits"),
    ("leakage.memoriser", "leakage", "memoriser"),
    ("leakage.paired", "leakage", "paired"),
    ("leakage.splits", "leakage", "splits"),
    ("leakage.drive", "leakage", "drive"),
    ("leakage.seed", "leakage", "seed"),
    ("leakage.median_frame_interval_s", "leakage", "dt_s"),
    ("baselines.run", "shipped_report", "run"),
    ("baselines.test.n", "shipped_report", "test.n"),
    ("baselines.test.model", "shipped_report", "test.model"),
    ("baselines.test.baselines", "shipped_report", "test.baselines"),
    ("baselines.cross_drive.drive", "shipped_report", "cross_drive.drive"),
    ("baselines.cross_drive.n", "shipped_report", "cross_drive.n"),
    ("baselines.cross_drive.model", "shipped_report", "cross_drive.model"),
    ("baselines.cross_drive.baselines", "shipped_report", "cross_drive.baselines"),
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
            "shipped_report": site.run_reports[site.shipped]}


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


def test_no_string_the_page_shows_has_a_middle_dot(site):
    shown = [site.out / "data" / "results.json", site.out / "data" / "leakage" / "pairs.json",
             site.out / "data" / "selftest" / "expected.json"]
    shown += [_clip_dir(site, cid) / "clip.json" for cid in TITLES]
    for path in shown:
        text = path.read_text()
        assert "\\u00b7" not in text and "\u00b7" not in text, path.name  # escaped or not
        strings = [value for _, value in _leaves(json.loads(text)) if isinstance(value, str)]
        assert strings and not any("\u00b7" in s for s in strings), path.name


def results_out_of_sync(results: dict, summary_path: Path, reports_dir: Path) -> list[str]:
    """The paths at which ``results`` differs from the results.json the reports give now."""
    fresh = site_data.site_results(site_data.load_reports(summary_path, reports_dir))
    a, b = dict(_leaves(results)), dict(_leaves(fresh))
    return sorted(".".join(map(str, leaf)) for leaf in a.keys() | b.keys()
                  if leaf not in a or leaf not in b or a[leaf] != b[leaf] or type(a[leaf]) is not type(b[leaf]))


def test_the_sync_check_finds_a_value_that_drifted_from_its_report(site):
    assert results_out_of_sync(site.results, site.reports / "summary.json", site.reports) == []
    drifted = copy.deepcopy(site.results)
    drifted["paired_leakage"]["sightlinenet"]["ratio"]["mean"] += 1e-9
    assert results_out_of_sync(drifted, site.reports / "summary.json", site.reports) == \
        ["paired_leakage.sightlinenet.ratio.mean"]


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


# --- the committed site data --------------------------------------------------------------

needs_published = pytest.mark.skipif(not (PUBLISHED / "data" / "results.json").is_file(),
                                     reason="no site data is committed in web/public yet")


@needs_published
def test_the_committed_results_json_matches_the_committed_reports():
    results = json.loads((PUBLISHED / "data" / "results.json").read_text())
    assert results_out_of_sync(results, REPO / "reports" / "summary.json", REPO / "reports") == []


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
def test_the_committed_models_are_the_exported_ones_and_reproduce_the_self_test():
    export = json.loads((REPO / "reports" / "export.json").read_text())
    expected = json.loads((PUBLISHED / "data" / "selftest" / "expected.json").read_text())
    inputs = np.frombuffer((PUBLISHED / "data" / "selftest" / "frames.bin").read_bytes(), np.uint8)
    inputs = inputs.reshape(expected["shape"])
    for key, entry in export["models"].items():
        data = (PUBLISHED / entry["file"]).read_bytes()
        assert len(data) == entry["bytes"] <= 2_000_000 and hashlib.sha256(data).hexdigest() == entry["sha256"]
        session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
        angle, saliency = session.run(None, {"frames": inputs})
        maps = np.frombuffer(base64.b64decode(expected["saliency_f32_b64"][key]), "<f4").reshape(saliency.shape)
        assert np.abs(angle[:, 0] - np.asarray(expected["angles"][key])).max() <= expected["tolerance"]["angle_deg"]
        assert np.abs(saliency - maps).max() <= expected["tolerance"]["saliency"]
