"""The browser's data: models, demo clips, results and self-test fixtures, from the reports.

``build_site_data`` reads the summary (``sightline report``), the run reports and the
checkpoints of the shipped run and of its twin (the same model and seed trained on the
shuffled split), the leakage and standard-recipe reports and the two drives, and writes
everything the page loads under ``out_dir`` (``web/public``). Paths inside results.json are
relative to ``out_dir``; paths inside the other JSON files are relative to their own folder.

``models/sightline.onnx``, ``models/sightline-shuffled.onnx``
    The shipped run and its twin, each one graph from uint8 frame stacks to steering
    degrees and saliency, at most 2 MB, checked against PyTorch with onnxruntime on every
    clip input and every self-test input (``SHIPPED_ANGLE_TOLERANCE_DEG``,
    ``SHIPPED_SALIENCY_TOLERANCE``).
``data/results.json``
    The numbers the page shows, copied value for value from the reports, never computed
    again: the generalisation ladder, the paired leakage measurement, each group's mean and
    sd over seeds, the shipped run's baselines and regimes, the standard-recipe PilotNet,
    the leakage analysis, the drives, the training recipe and the library versions. The
    leakage report's ``median_s`` (a median frame count times the median frame interval,
    not a time anything took) is left out; the interval itself is ``median_frame_interval_s``.
``data/clips/<id>/``
    A demo clip of ``clip_frames`` scored frames: ``c2018`` from the blocked test stretches
    of the drive the shipped run trained on (minutes it never trained on), ``c2017`` from
    the other drive (another day). ``sprite-NN.jpg`` holds the model's 200x66 input frames,
    60 to a sheet stacked top to bottom (JPEG q95, no chroma subsampling). Sprite frame
    ``k`` is drive frame ``first_frame - lead_in + k``: the ``lead_in`` frames before the
    first scored frame give it its history, so no stacked frame is ever clamped. The
    sprites are the clip's model input: ``decoded_sha256`` holds the sha256 of each frame's
    RGB bytes as Pillow decodes the written JPEGs, so a browser can check that it decodes
    the same pixels, and ``expected`` both models' outputs on those decoded frames, computed
    by onnxruntime (CPU provider, one thread) from the very ONNX files the page loads; a
    client's own predictions must agree within ``tolerance``. ``clip.mp4`` shows the scored
    camera frames (video frame ``k`` is scored frame ``k``) as H.264 at 17.5 fps; 4:2:0
    video needs an even width, so each 455-pixel camera frame is followed by one repeated
    column (``video_size`` against ``camera_size``). ``clip_mae_deg`` is the mean absolute
    error on this clip alone, for each model and for steering straight ahead.
    ``twin_trained_frac`` is the share of the clip's scored frames the twin trained on and
    ``twin_neighbour_frac`` the share within one frame of a frame it trained on, both from
    the twin's saved training split; the twin trained on frames of its own drive only, so
    on the other drive's clip both are 0.
``data/selftest/``
    ``frames.bin``: eight stacked inputs from the 2018 clip as raw uint8 NHWC bytes, taken
    from the drive cache (no JPEG involved), and ``expected.json``: both models' angles and
    saliency maps on them from the same onnxruntime run, with the tolerance a client must
    meet.
``data/leakage/``
    ``pairs.json``, the leakage report's example frame pairs with their gaps in seconds
    rounded to the millisecond, and one camera frame (JPEG q85) per frame they name, read
    through the drive's label rows.

``export.json`` in ``reports_dir`` records each model's size, digest, checkpoint and
measured parity with the gate it met, and how each clip was chosen, how much of it the twin
saw in training and how each model does on it.

A clip is chosen from the labels alone, never from a model's predictions: among the
windows of ``clip_frames`` frames that fit, with the ``lead_in`` frames before them, inside
one stretch (a stretch of the test frames the shipped run saved, or the whole of the other
drive) and whose labels, the lead-in included, all stay within ``CLIP_MAX_ABS_DEG`` either
way (road following, not parking or intersection manoeuvres), the window whose labels have
the largest standard deviation. Windows start every 10 frames from the first that fits in
each stretch, and ties go to the earliest. The 2018 clip's distance from the frames the
shipped run trained on is then measured; under 2·firewall + 1 frames it is refused. A clip
is picked for being worth watching, so its MAE describes the clip, not the model.

Nothing is written before the reports, checkpoints, caches and drives have passed every
check. The build is staged in a folder beside ``out_dir`` (``web/.public-*`` for
``web/public``), which is always removed. The previous build's files and export.json are
then moved aside (into another such folder) and the new ones moved in; if anything fails on
the way the new files are removed and the old ones moved back, so a failed build leaves
``out_dir`` and export.json as they were. Run again on the same machine with the same library versions, the same
reports, checkpoints and drives give the same bytes apart from the videos; elsewhere the
values computed in floating point (expected outputs, self-test angles and saliency, parity
figures) can differ in their last bits.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sightline.data.frames import stack_frames
from sightline.data.preprocess import INPUT_HEIGHT, INPUT_WIDTH
from sightline.data.splits import PUBLIC_NAMES
from sightline.export.onnx_export import (
    ANGLE_TOLERANCE_DEG,
    INPUT_NAME,
    SALIENCY_TOLERANCE,
    export_onnx,
    open_drive_cache,
)
from sightline.leakage import LOOKUP_PROTOCOLS, frame_gaps
from sightline.paths import MissingDataError, UnusableRunError, Workspace
from sightline.reporting import checkpoint_sha256, load_run_model, load_split_indices, write_report
from sightline.train import code_version

__all__ = [
    "ASSETS",
    "CAMERA_SIZE",
    "CLIP_FPS",
    "CLIP_FRAMES",
    "CLIP_MAX_ABS_DEG",
    "CLIP_STEP",
    "CLIP_TITLES",
    "FRAMES_PER_SHEET",
    "MAX_MODEL_BYTES",
    "MODEL_FILES",
    "Reports",
    "SHIPPED_ANGLE_TOLERANCE_DEG",
    "SHIPPED_SALIENCY_TOLERANCE",
    "build_site_data",
    "choose_window",
    "decode_sheets",
    "encode_video",
    "frame_digests",
    "jpeg_bytes",
    "load_reports",
    "min_gap_to_training",
    "site_results",
    "sprite_sheets",
    "stretches",
    "twin_exposure",
]

#: scored frames per demo clip: about 24 s of driving
CLIP_FRAMES = 420
#: candidate clip windows start every CLIP_STEP frames
CLIP_STEP = 10
#: a clip shows road following: every label of its window, lead-in included, stays within
#: this many degrees either way (larger angles are parking and intersection manoeuvres)
CLIP_MAX_ABS_DEG = 90.0
#: the clips' playback rate, frames per second (the 2018 drive's median frame interval is 57 ms)
CLIP_FPS = 17.5
#: the demo clips by id: the training drive's blocked test frames, and the other drive
CLIP_TITLES = {"c2018": "2018: minutes it never trained on", "c2017": "2017: another day"}
FRAMES_PER_SHEET = 60
SPRITE_QUALITY = 95
PAIR_QUALITY = 85
#: the camera frames as recorded, (width, height)
CAMERA_SIZE = (455, 256)
#: the largest model the page downloads, bytes
MAX_MODEL_BYTES = 2_000_000
#: the parity the shipped models must meet against PyTorch on every clip and self-test input.
#: Trained models land near 2e-4 deg (float32 rounding ahead of the output scale) and 1e-5 in
#: saliency; a structural export fault is off by whole degrees.
SHIPPED_ANGLE_TOLERANCE_DEG = 5e-4
SHIPPED_SALIENCY_TOLERANCE = 1e-4
#: stacked inputs in the self-test fixture
SELFTEST_INPUTS = 8
#: onnxruntime computes the expected outputs this many stacks at a time, on one thread
EXPECTED_BATCH = 64
#: the page's two models, by the names it uses, and their files under ``out_dir``
MODEL_FILES = {"sightline": "models/sightline.onnx", "shuffled": "models/sightline-shuffled.onnx"}
VIDEO_OPTIONS = ("-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "slow",
                 "-movflags", "+faststart")
#: what results.json keeps of the standard-recipe report
LEGACY_KEYS = ("recipe", "trained_on", "seen_first_frame", "splits", "port_parity", "eval", "atan_cap")
#: every file the page loads besides results.json, relative to ``out_dir``
ASSETS = {
    "models": dict(MODEL_FILES),
    "clips": {cid: f"data/clips/{cid}/clip.json" for cid in CLIP_TITLES},
    "leakage_pairs": "data/leakage/pairs.json",
    "selftest": {"frames": "data/selftest/frames.bin", "expected": "data/selftest/expected.json"},
}
#: the files a build writes into each folder of ``out_dir``, and its files elsewhere in it;
#: a new build replaces all of them
GENERATED = {
    **{f"data/clips/{cid}": ("sprite-*.jpg", "clip.json", "clip.mp4") for cid in CLIP_TITLES},
    "data/leakage": ("frame-*.jpg", "pairs.json"),
    "data/selftest": ("frames.bin", "expected.json"),
}
FIXED_FILES = ("data/results.json", *MODEL_FILES.values())
#: what each kind of report must hold (dotted paths), and the command that writes it
REQUIRED = {
    "summary": (("shipped.run", "shipped.seed", "twin.run", "groups", "ladder", "paired_leakage",
                 "regimes.models", "regimes.zero", "drives", "versions", "train_config"), "sightline report"),
    "run": (("run", "model", "protocol", "seed", "frame_lags", "checkpoint_sha256", "drive", "splits.firewall",
             "splits.block", "val.model.mae_deg", "test.n", "test.model", "test.baselines", "test.regimes",
             "test.zero_regimes", "test.pearson_r", "test.skill_vs_zero", "paired.n", "paired.model.mae_deg",
             "cross_drive.drive", "cross_drive.n", "cross_drive.model", "cross_drive.baselines",
             "cross_drive.pearson_r", "cross_drive.skill_vs_zero"), "sightline eval --all"),
    "leakage": (("drive", "seed", "dt_s", "splits", "paired", "gaps", "memoriser", "pairs"), "sightline leakage"),
    "legacy": (LEGACY_KEYS, "sightline legacy"),
}
#: the per-seed statistics of a summary group, and where a run report holds each
GROUP_STATS = {
    "val_mae_deg": "val.model.mae_deg",
    "test_mae_deg": "test.model.mae_deg",
    "test_r": "test.pearson_r",
    "test_skill": "test.skill_vs_zero",
    "cross_mae_deg": "cross_drive.model.mae_deg",
    "cross_r": "cross_drive.pearson_r",
    "cross_skill": "cross_drive.skill_vs_zero",
}


# --- the reports and what results.json copies from them ----------------------------------

@dataclass(frozen=True)
class Reports:
    """The reports the site data is made from."""

    summary: dict
    #: the run reports of the shipped run and of its twin, by run name
    runs: dict
    leakage: dict
    legacy: dict
    #: each report's file as ``<folder>/<name>``, in the order they were read
    files: list


def _at(tree, dotted: str):
    for key in dotted.split("."):
        tree = tree[key]
    return tree


def _lacks(report: dict, dotted: str) -> bool:
    node = report
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return True
        node = node[key]
    return False


def _require(report: dict, kind: str, label: str) -> None:
    """A ValueError naming what ``report`` lacks of what ``kind`` must hold, and the command
    that writes it again."""
    paths, made_by = REQUIRED[kind]
    missing = [path for path in paths if _lacks(report, path)]
    if missing:
        named = ", ".join(repr(path) for path in missing[:3]) + (" and more" if len(missing) > 3 else "")
        raise ValueError(f"{label} lacks {named}; write it again with `{made_by}`")


def _read_report(path: Path, made_by: str) -> dict:
    if not path.is_file():
        raise MissingDataError(f"no report at {path}; write it with `{made_by}`")
    try:
        data = json.loads(path.read_text())
    except ValueError as exc:
        raise ValueError(f"{path} is not JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a report")
    return data


def load_reports(summary_path: str | Path, reports_dir: str | Path) -> Reports:
    """The summary, the shipped and twin run reports (``<reports_dir>/runs/<run>.json``), and
    ``leakage.json`` and ``legacy.json`` from ``reports_dir``, each checked for every key the
    site data reads (a ValueError names what is missing)."""
    summary_path, reports_dir = Path(summary_path), Path(reports_dir)
    folder = reports_dir.name
    summary_label = f"{summary_path.parent.name}/{summary_path.name}"
    summary = _read_report(summary_path, "sightline report")
    _require(summary, "summary", summary_label)
    names = list(dict.fromkeys((summary["shipped"]["run"], summary["twin"]["run"])))
    runs = {}
    for name in names:
        runs[name] = _read_report(reports_dir / "runs" / f"{name}.json", "sightline eval --all")
        _require(runs[name], "run", f"{folder}/runs/{name}.json")
    leakage = _read_report(reports_dir / "leakage.json", "sightline leakage")
    _require(leakage, "leakage", f"{folder}/leakage.json")
    legacy = _read_report(reports_dir / "legacy.json", "sightline legacy")
    _require(legacy, "legacy", f"{folder}/legacy.json")
    files = [summary_label, *(f"{folder}/runs/{name}.json" for name in names),
             f"{folder}/leakage.json", f"{folder}/legacy.json"]
    return Reports(summary=summary, runs=runs, leakage=leakage, legacy=legacy, files=files)


def _mean_and_sd(value):
    """A statistic over seeds (``{"mean", "sd", "values"}``) without its values; anything else as is."""
    if isinstance(value, dict) and set(value) == {"mean", "sd", "values"}:
        return {"mean": value["mean"], "sd": value["sd"]}
    return value


def site_results(reports: Reports) -> dict:
    """results.json: what the page shows, copied from the reports value for value."""
    summary, leakage = reports.summary, reports.leakage
    shipped = reports.runs[summary["shipped"]["run"]]
    results = {
        "assets": ASSETS,
        "baselines": {
            "run": shipped["run"],
            "test": {key: shipped["test"][key] for key in ("n", "model", "baselines")},
            "cross_drive": {key: shipped["cross_drive"][key] for key in ("drive", "n", "model", "baselines")},
        },
        "drives": summary["drives"],
        "generated_from": reports.files,
        "groups": {key: {name: _mean_and_sd(value) for name, value in group.items()}
                   for key, group in summary["groups"].items()},
        "ladder": summary["ladder"],
        "leakage": {
            "drive": leakage["drive"],
            "seed": leakage["seed"],
            "splits": leakage["splits"],
            "paired": leakage["paired"],
            "median_frame_interval_s": leakage["dt_s"],
            "gaps": {protocol: {side: {k: v for k, v in gaps.items() if k != "median_s"}
                                for side, gaps in sides.items()}
                     for protocol, sides in leakage["gaps"].items()},
            "memoriser": leakage["memoriser"],
        },
        "legacy": {key: reports.legacy[key] for key in LEGACY_KEYS},
        "paired_leakage": summary["paired_leakage"],
        "protocol_names": PUBLIC_NAMES,
        "regimes": summary["regimes"],
        "shipped": summary["shipped"],
        "train_config": summary["train_config"],
        "twin": summary["twin"],
        "versions": summary["versions"],
    }
    return copy.deepcopy(results)


def _check_summary_against_runs(reports: Reports) -> None:
    """Refuse a summary whose figures for the shipped run or its twin are not the ones their
    run reports hold now: its per-seed group statistics, the paired leakage figures of their
    seed and the shipped run's regimes."""
    summary = reports.summary
    redo = "summarise the reports again with `sightline report`"
    for role in ("shipped", "twin"):
        run = summary[role]["run"]
        report = reports.runs[run]
        key = f"{report['model']}/{report['protocol']}"
        group = summary["groups"].get(key)
        if not isinstance(group, dict) or run not in group.get("runs", []):
            raise ValueError(f"summary.json has no {key} group holding run {run!r}; {redo}")
        i = group["runs"].index(run)
        figures = [("seed", group["seeds"][i], report["seed"])]
        figures += [(stat, group[stat]["values"][i], _at(report, where))
                    for stat, where in GROUP_STATS.items() if stat in group]
        for stat, in_summary, in_report in figures:
            if in_summary != in_report:
                raise ValueError(f"summary.json gives run {run!r} {stat} {in_summary!r}, but its run report "
                                 f"says {in_report!r}; {redo}")
    shipped, twin = reports.runs[summary["shipped"]["run"]], reports.runs[summary["twin"]["run"]]
    entries = [p for p in summary["paired_leakage"].get(shipped["model"], {}).get("per_seed", [])
               if p.get("seed") == shipped["seed"]]
    if not entries:
        raise ValueError(f"summary.json has no paired leakage figures for {shipped['model']} seed "
                         f"{shipped['seed']}; {redo}")
    for stat, in_summary, in_report in (
        ("paired n", entries[0]["n"], shipped["paired"]["n"]),
        ("blocked paired MAE", entries[0]["blocked_mae_deg"], shipped["paired"]["model"]["mae_deg"]),
        ("shuffled paired MAE", entries[0]["shuffled_mae_deg"], twin["paired"]["model"]["mae_deg"]),
    ):
        if in_summary != in_report:
            raise ValueError(f"summary.json gives seed {shipped['seed']} a {stat} of {in_summary!r}, but the "
                             f"run reports say {in_report!r}; {redo}")
    chosen = summary["regimes"]["models"].get(shipped["model"], {})
    if chosen.get("run") != shipped["run"] or chosen.get("regimes") != shipped["test"]["regimes"] \
            or summary["regimes"]["zero"] != shipped["test"]["zero_regimes"]:
        raise ValueError(f"summary.json's regimes are not those of the shipped run {shipped['run']!r} "
                         f"in its run report; {redo}")


def _check_reports(reports: Reports) -> dict:
    """The shipped run's drive, other drive, firewall and block, once every report agrees
    on them; a ValueError names the report that does not."""
    summary = reports.summary
    shipped_run, twin_run = summary["shipped"]["run"], summary["twin"]["run"]
    shipped, twin = reports.runs[shipped_run], reports.runs[twin_run]
    for name, report in ((shipped_run, shipped), (twin_run, twin)):
        if report["run"] != name:
            raise ValueError(f"runs/{name}.json is the report of run {report['run']!r}")
    if shipped["protocol"] != "blocked":
        raise ValueError(f"the shipped run {shipped_run!r} trained on the "
                         f"{PUBLIC_NAMES.get(shipped['protocol'], shipped['protocol'])} split, not the blocked one")
    if twin["protocol"] != "leaky":
        raise ValueError(f"the twin {twin_run!r} trained on the {PUBLIC_NAMES.get(twin['protocol'], twin['protocol'])} "
                         "split, not the shuffled one")
    if twin["seed"] != shipped["seed"]:
        raise ValueError(f"the twin {twin_run!r} has seed {twin['seed']}, the shipped run seed {shipped['seed']}")
    if (twin["model"], twin["frame_lags"]) != (shipped["model"], shipped["frame_lags"]):
        raise ValueError(f"the twin {twin_run!r} is {twin['model']} with frame lags {twin['frame_lags']}, "
                         f"but the shipped run is {shipped['model']} with {shipped['frame_lags']}")
    drive, cross_drive = shipped["drive"], shipped["cross_drive"]["drive"]
    if cross_drive == drive:
        raise ValueError(f"the shipped run's other drive is its own training drive {drive!r}")
    splits = {"firewall": shipped["splits"]["firewall"], "block": shipped["splits"]["block"]}
    leakage = reports.leakage
    if (leakage["drive"], leakage["splits"]) != (drive, splits):
        raise ValueError(f"leakage.json analyses drive {leakage['drive']!r} with {leakage['splits']}, "
                         f"but the shipped run trained on {drive!r} with {splits}")
    if reports.legacy["splits"] != splits:
        raise ValueError(f"legacy.json scores test splits made with {reports.legacy['splits']}, "
                         f"but the shipped run trained with {splits}")
    for name, info in reports.legacy.get("drives", {}).items():
        recorded = summary["drives"].get(name)
        if recorded is not None and info.get("label_sha256") != recorded.get("label_sha256"):
            raise ValueError(f"legacy.json scored other labels of drive {name!r} than the run reports")
    _check_summary_against_runs(reports)
    return {"drive": drive, "cross_drive": cross_drive, **splits}


# --- models -------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Model:
    key: str
    run: str
    name: str
    model: torch.nn.Module
    checkpoint_sha256: str


def _load_model(workspace: Workspace, key: str, run: str, report: dict) -> _Model:
    """A run's model, refused unless its best.pt is the checkpoint its report describes."""
    run_dir = workspace.run_dir(run)
    model, history = load_run_model(run_dir)
    digest = checkpoint_sha256(run_dir)
    if report["checkpoint_sha256"] != digest:
        raise UnusableRunError(f"run {run!r}: its best.pt is another checkpoint than the one its report "
                               "describes; evaluate it again (`sightline eval --all`) and summarise "
                               "again (`sightline report`)")
    if [int(lag) for lag in model.frame_lags] != report["frame_lags"]:
        raise UnusableRunError(f"run {run!r} stacks frame lags {list(model.frame_lags)}, "
                               f"but its report says {report['frame_lags']}")
    return _Model(key=key, run=run, name=history["config"]["model"], model=model.eval(),
                  checkpoint_sha256=digest)


def _onnx_outputs(path: Path, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Steering degrees (N,) and saliency maps (N, 66, 200), float32, from onnxruntime's CPU
    provider running the graph at ``path`` on one thread, ``EXPECTED_BATCH`` stacks at a time."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    angles, maps = [], []
    for start in range(0, len(inputs), EXPECTED_BATCH):
        angle, saliency = session.run(None, {INPUT_NAME: np.ascontiguousarray(inputs[start:start + EXPECTED_BATCH])})
        angles.append(np.asarray(angle, dtype=np.float32)[:, 0])
        maps.append(np.asarray(saliency, dtype=np.float32))
    return np.concatenate(angles), np.concatenate(maps)


# --- drives and clips -----------------------------------------------------------------------

def _open_drive(workspace: Workspace, drive: str, summary: dict):
    """A drive's labels and checked cache, refused unless it is the drive the reports describe."""
    labels, cache = open_drive_cache(workspace, drive)
    recorded = summary["drives"].get(drive) or {}
    found = (len(cache), cache.meta.get("label_sha256"))
    if found != (recorded.get("frames"), recorded.get("label_sha256")):
        raise ValueError(f"drive {drive!r} on disk ({found[0]} frames, labels {str(found[1])[:12]}) is not the "
                         f"drive the reports describe ({recorded.get('frames')} frames, labels "
                         f"{str(recorded.get('label_sha256'))[:12]})")
    return labels, cache


def stretches(frames: np.ndarray) -> list[tuple[int, int]]:
    """The runs of consecutive frame indices in ``frames``, as ``(start, stop)`` with ``stop``
    excluded, in order."""
    frames = np.unique(np.asarray(frames, dtype=np.int64))
    if frames.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(frames) != 1) + 1
    starts = np.r_[frames[0], frames[cuts]]
    stops = np.r_[frames[cuts - 1] + 1, frames[-1] + 1]
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def choose_window(angles_deg: np.ndarray, spans: list[tuple[int, int]], n: int, lead_in: int,
                  step: int = CLIP_STEP, max_abs_deg: float = CLIP_MAX_ABS_DEG) -> dict:
    """The clip window: ``n`` scored frames, with the ``lead_in`` frames before them in the
    same span and every one of those labels within ``max_abs_deg`` either way, whose labels
    have the largest (population) standard deviation.

    Windows start every ``step`` frames from the first that fits in each ``(start, stop)``
    span; ties go to the earliest. Returns its ``first_frame`` (the first scored frame), its
    ``label_sd_deg`` and how many ``windows`` within the limit were compared; a ValueError
    when none is.
    """
    angles = np.asarray(angles_deg, dtype=np.float64)
    best_first, best_sd, windows = None, -1.0, 0
    for start, stop in spans:
        for first in range(start + lead_in, stop - n + 1, step):
            if np.abs(angles[first - lead_in:first + n]).max() > max_abs_deg:
                continue
            windows += 1
            sd = float(np.std(angles[first:first + n]))
            if sd > best_sd:
                best_first, best_sd = first, sd
    if best_first is None:
        longest = max((stop - start for start, stop in spans), default=0)
        raise ValueError(f"no window of {n} frames, with the {lead_in} before it, fits inside a stretch with "
                         f"every label within {max_abs_deg:g} deg (the longest stretch has {longest} frames)")
    return {"first_frame": best_first, "label_sd_deg": best_sd, "windows": windows}


def min_gap_to_training(frames: np.ndarray, trained: np.ndarray, minimum: int) -> int:
    """The distance in frames from ``frames`` to the nearest of ``trained``; a ValueError
    when it is below ``minimum``."""
    gaps = frame_gaps(trained, frames)
    gap = int(gaps.min())
    if gap < minimum:
        raise ValueError(f"frame {int(frames[np.argmin(gaps)])} is {gap} frames from a training frame; "
                         f"every frame must be at least {minimum} away")
    return gap


def twin_exposure(frames: np.ndarray, trained: np.ndarray) -> dict:
    """How much of a clip the twin saw in training: ``twin_trained_frac``, the share of the
    clip's scored ``frames`` among ``trained``, the twin's training frames of the clip's drive,
    and ``twin_neighbour_frac``, the share within one frame of one of them. The twin trained
    on frames of its own drive only, so for the other drive's clip ``trained`` is empty and
    both are 0."""
    frames, trained = np.asarray(frames, dtype=np.int64), np.asarray(trained, dtype=np.int64)
    if trained.size == 0:
        return {"twin_trained_frac": 0.0, "twin_neighbour_frac": 0.0}
    return {"twin_trained_frac": float(np.isin(frames, trained).mean()),
            "twin_neighbour_frac": float((frame_gaps(trained, frames) <= 1).mean())}


def _rule(where: str, n: int, lead_in: int) -> str:
    return (f"the {n}-frame window with the largest standard deviation of its steering labels, among the "
            f"windows that fit, with the {lead_in} frames before them, inside {where} and whose labels, those "
            f"{lead_in} frames included, all stay within {CLIP_MAX_ABS_DEG:g} degrees either way, so that the "
            "clip shows road following rather than parking or intersection manoeuvres; windows start every "
            f"{CLIP_STEP} frames and ties go to the earliest")


# --- images and video -------------------------------------------------------------------------

def jpeg_bytes(rgb: np.ndarray, quality: int, subsampling: int | None = None) -> bytes:
    """An H x W x 3 uint8 RGB image as a baseline JPEG (Huffman tables optimised): the same
    pixels and settings always give the same bytes. ``subsampling`` 0 keeps full chroma."""
    pixels = np.asarray(rgb)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError(f"expected an H x W x 3 uint8 RGB image, got {pixels.dtype} {pixels.shape}")
    options = {"quality": quality, "optimize": True}
    if subsampling is not None:
        options["subsampling"] = subsampling
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(pixels)).save(buffer, format="JPEG", **options)
    return buffer.getvalue()


def sprite_sheets(frames: np.ndarray) -> list[bytes]:
    """Model input frames (N x 66 x 200 x 3 uint8) as JPEG sheets of up to
    ``FRAMES_PER_SHEET`` frames stacked top to bottom, q95 without chroma subsampling."""
    frames = np.asarray(frames)
    return [jpeg_bytes(frames[start:start + FRAMES_PER_SHEET].reshape(-1, INPUT_WIDTH, 3), SPRITE_QUALITY,
                       subsampling=0)
            for start in range(0, len(frames), FRAMES_PER_SHEET)]


def decode_sheets(paths) -> np.ndarray:
    """The frames of sprite sheets, in order, as Pillow decodes them: N x 66 x 200 x 3 uint8."""
    frames = []
    for path in paths:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if pixels.shape[1] != INPUT_WIDTH or pixels.shape[0] % INPUT_HEIGHT:
            raise ValueError(f"{path} is not a sheet of {INPUT_WIDTH}x{INPUT_HEIGHT} frames")
        frames.append(pixels.reshape(-1, INPUT_HEIGHT, INPUT_WIDTH, 3))
    return np.concatenate(frames)


def frame_digests(frames: np.ndarray) -> list[str]:
    """sha256 of each frame's RGB bytes, rows first."""
    return [hashlib.sha256(np.ascontiguousarray(frame, dtype=np.uint8).tobytes()).hexdigest() for frame in frames]


def _camera_frame(labels, index: int) -> np.ndarray:
    """Frame ``index`` of a drive, the camera image of its label row, as H x W x 3 RGB."""
    if not 0 <= index < len(labels):
        raise ValueError(f"drive {labels.root.name!r} has no frame {index} ({len(labels)} frames)")
    path = labels.frame_path(index)
    if not path.is_file():
        raise MissingDataError(f"frame {index} of drive {labels.root.name!r} is missing: expected {path}")
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if (rgb.shape[1], rgb.shape[0]) != CAMERA_SIZE:
        raise ValueError(f"{path} is {rgb.shape[1]}x{rgb.shape[0]}, not a "
                         f"{CAMERA_SIZE[0]}x{CAMERA_SIZE[1]} camera frame")
    return rgb


def _find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise MissingDataError("ffmpeg is not on the PATH, and the clip videos need it; install ffmpeg, "
                               "or pass --no-video to write everything else")
    return path


def _ffmpeg_version(ffmpeg: str) -> str:
    out = subprocess.run([ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True).stdout
    words = (out.splitlines() or [""])[0].split()
    return words[2] if words[:2] == ["ffmpeg", "version"] and len(words) > 2 else " ".join(words)


def encode_video(ffmpeg: str, frames: np.ndarray, out_path: Path, fps: float = CLIP_FPS) -> tuple[int, int]:
    """``frames`` (N x H x W x 3 uint8 RGB) as an H.264 MP4 (yuv420p, CRF 23, preset slow,
    index first) at ``fps``, with no audio. 4:2:0 needs an even width and height, so an odd
    one gets its last column or row repeated once. Returns the video's (width, height)."""
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.shape[2] % 2:
        frames = np.concatenate([frames, frames[:, :, -1:]], axis=2)
    if frames.shape[1] % 2:
        frames = np.concatenate([frames, frames[:, -1:]], axis=1)
    _, height, width, _ = frames.shape
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-framerate", f"{fps:g}",
               "-i", "pipe:0", "-an", *VIDEO_OPTIONS, "-f", "mp4", str(out_path)]
    done = subprocess.run(command, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    if done.returncode != 0:
        said = done.stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError(f"ffmpeg could not encode {out_path.name}: "
                           f"{said[-1] if said else f'exit status {done.returncode}'}")
    return width, height


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- the build ------------------------------------------------------------------------------

def _move(source: Path, target: Path, created: list[Path]) -> None:
    """Move ``source`` to ``target``, making the folders ``target`` lacks (each recorded in
    ``created``, outermost first)."""
    missing, parent = [], target.parent
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for folder in reversed(missing):
        folder.mkdir()
        created.append(folder)
    os.replace(source, target)


def _swap_in(new_root: Path, out_dir: Path, new_export: Path, export_path: Path) -> None:
    """Put a staged build in place: the previous build's files and export.json are moved into
    a folder beside ``out_dir``, then the files under ``new_root`` into ``out_dir`` and
    ``new_export`` to ``export_path``. If anything fails, every new file and folder is removed
    and the old files are moved back, so ``out_dir`` and ``export_path`` are as they were; the
    folder of old files is removed either way, unless moving them back failed, and then the
    error names it."""
    old = sorted({path for folder, patterns in GENERATED.items() for pattern in patterns
                  for path in (out_dir / folder).glob(pattern) if path.is_file()}
                 | {out_dir / name for name in FIXED_FILES if (out_dir / name).is_file()}
                 | ({export_path} if export_path.is_file() else set()))
    aside = Path(tempfile.mkdtemp(dir=out_dir.parent, prefix=f".{out_dir.name}-previous-"))
    moved, placed, created = [], [], []
    try:
        for k, path in enumerate(old):
            os.replace(path, aside / str(k))
            moved.append((aside / str(k), path))
        for path in sorted(p for p in new_root.rglob("*") if p.is_file()):
            target = out_dir / path.relative_to(new_root)
            _move(path, target, created)
            placed.append(target)
        _move(new_export, export_path, created)
        placed.append(export_path)
    except BaseException as failure:
        for path in reversed(placed):
            with contextlib.suppress(OSError):
                path.unlink()
        for folder in reversed(created):
            with contextlib.suppress(OSError):
                folder.rmdir()
        try:
            for kept, path in reversed(moved):
                os.replace(kept, path)
        except OSError as exc:
            raise RuntimeError(f"the new site data could not be put in place ({failure}), and the previous "
                               f"files could not all be put back ({exc}); they are in {aside}") from failure
        shutil.rmtree(aside, ignore_errors=True)
        raise
    shutil.rmtree(aside, ignore_errors=True)


def _versions(ffmpeg: str | None) -> dict:
    import onnx
    import onnxruntime
    import PIL

    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__, "pillow": PIL.__version__,
            "ffmpeg": None if ffmpeg is None else _ffmpeg_version(ffmpeg)}


def build_site_data(summary_path: str | Path, workspace: Workspace, out_dir: str | Path = Path("web/public"),
                    reports_dir: str | Path = Path("reports"), *, clip_frames: int = CLIP_FRAMES,
                    video: bool = True) -> dict:
    """Write the site data (see the module docstring) and ``reports_dir/export.json``; return
    what export.json holds.

    Refuses, before writing anything: a missing ffmpeg when ``video`` is on; a report that
    lacks a key the site data reads; reports that disagree on the runs, the drive or the split
    settings, or a summary whose figures for the shipped run or its twin differ from their run
    reports; a run whose best.pt is not the checkpoint its report describes; a cache that is
    missing or not built from the drive's labels as they are now; a drive whose labels are not
    the ones the reports were made from; a 2018 clip closer than 2·firewall + 1 frames to a
    frame the shipped run trained on.
    """
    if clip_frames < SELFTEST_INPUTS:
        raise ValueError(f"a clip needs at least {SELFTEST_INPUTS} frames (the self-test inputs), got {clip_frames}")
    out_dir, reports_dir = Path(out_dir), Path(reports_dir)
    ffmpeg = _find_ffmpeg() if video else None
    reports = load_reports(summary_path, reports_dir)
    setting = _check_reports(reports)
    summary = reports.summary
    models = [_load_model(workspace, key, summary[which]["run"], reports.runs[summary[which]["run"]])
              for key, which in (("sightline", "shipped"), ("shuffled", "twin"))]
    lags = tuple(int(lag) for lag in models[0].model.frame_lags)
    lead_in = max(lags, default=0)
    drives = {"c2018": setting["drive"], "c2017": setting["cross_drive"]}
    opened = {drive: _open_drive(workspace, drive, summary) for drive in dict.fromkeys(drives.values())}
    results = site_results(reports)

    firewall = setting["firewall"]
    train_cache = opened[setting["drive"]][1]
    # the frames each run held out or trained on, as it saved them
    shipped_dir = workspace.run_dir(summary["shipped"]["run"])
    held_out, trained = load_split_indices(shipped_dir, "test"), load_split_indices(shipped_dir, "train")
    twin_trained = load_split_indices(workspace.run_dir(summary["twin"]["run"]), "train")
    twin_drive = reports.runs[summary["twin"]["run"]]["drive"]
    places = {
        "c2018": (stretches(held_out), f"one blocked test stretch of {setting['drive']} (every frame at least "
                                       f"{2 * firewall + 1} frames from training)"),
        "c2017": ([(0, len(opened[setting["cross_drive"]][1]))], f"the whole of {setting['cross_drive']}"),
    }
    clips = {}
    for cid, drive in drives.items():
        spans, where = places[cid]
        try:
            window = choose_window(opened[drive][1].angles_deg, spans, clip_frames, lead_in)
        except ValueError as exc:
            raise ValueError(f"no {clip_frames}-frame clip fits in {where}: {exc}") from None
        first = window["first_frame"]
        seen = twin_trained if drive == twin_drive else np.empty(0, dtype=np.int64)
        clips[cid] = {"drive": drive, "rule": _rule(where, clip_frames, lead_in), **window,
                      "min_frames_from_training": None,
                      **twin_exposure(np.arange(first, first + clip_frames), seen)}
    # the page says the 2018 clip shows minutes the model never trained on: measured, not assumed
    first = clips["c2018"]["first_frame"]
    try:
        clips["c2018"]["min_frames_from_training"] = min_gap_to_training(
            np.arange(first - lead_in, first + clip_frames), trained, 2 * firewall + 1)
    except ValueError as exc:
        raise ValueError(f"the c2018 clip of {setting['drive']} is too close to training: {exc}") from None

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir.parent, prefix=f".{out_dir.name}-") as tmp:
        staging = Path(tmp) / "new"
        # the sprites, and the model inputs as Pillow decodes them
        stacks, sheets = {}, {}
        for cid, clip in clips.items():
            folder = staging / "data" / "clips" / cid
            folder.mkdir(parents=True)
            first = clip["first_frame"]
            frames = np.asarray(opened[clip["drive"]][1].inputs[first - lead_in:first + clip_frames])
            sheets[cid] = [f"sprite-{k:02d}.jpg" for k in range((len(frames) + FRAMES_PER_SHEET - 1)
                                                                 // FRAMES_PER_SHEET)]
            for name, data in zip(sheets[cid], sprite_sheets(frames)):
                (folder / name).write_bytes(data)
            decoded = decode_sheets([folder / name for name in sheets[cid]])
            clip["decoded_sha256"] = frame_digests(decoded)
            stacks[cid] = stack_frames(decoded, np.arange(lead_in, lead_in + clip_frames), lags)

        # the self-test inputs: raw cache frames of the 2018 clip, spread over it
        positions = np.unique(np.linspace(0, clip_frames - 1, SELFTEST_INPUTS).round().astype(np.int64))
        selftest_frames = clips["c2018"]["first_frame"] + positions
        selftest_inputs = stack_frames(train_cache.inputs, selftest_frames, lags)

        # the models, kept only if onnxruntime reproduces PyTorch on every input the page feeds
        # them; the expected outputs then come from the very files the page loads
        check = np.concatenate([stacks["c2018"], stacks["c2017"], selftest_inputs])
        exported = {m.key: export_onnx(m.model, staging / MODEL_FILES[m.key], check, model_name=m.name, run=m.run,
                                       max_bytes=MAX_MODEL_BYTES, angle_tolerance_deg=SHIPPED_ANGLE_TOLERANCE_DEG,
                                       saliency_tolerance=SHIPPED_SALIENCY_TOLERANCE)
                    for m in models}
        expected = {cid: {m.key: _onnx_outputs(staging / MODEL_FILES[m.key], stacks[cid])[0] for m in models}
                    for cid in clips}
        selftest = {m.key: _onnx_outputs(staging / MODEL_FILES[m.key], selftest_inputs) for m in models}

        for cid, clip in clips.items():
            folder = staging / "data" / "clips" / cid
            labels, cache = opened[clip["drive"]]
            first = clip["first_frame"]
            truth = cache.angles_deg[first:first + clip_frames]
            clip["clip_mae_deg"] = {
                **{key: float(np.abs(values.astype(np.float64) - truth.astype(np.float64)).mean())
                   for key, values in expected[cid].items()},
                "zero": float(np.abs(truth.astype(np.float64)).mean()),
            }
            size = None
            clip["video"] = None
            if ffmpeg is not None:
                camera = np.stack([_camera_frame(labels, i) for i in range(first, first + clip_frames)])
                size = encode_video(ffmpeg, camera, folder / "clip.mp4")
                clip["video"] = {"file": f"data/clips/{cid}/clip.mp4", "bytes": (folder / "clip.mp4").stat().st_size,
                                 "sha256": _file_sha256(folder / "clip.mp4"), "width": size[0], "height": size[1]}
            write_report(folder / "clip.json", {
                "id": cid,
                "drive": clip["drive"],
                "title": CLIP_TITLES[cid],
                "fps": CLIP_FPS,
                "n": clip_frames,
                "lead_in": lead_in,
                "first_frame": first,
                "frame_lags": list(lags),
                "rule": clip["rule"],
                "labels_deg": [float(a) for a in truth],
                "expected": {key: [float(a) for a in values] for key, values in expected[cid].items()},
                "tolerance": {"angle_deg": ANGLE_TOLERANCE_DEG},
                "sprite": {"files": sheets[cid], "frames_per_sheet": FRAMES_PER_SHEET, "frame_w": INPUT_WIDTH,
                           "frame_h": INPUT_HEIGHT, "decoded_sha256": clip["decoded_sha256"]},
                "video": None if size is None else "clip.mp4",
                "video_size": None if size is None else list(size),
                "camera_size": list(CAMERA_SIZE),
                "clip_mae_deg": clip["clip_mae_deg"],
                "twin_trained_frac": clip["twin_trained_frac"],
                "twin_neighbour_frac": clip["twin_neighbour_frac"],
            })

        folder = staging / "data" / "selftest"
        folder.mkdir(parents=True)
        (folder / "frames.bin").write_bytes(np.ascontiguousarray(selftest_inputs).tobytes())
        write_report(folder / "expected.json", {
            "clip": "c2018",
            "drive": setting["drive"],
            "frames": [int(f) for f in selftest_frames],
            "shape": list(selftest_inputs.shape),
            "dtype": "uint8",
            "angles": {key: [float(a) for a in angles] for key, (angles, _) in selftest.items()},
            "saliency_shape": list(next(iter(selftest.values()))[1].shape),
            "saliency_f32_b64": {key: base64.b64encode(maps.astype("<f4").tobytes()).decode("ascii")
                                 for key, (_, maps) in selftest.items()},
            "tolerance": {"angle_deg": ANGLE_TOLERANCE_DEG, "saliency": SALIENCY_TOLERANCE},
        })

        folder = staging / "data" / "leakage"
        folder.mkdir(parents=True)
        leakage, labels = reports.leakage, opened[setting["drive"]][0]
        named = sorted({pair["frame"] for pair in leakage["pairs"]}
                       | {pair[p]["nn_frame"] for pair in leakage["pairs"] for p in LOOKUP_PROTOCOLS})
        for frame in named:
            (folder / f"frame-{frame}.jpg").write_bytes(jpeg_bytes(_camera_frame(labels, frame), PAIR_QUALITY))
        pairs = []
        for pair in leakage["pairs"]:
            entry = {"frame": pair["frame"], "image": f"frame-{pair['frame']}.jpg", "angle_deg": pair["angle_deg"]}
            for protocol in LOOKUP_PROTOCOLS:
                side = pair[protocol]
                entry[protocol] = {
                    "nn_frame": side["nn_frame"],
                    "image": f"frame-{side['nn_frame']}.jpg",
                    "gap_frames": side["gap_frames"],
                    # timestamps are epoch seconds, so their differences carry float noise
                    "gap_s": None if side["gap_s"] is None else round(side["gap_s"], 3),
                    "pixel_rmse": side["pixel_rmse"],
                    "nn_angle_deg": side["nn_angle_deg"],
                }
            pairs.append(entry)
        write_report(folder / "pairs.json", {"drive": leakage["drive"], "seed": leakage["seed"],
                                             "median_frame_interval_s": leakage["dt_s"], "pairs": pairs})

        write_report(staging / "data" / "results.json", results)
        export = {
            "code_version": code_version(),
            "device": "cpu",
            "models": {
                m.key: {
                    "run": m.run,
                    "model": m.name,
                    "file": MODEL_FILES[m.key],
                    "bytes": exported[m.key]["bytes"],
                    "sha256": exported[m.key]["sha256"],
                    "checkpoint_sha256": m.checkpoint_sha256,
                    "frame_lags": list(lags),
                    "opset": exported[m.key]["opset"],
                    "parity": {"frames": exported[m.key]["check_frames"],
                               "max_angle_diff_deg": exported[m.key]["max_angle_diff_deg"],
                               "max_saliency_diff": exported[m.key]["max_saliency_diff"],
                               "angle_tolerance_deg": exported[m.key]["angle_tolerance_deg"],
                               "saliency_tolerance": exported[m.key]["saliency_tolerance"]},
                }
                for m in models
            },
            "clips": {
                cid: {"drive": clip["drive"], "first_frame": clip["first_frame"], "n": clip_frames,
                      "lead_in": lead_in, "rule": clip["rule"], "label_sd_deg": clip["label_sd_deg"],
                      "windows": clip["windows"], "min_frames_from_training": clip["min_frames_from_training"],
                      "twin_trained_frac": clip["twin_trained_frac"],
                      "twin_neighbour_frac": clip["twin_neighbour_frac"],
                      "clip_mae_deg": clip["clip_mae_deg"], "video": clip["video"]}
                for cid, clip in clips.items()
            },
            "selftest": {"clip": "c2018", "frames": [int(f) for f in selftest_frames]},
            "versions": _versions(ffmpeg),
        }
        write_report(Path(tmp) / "export.json", export)
        _swap_in(staging, out_dir, Path(tmp) / "export.json", reports_dir / "export.json")
    return export
