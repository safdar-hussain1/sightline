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
    the leakage analysis, the drives, the training recipe, the library versions and the
    evaluation's ``cutoffs`` (the steering regimes' edges, the smallest angle the sign
    agreement counts and the tolerances of the shares of frames within them, in degrees, so
    the page never types them). The leakage report's ``median_s`` (a median frame count times
    the median frame interval, not a time anything took) is left out; the interval itself is
    ``median_frame_interval_s``. ``parity`` gives, for each of the page's models, how closely
    onnxruntime reproduced PyTorch when this build exported it, as export.json records it: the
    run, the largest angle and saliency differences over the ``stacks`` frame stacks checked,
    the gates they met and the onnxruntime version. ``generated_from`` names the files it is
    made from: the summary, every run report it lists, the leakage and standard-recipe reports,
    in the order they are read, and last the export record this build writes (export.json).

    Three more fields are made here. ``drive_profile`` holds, for each drive, its steering
    labels as ``repair_dropouts`` leaves them (read from the label files, never from the
    cache) averaged over buckets of ``bucket_frames`` consecutive frames: per bucket its
    ``start_frame``, its mean angle ``mean_deg`` and its mean absolute angle ``mean_abs_deg``,
    in degrees to 0.01; for a drive with timestamps also each bucket's ``start_s``, seconds
    since the drive's first frame to the millisecond, and the drive's ``duration_s``. The
    2017 drive has no timestamps, so both are null. The last bucket holds the frames left
    over, so every frame is in exactly one bucket. The labels must be the ones the reports
    were scored on: as many frames, as many repaired dropouts and the same label file
    (sha256) as the summary records for the drive. ``split_rules`` states how each protocol
    cuts a drive: the block and firewall every run report, the leakage report and the
    standard-recipe report agree on, and the splits module's pattern and fractions.
    ``split_sizes`` gives each protocol's train, validation and test frame counts from the run
    reports, and the frames its firewalls dropped; every run of a protocol must have recorded
    the same counts, they must be the ones the splits module makes from those settings, and
    every blocked and shuffled run's paired frames (their count and digest) the ones it pairs;
    the build also checks the shipped run's and its twin's saved splits frame by frame.

    ``timing`` gives, for each drive, how far apart in time its frames are, measured from the
    timestamps of its label file (the labels ``drive_profile`` reads), never estimated from the
    median frame interval: ``frame_interval_s``, the spread of the time from each frame to the
    next over the whole drive; ``lag_gap_s``, for each frame lag of the shipped run (its
    report's ``frame_lags``), the spread of the time from frame ``i - lag`` to frame ``i`` over
    every frame ``i`` from ``lag`` on; and ``closest_eval_s``, on the drive the runs trained on
    (null on any other): for the blocked and the later-in-the-drive split, as the splits module
    makes them with the block and firewall every run recorded (``split_rules``), and for their
    validation and test frames in turn, the evaluation frame closest in time to a training frame
    (``eval_frame``), that training frame (``train_frame``) and the time between them
    (``gap_s``). ``gap_s`` is measured between the scored frames, rounded to the nearest
    millisecond; with lag 2 the model's inputs reach two frames nearer, so any text that states
    it as a lower bound must round down. The shuffled split has none: its evaluation frames sit
    next to training frames by design, and ``leakage.gaps`` gives that distance in frames. A
    spread is the least value, the 5th percentile, the median, the mean, the 95th percentile and
    the largest (``min``, ``p05``, ``median``, ``mean``, ``p95``, ``max``). A drive without
    timestamps (the 2017 one) has null. Every time is in seconds to the millisecond, the
    resolution of the timestamps; ``drive_timing`` says how they are worked out and rounded.
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
    column (``video_size`` against ``camera_size``). ``first_s`` and ``last_s`` are the times
    of the first and last scored frames in seconds since the drive's first frame, to the
    millisecond as in the drive profile, and null for a drive without timestamps (the 2017
    one). ``clip_mae_deg`` is the mean absolute error on this clip alone, for each model and
    for steering straight ahead.
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

Every check that needs only the inputs runs before anything is written: the reports, the
checkpoints and the saved splits of the shipped run and its twin, the caches, the drives and the
clip rule. The build is then staged in a folder beside ``out_dir`` (``web/.public-*`` for
``web/public``), which is always removed, and export.json in a hidden file beside its own final
path. The models are exported there and
must pass their parity gates and the size limit, the sprites are written and decoded and the
videos encoded, so a failure at any of those steps leaves ``out_dir`` and export.json as
they were. Only then is the build swapped in: the previous build's files are moved into
another folder beside ``out_dir`` and the previous export.json beside itself, then the new
files are moved in, each move a rename within one folder's volume. If a move fails, the new
files are removed and the old ones moved back, so ``out_dir`` and export.json are as they
were; if moving them back fails as well, the error says so and names where the previous
files are, and they are left there. Run again on the same machine with the same library
versions, the same reports, checkpoints and drives give the same bytes apart from the
videos; elsewhere the values computed in floating point (expected outputs, self-test angles
and saliency, parity figures) can differ in their last bits.
"""

from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import platform
import secrets
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sightline.data.frames import stack_frames
from sightline.data.labels import DriveLabels
from sightline.data.preprocess import INPUT_HEIGHT, INPUT_WIDTH
from sightline.data.splits import (
    BLOCK_PATTERN,
    PROTOCOLS,
    PUBLIC_NAMES,
    TRAIN_FRAC,
    VAL_FRAC,
    frames_sha256,
    make_split,
    paired_frames,
)
from sightline.data.verify import label_sha256
from sightline.export.onnx_export import (
    ANGLE_TOLERANCE_DEG,
    INPUT_NAME,
    SALIENCY_TOLERANCE,
    export_onnx,
    open_drive_cache,
)
from sightline.leakage import LOOKUP_PROTOCOLS, frame_gaps
from sightline.paths import MissingDataError, UnusableRunError, Workspace
from sightline.reporting import PAIRED_PROTOCOLS, checkpoint_sha256, load_run_model, load_split_indices, write_report
from sightline.train import code_version

__all__ = [
    "ASSETS",
    "BUCKET_FRAMES",
    "CAMERA_SIZE",
    "CLIP_FPS",
    "CLIP_FRAMES",
    "CLIP_MAX_ABS_DEG",
    "CLIP_STEP",
    "CLIP_TITLES",
    "FRAMES_PER_SHEET",
    "MAX_MODEL_BYTES",
    "MODEL_FILES",
    "PERCENTILE_METHOD",
    "Reports",
    "SHIPPED_ANGLE_TOLERANCE_DEG",
    "SHIPPED_SALIENCY_TOLERANCE",
    "SPREAD_PERCENTILES",
    "TIMED_PROTOCOLS",
    "build_site_data",
    "choose_window",
    "clip_times",
    "decode_sheets",
    "drive_profile",
    "drive_timing",
    "encode_video",
    "export_parity",
    "frame_digests",
    "frame_timing",
    "jpeg_bytes",
    "leakage_pairs",
    "load_reports",
    "min_gap_to_training",
    "missing_key",
    "require_what_the_site_reads",
    "site_results",
    "split_rules",
    "split_sizes",
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
#: the clips' playback rate, frames per second: the 2018 drive's measured rate, one over the
#: median interval between its timestamps (57.0 ms, so 17.54 fps), to one decimal, as the demo
#: command plays; the 2017 drive has no timestamps and plays at the same rate
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
#: Measured on six checkpoints, the two-frame SightlineNet runs of seeds 0 to 2 on the blocked and
#: on the shuffled split: 1.2e-4 to 1.8e-4 deg on those inputs and up to 3.4e-4 deg on each run's
#: own test frames (float32 rounding ahead of the output scale), and under 2e-5 in saliency; a
#: structural export fault is off by whole degrees.
SHIPPED_ANGLE_TOLERANCE_DEG = 5e-4
SHIPPED_SALIENCY_TOLERANCE = 1e-4
#: a drive profile averages this many consecutive frames into one point: 998 points for the
#: 2018 drive (63,825 frames), 710 for the 2017 one (45,406)
BUCKET_FRAMES = 64
#: a drive profile's angles are rounded to 0.01 degree, the resolution of the logged labels,
#: and its times to the millisecond, the resolution of the timestamps
PROFILE_ANGLE_DECIMALS = 2
PROFILE_TIME_DECIMALS = 3
#: the percentiles a spread of frame times gives besides its least, largest and mean values, by key
SPREAD_PERCENTILES = {"p05": 5, "median": 50, "p95": 95}
#: how numpy computes them: its default (linear interpolation between the two nearest values in
#: order, Hyndman and Fan's type 7, whose 50th percentile is ``np.median``), named so that another
#: default cannot change them
PERCENTILE_METHOD = "linear"
#: the splits whose closest evaluation frames the timing measures in seconds: the two that keep their
#: evaluation frames away from training; the shuffled split puts them next to it by design
TIMED_PROTOCOLS = ("blocked", "forward")
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
#: a split score's steering regimes and baselines, by their keys in the reports
REGIME_KEYS = ("straight", "gentle", "sharp")
BASELINE_KEYS = ("zero", "mean", "zoh", "extrap")
#: what each example frame pair of the leakage report gives for each split searched
PAIR_KEYS = ("nn_frame", "gap_frames", "gap_s", "pixel_rmse", "nn_angle_deg")


def _regimes(prefix: str) -> tuple[str, ...]:
    """The keys of a regime breakdown under ``prefix``."""
    return (*(f"{prefix}.{regime}.{key}" for regime in REGIME_KEYS for key in ("n", "mae_deg")),
            f"{prefix}.balanced_mae_deg")


def _scores(split: str) -> tuple[str, ...]:
    """What the site reads of a run report's score on ``split`` (``test`` or ``cross_drive``)."""
    return (f"{split}.n", f"{split}.model.n", f"{split}.model.mae_deg", f"{split}.baselines.zero.n",
            *(f"{split}.baselines.{name}.mae_deg" for name in BASELINE_KEYS), f"{split}.pearson_r",
            f"{split}.skill_vs_zero")


#: everything the site reads of each kind of report, as dotted paths, with the command that writes
#: the report: what this module reads, and what the page reads of the values results.json copies
#: whole. ``*`` stands for every key of a mapping or every item of a list, of which there must be
#: one at least. ``load_reports`` checks the reports before anything is computed, and
#: ``export_parity`` the export record, so a report made by an older command is refused in one
#: line naming what it lacks, never read half-way. What this module reads is listed in full, which a
#: test checks by taking every key of the reports away in turn: a key this module reads but this list
#: does not name makes the build fail some other way. What the page reads is listed from the page's
#: code by hand, as nothing in this package reads the page; a key the page starts to read must be
#: added here with it. A ``*`` never names an entry, so which drives, groups, models or protocols the
#: page needs is not checked here.
REQUIRED = {
    "summary": ((
        "shipped.run", "shipped.seed", "shipped.model", "shipped.protocol", "twin.run", "runs", "versions",
        "train_config", "cutoffs.regime_bins_deg", "cutoffs.sign_min_abs_deg", "cutoffs.tolerances_deg",
        *(f"drives.*.{key}" for key in ("frames", "label_sha256", "repaired_labels", "median_dt_s")),
        *(f"groups.*.{key}" for key in ("model", "protocol", "runs", "seeds", "test_zero_mae_deg",
                                        "cross_zero_mae_deg")),
        *(f"groups.*.{stat}.{part}" for stat in GROUP_STATS for part in ("mean", "sd", "values")),
        *(f"ladder.*.*.{key}" for key in ("rung", "protocol", "split", "drive", "seeds", "mae_deg.mean",
                                          "mae_deg.sd", "zero_mae_deg")),
        *(f"paired_leakage.*.per_seed.*.{key}" for key in ("seed", "n", "shuffled_mae_deg", "blocked_mae_deg",
                                                            "ratio", "difference_deg", "zero_mae_deg")),
        *(f"paired_leakage.*.{stat}.{part}" for stat in ("ratio", "difference_deg", "shuffled_mae_deg",
                                                          "blocked_mae_deg") for part in ("mean", "sd")),
        "paired_leakage.*.ratio.values", "regimes.protocol", "regimes.split", *_regimes("regimes.zero"),
        "regimes.models.*.run", "regimes.models.*.seed", *_regimes("regimes.models.*.regimes"),
    ), "sightline report"),
    #: the shipped run's report and its twin's; the shipped run's ``frame_lags`` are also the lags whose
    #: gaps in time ``timing`` measures, and its ``drive`` and split settings the splits it times
    "run": ((
        "run", "model", "protocol", "seed", "frame_lags", "checkpoint_sha256", "drive", "splits.firewall",
        "splits.block", "val.model.mae_deg", *_scores("test"), *_regimes("test.regimes"),
        *_regimes("test.zero_regimes"), "paired.n", "paired.frames_sha256", "paired.model.mae_deg",
        "cross_drive.drive", *_scores("cross_drive"),
    ), "sightline eval --all"),
    #: every run report the summary lists, for the split sizes and the checks on them
    "splits": (("run", "model", "protocol", "seed", "drive", "drives.*.frames", "splits.train_n", "splits.val_n",
                "splits.test_n", "splits.firewall", "splits.block"), "sightline eval --all"),
    #: every run report of a protocol scored on the paired frames too (``PAIRED_PROTOCOLS``)
    "paired": (("paired.n", "paired.frames_sha256"), "sightline eval --all"),
    "leakage": ((
        "drive", "seed", "dt_s", "splits.firewall", "splits.block", "paired.n", "paired.frames_sha256",
        "gaps.*.test.frac_within_1", "gaps.*.test.min_frames",
        *(f"memoriser.{protocol}.{key}" for protocol in LOOKUP_PROTOCOLS for key in ("n", "mae_deg")),
        "pairs.*.frame", "pairs.*.angle_deg",
        *(f"pairs.*.{protocol}.{key}" for protocol in LOOKUP_PROTOCOLS for key in PAIR_KEYS),
    ), "sightline leakage"),
    "legacy": ((
        *LEGACY_KEYS, "splits.firewall", "splits.block", "drives.*.label_sha256", "port_parity.frames",
        "port_parity.max_abs_diff_deg", "eval.*.model.n", "eval.*.model.mae_deg", "eval.*.zero_mae_deg",
        "atan_cap.max_abs_pred_deg", "atan_cap.labels_beyond_180.*.n",
    ), "sightline legacy"),
    #: what results.json's ``parity`` copies of an export record (export.json), for each page model
    "export": ((
        *(f"models.{key}.{field}" for key in MODEL_FILES for field in (
            "run", "checkpoint_sha256", "parity.frames", "parity.max_angle_diff_deg", "parity.max_saliency_diff",
            "parity.angle_tolerance_deg", "parity.saliency_tolerance")),
        "versions.onnxruntime",
    ), "sightline site-data"),
}


# --- the reports and what results.json copies from them ----------------------------------

@dataclass(frozen=True)
class Reports:
    """The reports the site data is made from."""

    summary: dict
    #: the run reports of every run the summary lists, by run name, the shipped run's and its
    #: twin's first
    runs: dict
    leakage: dict
    legacy: dict
    #: each report's file as ``<folder>/<name>``, in the order they were read
    files: list


def _at(tree, dotted: str):
    for key in dotted.split("."):
        tree = tree[key]
    return tree


def missing_key(tree, dotted: str) -> str | None:
    """Where ``tree`` lacks the dotted path ``dotted`` (``*`` stands for every key of a mapping or
    every item of a list, and needs one at least): the path as far as its first missing key, or
    ``<path>.*`` for an empty mapping or list; None when ``tree`` holds all of it."""
    def walk(node, keys: list[str], walked: list[str]) -> str | None:
        if not keys:
            return None
        key, rest = keys[0], keys[1:]
        if key == "*":
            if isinstance(node, dict) and node:
                children = node.items()
            elif isinstance(node, list) and node:
                children = enumerate(node)
            else:
                return ".".join([*walked, "*"])
            return next((found for name, child in children
                         if (found := walk(child, rest, [*walked, str(name)])) is not None), None)
        if not isinstance(node, dict) or key not in node:
            return ".".join([*walked, key])
        return walk(node[key], rest, [*walked, key])

    return walk(tree, dotted.split("."), [])


def _require(report: dict, kind: str, label: str) -> None:
    """A ValueError naming the first places where ``report`` lacks what ``kind`` must hold
    (``REQUIRED``), and the command that writes it again."""
    paths, made_by = REQUIRED[kind]
    missing = []
    for path in paths:
        where = missing_key(report, path)
        if where is not None and not any(where == m or where.startswith(f"{m}.") for m in missing):
            missing.append(where)
    if missing:
        named = ", ".join(repr(path) for path in missing[:3]) + (" and more" if len(missing) > 3 else "")
        raise ValueError(f"{label} lacks {named}; write it again with `{made_by}`")


def require_what_the_site_reads(reports: Reports) -> None:
    """Refuse reports that lack anything the site reads of them (``REQUIRED``): the summary, the
    shipped run's and its twin's reports in full, every run report what its split needs, the
    leakage and standard-recipe reports. A ValueError names the report, where it lacks a key and
    the command that writes it again."""
    folder = reports.files[-2].rsplit("/", 1)[0]  # where leakage.json is, beside runs/
    _require(reports.summary, "summary", reports.files[0])
    chosen = (reports.summary["shipped"]["run"], reports.summary["twin"]["run"])
    for name, report in reports.runs.items():
        label = f"{folder}/runs/{name}.json"
        if name in chosen:
            _require(report, "run", label)
        _require(report, "splits", label)
        if report["protocol"] in PAIRED_PROTOCOLS:
            _require(report, "paired", label)
    _require(reports.leakage, "leakage", reports.files[-2])
    _require(reports.legacy, "legacy", reports.files[-1])


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
    """The summary, the run report of every run it lists (``<reports_dir>/runs/<run>.json``),
    and ``leakage.json`` and ``legacy.json`` from ``reports_dir``, each checked for every key
    the site data reads (``require_what_the_site_reads``)."""
    summary_path, reports_dir = Path(summary_path), Path(reports_dir)
    folder = reports_dir.name
    summary_label = f"{summary_path.parent.name}/{summary_path.name}"
    summary = _read_report(summary_path, "sightline report")
    _require(summary, "summary", summary_label)  # it names the run reports to read
    names = list(dict.fromkeys((summary["shipped"]["run"], summary["twin"]["run"], *summary["runs"])))
    runs = {name: _read_report(reports_dir / "runs" / f"{name}.json", "sightline eval --all") for name in names}
    leakage = _read_report(reports_dir / "leakage.json", "sightline leakage")
    legacy = _read_report(reports_dir / "legacy.json", "sightline legacy")
    files = [summary_label, *(f"{folder}/runs/{name}.json" for name in names),
             f"{folder}/leakage.json", f"{folder}/legacy.json"]
    reports = Reports(summary=summary, runs=runs, leakage=leakage, legacy=legacy, files=files)
    require_what_the_site_reads(reports)
    return reports


def _mean_and_sd(value):
    """A statistic over seeds (``{"mean", "sd", "values"}``) without its values; anything else as is."""
    if isinstance(value, dict) and set(value) == {"mean", "sd", "values"}:
        return {"mean": value["mean"], "sd": value["sd"]}
    return value


def _rounded(value: float, decimals: int) -> float:
    """``value`` rounded to ``decimals`` places, a negative zero written as 0.0."""
    return round(value, decimals) + 0.0


def drive_profile(labels: DriveLabels, bucket_frames: int = BUCKET_FRAMES) -> dict:
    """A drive's steering labels averaged over buckets of consecutive frames, for the page's
    drive chart. ``labels`` are the drive's dropout-repaired labels (``repair_dropouts``), so
    the figures are the ones every score is measured against; they are never read from the cache.

    Bucket ``k`` holds frames ``k * bucket_frames`` up to ``(k + 1) * bucket_frames``, so every
    frame is in exactly one bucket and the last one holds the frames left over (fewer than
    ``bucket_frames`` unless they divide the drive). Returns, with one entry per bucket in the
    lists: ``start_frame``; ``mean_deg``, the mean steering angle, and ``mean_abs_deg``, the mean
    absolute angle, both rounded to ``PROFILE_ANGLE_DECIMALS`` places; ``start_s``, the time of
    the bucket's first frame in seconds since the drive's first frame, rounded to the millisecond;
    and the drive's ``frames`` and ``duration_s`` (first frame to last, rounded the same way).
    A drive without timestamps (the 2017 drive) has ``None`` for ``start_s`` and ``duration_s``;
    one whose timestamps run backwards is refused, as it has no time axis.

    The same labels always give the same values: each sum is exactly rounded (``math.fsum``),
    so a mean does not depend on the order numpy would add in, and a negative zero is written 0.0.
    """
    if bucket_frames < 1:
        raise ValueError(f"a bucket needs at least 1 frame, got {bucket_frames}")
    angles = labels.angles_deg.tolist()
    if not angles:
        raise ValueError("a drive without frames has no profile")
    starts = range(0, len(angles), bucket_frames)
    mean_deg, mean_abs_deg = [], []
    for start in starts:
        bucket = angles[start:start + bucket_frames]
        mean_deg.append(_rounded(math.fsum(bucket) / len(bucket), PROFILE_ANGLE_DECIMALS))
        mean_abs_deg.append(_rounded(math.fsum(map(abs, bucket)) / len(bucket), PROFILE_ANGLE_DECIMALS))
    start_s = duration_s = None
    if labels.timestamps_s is not None:
        stamps = labels.timestamps_s
        backwards = np.flatnonzero(np.diff(stamps) < 0)
        if backwards.size:
            raise ValueError(f"drive {labels.root.name!r}: frame {int(backwards[0]) + 1} is timed before "
                             f"frame {int(backwards[0])}, so the drive has no time axis")
        start_s = [_rounded(float(stamps[s] - stamps[0]), PROFILE_TIME_DECIMALS) for s in starts]
        duration_s = _rounded(float(stamps[-1] - stamps[0]), PROFILE_TIME_DECIMALS)
    return {
        "duration_s": duration_s,
        "frames": len(angles),
        "mean_abs_deg": mean_abs_deg,
        "mean_deg": mean_deg,
        "start_frame": list(starts),
        "start_s": start_s,
    }


def _whole_ms(labels: DriveLabels) -> np.ndarray:
    """A timed drive's timestamps as whole milliseconds since its first frame (int64). The label files
    record milliseconds and the loader adds them to epoch seconds, whose floating point noise (well
    under a microsecond) the rounding takes away, so these are the times recorded, exactly. A drive
    whose timestamps run backwards is refused."""
    stamps = np.asarray(labels.timestamps_s, dtype=np.float64)
    ms = np.rint((stamps - stamps[0]) * 1000.0).astype(np.int64)
    backwards = np.flatnonzero(np.diff(ms) < 0)
    if backwards.size:
        raise ValueError(f"drive {labels.root.name!r}: frame {int(backwards[0]) + 1} is timed before frame "
                         f"{int(backwards[0])}, so its frames have no intervals in time")
    return ms


def _ms_to_s(ms) -> float:
    """``ms`` milliseconds as seconds, rounded to the millisecond with an exact half going to the even
    one (Python's ``round``): a whole number as it is, a mean as an exact Fraction, and a float that
    numpy interpolated snapped to a millionth of a millisecond first, as floating point can leave a
    value that is exactly a half a hair to one side of it. The result is a whole number of
    milliseconds over a thousand, so never a negative zero."""
    if isinstance(ms, (float, np.floating)):
        ms = round(float(ms), 6)
    return round(ms) / 1000


def _spread(ms: np.ndarray) -> dict:
    """The least, the largest and the mean of ``ms`` (whole milliseconds) and its ``SPREAD_PERCENTILES``
    (numpy, ``PERCENTILE_METHOD``), in seconds to the millisecond."""
    keys, levels = zip(*SPREAD_PERCENTILES.items())
    values = np.percentile(ms, levels, method=PERCENTILE_METHOD)
    return {"min": _ms_to_s(ms.min()), "max": _ms_to_s(ms.max()),
            "mean": _ms_to_s(Fraction(int(ms.sum()), int(ms.size))),
            **{key: _ms_to_s(value) for key, value in zip(keys, values)}}


def _closest_in_time(ms: np.ndarray, held: np.ndarray, trained: np.ndarray) -> dict:
    """The pair of an evaluation frame of ``held`` and a training frame of ``trained`` closest in time
    by ``ms`` (whole milliseconds that never decrease): ``eval_frame``, ``train_frame`` and ``gap_s``,
    the time between them. Ties go to the pair fewest frames apart, then to the earliest evaluation
    frame, then to the earliest training frame. The times never decrease, so the training frame
    closest to a frame is the nearest training frame before it or after it, any other being as far
    in time at least and further in frames: only those two are compared."""
    held = np.asarray(held, dtype=np.int64)
    trained = np.sort(np.asarray(trained, dtype=np.int64))
    pos = np.searchsorted(trained, held)
    eval_frames = np.concatenate([held, held])
    train_frames = np.concatenate([trained[np.maximum(pos - 1, 0)], trained[np.minimum(pos, trained.size - 1)]])
    apart_ms = np.abs(ms[eval_frames] - ms[train_frames])
    apart_frames = np.abs(eval_frames - train_frames)
    best = np.lexsort((train_frames, eval_frames, apart_frames, apart_ms))[0]
    return {"eval_frame": int(eval_frames[best]), "gap_s": _ms_to_s(apart_ms[best]),
            "train_frame": int(train_frames[best])}


def drive_timing(labels: DriveLabels, lags, splits: dict | None = None) -> dict | None:
    """How far apart in time a drive's frames are, from its timestamps: None for a drive without them.

    ``frame_interval_s`` is the spread of the times from each frame to the next, and ``lag_gap_s``,
    for each of ``lags`` in turn, ``lag`` and the spread of the times from frame ``i - lag`` to frame
    ``i`` over every frame ``i`` from ``lag`` on. A spread is the ``min``, the ``max``, the ``mean``
    and the ``SPREAD_PERCENTILES`` (``p05``, ``median``, ``p95``). ``closest_eval_s`` holds, for each
    split of ``splits`` (a protocol's ``SplitIndices`` of this drive, by protocol) and for its ``val``
    and ``test`` frames in turn, the evaluation frame and the training frame closest in time
    (``_closest_in_time``); it is None without ``splits``, for a drive no run trained on.

    The times are worked out in whole milliseconds, the resolution of the timestamps (``_whole_ms``),
    so the floating point noise of epoch seconds never reaches them and every step is exact but the
    last: intervals, gaps, the least and largest are whole numbers, the mean an exact fraction, and
    the percentiles numpy's, with ``PERCENTILE_METHOD`` named. Each value is then rounded once, to the
    millisecond (``_ms_to_s``), so the same timestamps give the same values on any machine.

    Raises a ValueError for timestamps that run backwards, for a lag that is not a whole number of
    frames, 1 or more, and for a drive too short to time an interval or a lag.
    """
    lags = list(lags)
    for lag in lags:
        if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 1:
            raise ValueError(f"a frame lag is a whole number of frames, 1 or more, not {lag!r}")
    if labels.timestamps_s is None:
        return None
    ms = _whole_ms(labels)
    longest = max([1, *lags])
    if ms.size <= longest:
        raise ValueError(f"drive {labels.root.name!r} has too few timed frames ({ms.size}) to time frames "
                         f"{longest} apart")
    return {
        "closest_eval_s": None if splits is None else {
            protocol: {which: _closest_in_time(ms, getattr(split, which), split.train) for which in ("val", "test")}
            for protocol, split in splits.items()},
        "frame_interval_s": _spread(np.diff(ms)),
        "lag_gap_s": [{"lag": int(lag), **_spread(ms[lag:] - ms[:-lag])} for lag in lags],
    }


def frame_timing(drives: dict, lags, setting: dict) -> dict:
    """results.json's ``timing``: ``drive_timing`` of each drive of ``drives`` (a drive's name to its
    dropout-repaired labels, as ``drive_profile`` reads them) with ``lags``, the shipped run's frame
    lags; for the drive the runs trained on (``setting["drive"]``), with its ``TIMED_PROTOCOLS``
    splits as the splits module makes them of its frames with the firewall and block of ``setting``,
    the ones every run recorded."""
    timing = {}
    for name, labels in sorted(drives.items()):
        splits = None
        if name == setting["drive"]:
            splits = {protocol: make_split(protocol, len(labels), firewall=setting["firewall"],
                                           block=setting["block"])
                      for protocol in TIMED_PROTOCOLS}
        timing[name] = drive_timing(labels, lags, splits)
    return timing


def split_rules(block: int, firewall: int) -> dict:
    """How each protocol cuts a drive, by protocol, each with its public name as ``label``. The
    blocked split cuts the drive into ``block``-frame blocks laid out by the repeating
    ``pattern`` (T train, V validation, E test) and drops ``firewall`` frames on each side of
    every boundary between a training block and a validation or test block (two training
    blocks in a row are not cut apart). The forward split trains on the first
    ``train_frac`` of the drive, validates on the next ``val_frac`` and tests on the rest
    (``test_frac``), dropping ``firewall`` frames on each side of both boundaries. The
    shuffled split shuffles the frames, then slices them by the same three fractions.

    ``block`` and ``firewall`` are the settings the runs were made with (their reports record
    them, and every run's must agree with the shipped run's, the leakage report's and the
    standard-recipe report's); the pattern and the fractions are the splits module's, which no
    command changes. ``test_frac`` is rounded to ten places, so that the floating
    point noise in ``1 - train_frac - val_frac`` does not reach the page.
    """
    shares = {"train_frac": TRAIN_FRAC, "val_frac": VAL_FRAC, "test_frac": round(1.0 - TRAIN_FRAC - VAL_FRAC, 10)}
    return {
        "blocked": {"label": PUBLIC_NAMES["blocked"], "block": block, "firewall": firewall,
                    "pattern": BLOCK_PATTERN},
        "forward": {"label": PUBLIC_NAMES["forward"], "firewall": firewall, **shares},
        "leaky": {"label": PUBLIC_NAMES["leaky"], **shares},
    }


def _frames_of(report: dict) -> int:
    """The frames of the drive a run trained on, as its report describes the drive."""
    info = report["drives"].get(report["drive"])
    if not isinstance(info, dict) or "frames" not in info:
        raise ValueError(f"run {report['run']!r} does not describe its training drive {report['drive']!r}; "
                         "evaluate it again with `sightline eval --all`")
    return info["frames"]


def _agreed(by_run: dict, what: str, show):
    """The one value every run of ``by_run`` (run name to value) has. A ValueError, naming each
    value (as ``show`` writes it) with its runs, the rarest value first, when they differ."""
    runs_by_value: dict = {}
    for run, value in by_run.items():
        runs_by_value.setdefault(value, []).append(run)
    if len(runs_by_value) > 1:
        parts = sorted(((show(value), runs) for value, runs in runs_by_value.items()),
                       key=lambda part: (len(part[1]), part[0]))
        raise ValueError(f"the run reports disagree on {what}: " + "; ".join(
            f"{shown} in {', '.join(runs[:3])}" + (f" and {len(runs) - 3} more" if len(runs) > 3 else "")
            for shown, runs in parts))
    return next(iter(by_run.values()))


def split_sizes(reports: Reports) -> dict:
    """The frames each protocol trains, validates and tests on, by protocol, as every run of
    the protocol recorded them: ``train``, ``val`` and ``test`` counts, the training ``drive``
    and its ``frames``, ``dropped`` (the frames of the drive that are in none of the three: the
    firewalls' share), the ``seeds`` and the number of ``runs`` that recorded them, and the
    protocol's public name as ``label``. A ValueError when the runs of a protocol recorded
    different counts, drives or frame counts; a protocol none of the runs follows is left out."""
    by_protocol: dict = {}
    for report in reports.runs.values():
        by_protocol.setdefault(report["protocol"], []).append(report)
    unknown = sorted(set(by_protocol) - set(PROTOCOLS))
    if unknown:
        raise ValueError(f"run {by_protocol[unknown[0]][0]['run']!r} follows the protocol {unknown[0]!r}, "
                         f"which is not one of {', '.join(PROTOCOLS)}")
    sizes = {}
    for protocol in PROTOCOLS:
        members = by_protocol.get(protocol)
        if not members:
            continue
        label = PUBLIC_NAMES[protocol]
        counts = {r["run"]: (r["drive"], _frames_of(r), r["splits"]["train_n"], r["splits"]["val_n"],
                             r["splits"]["test_n"]) for r in members}
        drive, frames, train, val, test = _agreed(
            counts, f"the {label} split's train, validation and test frames",
            lambda c: f"{c[2]}/{c[3]}/{c[4]} of the {c[1]} frames of {c[0]}")
        if train + val + test > frames:
            raise ValueError(f"the {label} split's {train}/{val}/{test} train/validation/test frames do not fit "
                             f"in the {frames} frames of {drive}")
        sizes[protocol] = {
            "drive": drive,
            "dropped": frames - train - val - test,
            "frames": frames,
            "label": label,
            "runs": len(members),
            "seeds": sorted({r["seed"] for r in members}),
            "test": test,
            "train": train,
            "val": val,
        }
    return sizes


def _check_labels(name: str, labels: DriveLabels, recorded: dict) -> None:
    """Refuse labels of drive ``name`` that are not the ones the reports were scored on: another
    number of frames, of repaired dropouts or another label file than ``recorded`` (the
    summary's record of the drive) gives."""
    if len(labels) != recorded.get("frames"):
        raise ValueError(f"drive {name!r} has {len(labels)} labelled frames, but summary.json "
                         f"describes {recorded.get('frames')}")
    repaired = int(labels.repaired_indices.size)
    if repaired != recorded.get("repaired_labels"):
        raise ValueError(f"drive {name!r} has {repaired} repaired dropouts, but summary.json records "
                         f"{recorded.get('repaired_labels')}; the profile is made from the labels as "
                         "repair_dropouts leaves them")
    label_file = labels.root / "data.txt"
    digest = label_sha256(label_file) if label_file.is_file() else None
    if digest != recorded.get("label_sha256"):
        raise ValueError(f"drive {name!r}: its label file {label_file} (sha256 {str(digest)[:12]}) is not the "
                         f"one the reports were made from ({str(recorded.get('label_sha256'))[:12]})")


def _check_inputs(reports: Reports, drives: dict) -> dict:
    """The setting every report agrees on (``_check_reports``), once the labels in ``drives`` are
    those the reports were scored on (``_check_labels``)."""
    setting = _check_reports(reports)
    for name, labels in sorted(drives.items()):
        _check_labels(name, labels, reports.summary["drives"].get(name) or {})
    return setting


def export_parity(reports: Reports, export: dict) -> dict:
    """results.json's ``parity``: for each of the page's models, how closely onnxruntime reproduced
    PyTorch when the model was exported, as ``export`` (what export.json holds) records it: the run,
    the largest angle and saliency differences over the frame stacks checked (``stacks``), the gates
    they met and the onnxruntime version. ``export`` must describe the shipped run and its twin, each
    with the checkpoint its run report describes."""
    _require(export, "export", "export.json")
    parity = {}
    for key, role in (("sightline", "shipped"), ("shuffled", "twin")):
        model, run = export["models"][key], reports.summary[role]["run"]
        if (model["run"], model["checkpoint_sha256"]) != (run, reports.runs[run]["checkpoint_sha256"]):
            raise ValueError(f"export.json's {key} model is run {model['run']!r} (checkpoint "
                             f"{str(model['checkpoint_sha256'])[:12]}), not the {role} run {run!r} as its report "
                             "describes it; write the site data again with `sightline site-data`")
        checked = model["parity"]
        parity[key] = {"run": run, "stacks": checked["frames"], "max_abs_angle_deg": checked["max_angle_diff_deg"],
                       "max_abs_saliency": checked["max_saliency_diff"],
                       "angle_tolerance_deg": checked["angle_tolerance_deg"],
                       "saliency_tolerance": checked["saliency_tolerance"],
                       "onnxruntime": export["versions"]["onnxruntime"]}
    return parity


def generated_from(reports: Reports) -> list:
    """results.json's ``generated_from``: the reports' files in the order they were read
    (``load_reports``), then the export record the build writes beside them (export.json), whose
    ``parity`` results.json copies."""
    folder = reports.files[-1].rsplit("/", 1)[0]  # where legacy.json is, beside export.json
    return [*reports.files, f"{folder}/export.json"]


def site_results(reports: Reports, drives: dict, export: dict) -> dict:
    """results.json: what the page shows, copied from the reports value for value, with the
    drive profiles, split rules, split sizes and frame timing added (see the module docstring), and
    the export parity copied from ``export``, what export.json holds: the build passes the record it
    is about to write, never the file a previous build left.

    The reports must agree with one another as ``build_site_data`` requires (``_check_reports``),
    so the split rules, the split sizes and the splits the timing measures come from the one
    firewall and block they share. ``drives`` maps each drive the page describes to its
    dropout-repaired labels (``repair_dropouts``); each must have as many frames and repaired
    dropouts as the summary records for it, and come from the label file it records (sha256).
    """
    setting = _check_inputs(reports, drives)
    summary, leakage = reports.summary, reports.leakage
    shipped = reports.runs[summary["shipped"]["run"]]
    profiles = {name: drive_profile(labels) for name, labels in sorted(drives.items())}
    results = {
        "assets": ASSETS,
        "baselines": {
            "run": shipped["run"],
            "test": {key: shipped["test"][key] for key in ("n", "model", "baselines")},
            "cross_drive": {key: shipped["cross_drive"][key] for key in ("drive", "n", "model", "baselines")},
        },
        "cutoffs": summary["cutoffs"],
        "drive_profile": {"bucket_frames": BUCKET_FRAMES, "drives": profiles},
        "drives": summary["drives"],
        "generated_from": generated_from(reports),
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
        "parity": export_parity(reports, export),
        "protocol_names": PUBLIC_NAMES,
        "regimes": summary["regimes"],
        "shipped": summary["shipped"],
        "split_rules": split_rules(setting["block"], setting["firewall"]),
        "split_sizes": split_sizes(reports),
        "timing": frame_timing(drives, shipped["frame_lags"], setting),
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


def _paired_problem(recorded: dict, frames: int, seed: int, setting: dict) -> str | None:
    """How ``recorded`` (a report's ``paired``: ``n`` and ``frames_sha256``) differs from the
    paired frames the splits module makes of ``frames`` frames for ``seed`` with the shared
    settings, or None when it does not."""
    paired = paired_frames(make_split("blocked", frames, firewall=setting["firewall"], block=setting["block"]),
                           make_split("leaky", frames, seed=seed))
    found = (recorded["n"], recorded["frames_sha256"])
    expected = (int(paired.size), frames_sha256(paired))
    if found == expected:
        return None
    return (f"{found[0]} paired frames (sha256 {str(found[1])[:12]}), but the splits module pairs {expected[0]} "
            f"(sha256 {expected[1][:12]}) of the same {frames} frames for seed {seed} with the same settings")


def _check_runs(reports: Reports, setting: dict) -> None:
    """Refuse a run report made on another drive or with other split settings than the shipped
    run's, one that describes the drive with another frame count than the summary does, one
    whose split sizes are not the ones the splits module makes from its drive's frames, its
    protocol, its seed and the settings it records, and a blocked or shuffled run, or the
    leakage report, whose paired frames are not the ones the splits module pairs: equal sizes
    do not show that two splits test on the same frames, the test frames they pair do. No
    validation frame is paired, so a split that moves validation frames alone passes here; the
    build compares the shipped run's and the twin's saved splits as well (``_check_saved_splits``)."""
    drive = setting["drive"]
    shipped_with = {"firewall": setting["firewall"], "block": setting["block"]}
    recorded = (reports.summary["drives"].get(drive) or {}).get("frames")
    for name, report in reports.runs.items():
        if report["drive"] != drive:
            raise ValueError(f"run {name!r} trained on drive {report['drive']!r}, the shipped run on {drive!r}")
        splits = report["splits"]
        made_with = {"firewall": splits["firewall"], "block": splits["block"]}
        if made_with != shipped_with:
            raise ValueError(f"run {name!r} made its splits with {made_with}, but the shipped run with "
                             f"{shipped_with}")
        frames = _frames_of(report)
        if frames != recorded:
            raise ValueError(f"run {name!r} describes drive {drive!r} with {frames} frames, but summary.json "
                             f"with {recorded}")
        protocol, label = report["protocol"], PUBLIC_NAMES.get(report["protocol"], report["protocol"])
        try:
            made = make_split(protocol, frames, seed=report["seed"], firewall=splits["firewall"],
                              block=splits["block"])
        except ValueError as exc:
            raise ValueError(f"run {name!r}: the splits module cannot make its {label} split of "
                             f"{frames} frames: {exc}") from None
        found = (splits["train_n"], splits["val_n"], splits["test_n"])
        expected = (made.train.size, made.val.size, made.test.size)
        if found != expected:
            raise ValueError(f"run {name!r} recorded {'/'.join(map(str, found))} train/validation/test frames "
                             f"for its {label} split, but the splits module makes "
                             f"{'/'.join(map(str, expected))} of the same {frames} frames with the same "
                             "settings; the split rules are not the ones it was trained with")
        if protocol in PAIRED_PROTOCOLS:
            problem = _paired_problem(report["paired"], frames, report["seed"], setting)
            if problem is not None:
                raise ValueError(f"run {name!r} recorded {problem}; the split rules are not the ones it was "
                                 "trained with")
    problem = _paired_problem(reports.leakage["paired"], recorded, reports.leakage["seed"], setting)
    if problem is not None:
        raise ValueError(f"leakage.json analyses {problem}; analyse the drive again with `sightline leakage`")


def _check_reports(reports: Reports) -> dict:
    """The shipped run's drive, other drive, firewall and block, once every report agrees
    on them; a ValueError names the report that does not."""
    summary = reports.summary
    shipped_run, twin_run = summary["shipped"]["run"], summary["twin"]["run"]
    shipped, twin = reports.runs[shipped_run], reports.runs[twin_run]
    for name, report in reports.runs.items():
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
    _check_runs(reports, {"drive": drive, **splits})
    _check_summary_against_runs(reports)
    return {"drive": drive, "cross_drive": cross_drive, **splits}


def _check_saved_splits(workspace: Workspace, reports: Reports) -> None:
    """Refuse a shipped run or twin whose saved split, the training, validation and test frames
    its run folder holds (``<split>_idx.npy``), is not the split the splits module makes of its
    drive's frames with its protocol, its seed and the settings its report records. The reports
    alone show the split sizes and the test frames the paired measurement uses; a split that
    moves validation frames alone keeps both, and only the saved frames show it. The run folders
    are not part of a clone, so this runs where the site data is built, never in
    ``site_results``."""
    for role in ("shipped", "twin"):
        name = reports.summary[role]["run"]
        report = reports.runs[name]
        frames, splits = _frames_of(report), report["splits"]
        label = PUBLIC_NAMES.get(report["protocol"], report["protocol"])
        made = make_split(report["protocol"], frames, seed=report["seed"], firewall=splits["firewall"],
                          block=splits["block"])
        run_dir = workspace.run_dir(name)
        for split in ("train", "val", "test"):
            if not np.array_equal(load_split_indices(run_dir, split), getattr(made, split)):
                raise ValueError(
                    f"run {name!r}: its saved {split} frames ({split}_idx.npy) are not the {label} split the "
                    f"splits module makes of the {frames} frames of {report['drive']} with seed {report['seed']}, "
                    f"firewall {splits['firewall']} and block {splits['block']}; the split rules are not the ones "
                    "it was trained with")


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


def clip_times(labels: DriveLabels, first_frame: int, n: int) -> dict:
    """``first_s`` and ``last_s``: the times of a clip's first and last scored frames (``first_frame``
    and the ``n - 1`` after it) in seconds since the drive's first frame, rounded to the millisecond
    like the drive profile's times; both None for a drive without timestamps."""
    if not (n >= 1 and 0 <= first_frame and first_frame + n <= len(labels)):
        raise ValueError(f"a clip of {n} frames from frame {first_frame} does not fit in the {len(labels)} "
                         f"frames of drive {labels.root.name!r}")
    if labels.timestamps_s is None:
        return {"first_s": None, "last_s": None}
    stamps = labels.timestamps_s
    return {"first_s": _rounded(float(stamps[first_frame] - stamps[0]), PROFILE_TIME_DECIMALS),
            "last_s": _rounded(float(stamps[first_frame + n - 1] - stamps[0]), PROFILE_TIME_DECIMALS)}


def leakage_pairs(leakage: dict) -> dict:
    """pairs.json: the leakage report's example frame pairs as the page shows them, each frame
    named by the image file the build writes for it, the gaps in seconds rounded to the
    millisecond (timestamps are epoch seconds, so their differences carry float noise)."""
    pairs = []
    for pair in leakage["pairs"]:
        entry = {"frame": pair["frame"], "image": f"frame-{pair['frame']}.jpg", "angle_deg": pair["angle_deg"]}
        for protocol in LOOKUP_PROTOCOLS:
            side = pair[protocol]
            entry[protocol] = {
                "nn_frame": side["nn_frame"],
                "image": f"frame-{side['nn_frame']}.jpg",
                "gap_frames": side["gap_frames"],
                "gap_s": None if side["gap_s"] is None else round(side["gap_s"], 3),
                "pixel_rmse": side["pixel_rmse"],
                "nn_angle_deg": side["nn_angle_deg"],
            }
        pairs.append(entry)
    return {"drive": leakage["drive"], "seed": leakage["seed"], "median_frame_interval_s": leakage["dt_s"],
            "pairs": pairs}


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
    """Put a staged build in place, every move a rename within one folder's volume: the previous
    build's files are moved into a folder beside ``out_dir`` and the previous export.json to a
    hidden name beside itself, then the files under ``new_root`` (a folder beside ``out_dir``)
    into ``out_dir`` and ``new_export`` (a file beside ``export_path``) to ``export_path``. If
    anything fails, every new file and folder is removed and the old files are moved back, so
    ``out_dir`` and ``export_path`` are as they were; the old files are removed either way,
    unless moving them back failed, and then the error says where they are."""
    old = sorted({path for folder, patterns in GENERATED.items() for pattern in patterns
                  for path in (out_dir / folder).glob(pattern) if path.is_file()}
                 | {out_dir / name for name in FIXED_FILES if (out_dir / name).is_file()})
    aside = Path(tempfile.mkdtemp(dir=out_dir.parent, prefix=f".{out_dir.name}-previous-"))
    old_export = export_path.with_name(f".{export_path.name}.{os.getpid()}-{secrets.token_hex(4)}.previous")
    moved, placed, created = [], [], []
    try:
        for k, path in enumerate(old):
            os.replace(path, aside / str(k))
            moved.append((aside / str(k), path))
        if export_path.is_file():
            os.replace(export_path, old_export)
            moved.append((old_export, export_path))
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
            where = f"{aside}" + (f" and {old_export}" if old_export.exists() else "")
            raise RuntimeError(f"the new site data could not be put in place ({failure}), and the previous "
                               f"files could not all be put back ({exc}); they are in {where}") from failure
        shutil.rmtree(aside, ignore_errors=True)
        raise
    shutil.rmtree(aside, ignore_errors=True)
    with contextlib.suppress(OSError):
        old_export.unlink(missing_ok=True)


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
    lacks a key the site data reads (``REQUIRED``); reports that disagree on the runs, the drive
    or the split settings, a run report whose split sizes or paired frames are not the ones the
    splits module makes from its settings, or a summary whose figures for the shipped run or its
    twin differ from their run reports; a run whose best.pt is not the checkpoint its report
    describes, or whose saved training, validation or test frames are not the split the splits
    module makes from its report's settings (``_check_saved_splits``, the shipped run and its
    twin); a cache that is missing or not built from the drive's labels as they are now; a
    drive whose labels are not the ones the reports were made from; a 2018 clip closer than
    ``2 * firewall + 1`` frames to a frame the shipped run trained on. A model whose parity or size
    fails its gate is refused after the build was staged, and nothing of it is kept.
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
    _check_saved_splits(workspace, reports)
    lags = tuple(int(lag) for lag in models[0].model.frame_lags)
    lead_in = max(lags, default=0)
    drives = {"c2018": setting["drive"], "c2017": setting["cross_drive"]}
    opened = {drive: _open_drive(workspace, drive, summary) for drive in dict.fromkeys(drives.values())}
    # the labels of each drive are the repaired ones its cache was just checked against
    labels_by_drive = {drive: labels for drive, (labels, _) in opened.items()}
    _check_inputs(reports, labels_by_drive)

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
                **clip_times(labels, first, clip_frames),
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
        write_report(folder / "pairs.json", leakage_pairs(leakage))

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
        # results.json carries this build's own export record, the one export.json is about to get
        write_report(staging / "data" / "results.json", site_results(reports, labels_by_drive, export))
        # staged beside its final path, so putting it in place is a rename on the reports' own volume
        staged_export = reports_dir / f".export.json.{os.getpid()}-{secrets.token_hex(4)}.new"
        try:
            write_report(staged_export, export)
            _swap_in(staging, out_dir, staged_export, reports_dir / "export.json")
        finally:
            with contextlib.suppress(OSError):
                staged_export.unlink(missing_ok=True)
    return export
