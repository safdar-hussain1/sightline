"""The browser's data: models, demo clips, results and self-test fixtures, from the reports.

``build_site_data`` reads the summary (``sightline report``), the run reports and the
checkpoints of the shipped run and of its twin (the same model trained on the shuffled
split), the leakage and standard-recipe reports and the two drives, and writes everything
the page loads under ``out_dir`` (``web/public``). Paths inside results.json are relative
to ``out_dir``; paths inside the other JSON files are relative to their own folder.

``models/sightline.onnx``, ``models/sightline-shuffled.onnx``
    The shipped run and its twin, each one graph from uint8 frame stacks to steering
    degrees and saliency, at most 2 MB, checked against PyTorch with onnxruntime on every
    clip input and every self-test input.
``data/results.json``
    The numbers the page shows, copied value for value from the reports, never computed
    again: the generalisation ladder, the paired leakage measurement, each group's mean and
    sd over seeds, the shipped run's baselines and regimes, the standard-recipe PilotNet's
    scores, the leakage analysis, the drives, the training recipe and the library versions. The
    leakage report's ``median_s`` (a median frame count times the median frame interval,
    not a time anything took) is left out; the interval itself is ``median_frame_interval_s``.
``data/clips/<id>/``
    A demo clip of ``clip_frames`` scored frames: ``c2018`` from the blocked test stretches
    of the drive the shipped run trained on (minutes it never trained on), ``c2017`` from
    the other drive (another day). ``sprite-NN.jpg`` holds the model's 200x66 input frames,
    60 to a sheet stacked top to bottom (JPEG q95, no chroma subsampling). Sprite frame
    ``k`` is drive frame ``first_frame - lead_in + k``: the ``lead_in`` frames before the
    first scored frame give it its history, so no stacked frame is ever clamped. The
    sprites are the clip's model input: ``expected`` holds both models' CPU predictions on
    the frames exactly as Pillow decodes the written JPEGs, and ``decoded_sha256`` the
    sha256 of each decoded frame's RGB bytes, so a browser can check that it decodes the
    same pixels. ``clip.mp4`` shows the scored camera frames (video frame ``k`` is scored
    frame ``k``) as H.264 at 17.5 fps; 4:2:0 video needs an even width, so each 455-pixel
    camera frame is followed by one repeated column (``video_size`` against
    ``camera_size``). ``clip_mae_deg`` is the mean absolute error on this clip alone, for
    each model and for steering straight ahead.
``data/selftest/``
    ``frames.bin``: eight stacked inputs from the 2018 clip as raw uint8 NHWC bytes, taken
    from the drive cache (no JPEG involved), and ``expected.json``: both models' angles and
    saliency maps on them, with the tolerance a client must meet.
``data/leakage/``
    ``pairs.json``, the leakage report's example frame pairs with their gaps in seconds
    rounded to the millisecond, and one camera frame (JPEG q85) per frame they name, read
    through the drive's label rows.

``export.json`` in ``reports_dir`` records each model's size, digest, measured parity and
checkpoint, and how each clip was chosen and how each model does on it.

A clip is chosen from the labels alone, never from a model's predictions: among the
windows of ``clip_frames`` frames that fit, with the ``lead_in`` frames before them, inside
one stretch (a stretch of the test frames the shipped run saved, or the whole of the other
drive), the window whose labels have the largest standard deviation. Windows start every
10 frames from the first that fits in each stretch, and ties go to the earliest. The 2018
clip's distance from the frames the shipped run trained on is then measured; under
2·firewall + 1 frames it is refused. A clip is picked for being worth watching, so its MAE
describes the clip, not the model.

Everything is computed and written to a staging folder beside ``out_dir`` first; the files
then replace the previous build's together, so a failed build changes nothing. Apart from
the videos, the same reports, checkpoints and drives give the same bytes.
"""

from __future__ import annotations

import base64
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
from sightline.export.onnx_export import ANGLE_TOLERANCE_DEG, SALIENCY_TOLERANCE, export_onnx, open_drive_cache
from sightline.leakage import LOOKUP_PROTOCOLS, frame_gaps
from sightline.paths import MissingDataError, UnusableRunError, Workspace
from sightline.reporting import checkpoint_sha256, load_run_model, load_split_indices, write_report
from sightline.saliency import SteeringWithSaliency
from sightline.train import code_version

__all__ = [
    "ASSETS",
    "CAMERA_SIZE",
    "CLIP_FPS",
    "CLIP_FRAMES",
    "CLIP_STEP",
    "CLIP_TITLES",
    "FRAMES_PER_SHEET",
    "MAX_MODEL_BYTES",
    "MODEL_FILES",
    "Reports",
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
]

#: scored frames per demo clip: about 24 s of driving
CLIP_FRAMES = 420
#: candidate clip windows start every CLIP_STEP frames
CLIP_STEP = 10
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
#: stacked inputs in the self-test fixture
SELFTEST_INPUTS = 8
#: the page's two models, by the names it uses, and their files under ``out_dir``
MODEL_FILES = {"sightline": "models/sightline.onnx", "shuffled": "models/sightline-shuffled.onnx"}
VIDEO_OPTIONS = ("-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "slow",
                 "-movflags", "+faststart")
#: what results.json keeps of the standard-recipe report: its numbers and names; the page
#: describes the recipe in its own words
LEGACY_KEYS = ("trained_on", "seen_first_frame", "splits", "port_parity", "eval", "atan_cap")
#: every file the page loads besides results.json, relative to ``out_dir``
ASSETS = {
    "models": dict(MODEL_FILES),
    "clips": {cid: f"data/clips/{cid}/clip.json" for cid in CLIP_TITLES},
    "leakage_pairs": "data/leakage/pairs.json",
    "selftest": {"frames": "data/selftest/frames.bin", "expected": "data/selftest/expected.json"},
}
#: the files a build writes into each folder; a new build replaces all of them
GENERATED = {
    **{f"data/clips/{cid}": ("sprite-*.jpg", "clip.json", "clip.mp4") for cid in CLIP_TITLES},
    "data/leakage": ("frame-*.jpg", "pairs.json"),
    "data/selftest": ("frames.bin", "expected.json"),
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
    ``leakage.json`` and ``legacy.json`` from ``reports_dir``."""
    summary_path, reports_dir = Path(summary_path), Path(reports_dir)
    summary = _read_report(summary_path, "sightline report")
    for key in ("shipped", "twin"):
        if not isinstance(summary.get(key), dict) or "run" not in summary[key]:
            raise ValueError(f"{summary_path} names no {key} run; write it again with `sightline report`")
    names = list(dict.fromkeys((summary["shipped"]["run"], summary["twin"]["run"])))
    runs = {name: _read_report(reports_dir / "runs" / f"{name}.json", "sightline eval --all") for name in names}
    leakage = _read_report(reports_dir / "leakage.json", "sightline leakage")
    legacy = _read_report(reports_dir / "legacy.json", "sightline legacy")
    folder = reports_dir.name
    files = [f"{summary_path.parent.name}/{summary_path.name}", *(f"{folder}/runs/{name}.json" for name in names),
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


def _check_reports(reports: Reports) -> dict:
    """The shipped run's drive, other drive, firewall and block, once every report agrees
    on them; a ValueError names the report that does not."""
    summary = reports.summary
    shipped_run, twin_run = summary["shipped"]["run"], summary["twin"]["run"]
    shipped, twin = reports.runs[shipped_run], reports.runs[twin_run]
    for name, report in ((shipped_run, shipped), (twin_run, twin)):
        if report.get("run") != name:
            raise ValueError(f"runs/{name}.json is the report of run {report.get('run')!r}")
    if (twin["model"], twin["frame_lags"]) != (shipped["model"], shipped["frame_lags"]):
        raise ValueError(f"the twin {twin_run!r} is {twin['model']} with frame lags {twin['frame_lags']}, "
                         f"but the shipped run is {shipped['model']} with {shipped['frame_lags']}")
    if "cross_drive" not in shipped:
        raise ValueError(f"the report of {shipped_run!r} has no score on another drive; "
                         "evaluate it with --cross-drive")
    drive, cross_drive = shipped["drive"], shipped["cross_drive"]["drive"]
    if cross_drive == drive:
        raise ValueError(f"the shipped run's other drive is its own training drive {drive!r}")
    splits = {"firewall": shipped["splits"]["firewall"], "block": shipped["splits"]["block"]}
    leakage = reports.leakage
    if (leakage.get("drive"), leakage.get("splits")) != (drive, splits):
        raise ValueError(f"leakage.json analyses drive {leakage.get('drive')!r} with {leakage.get('splits')}, "
                         f"but the shipped run trained on {drive!r} with {splits}")
    if reports.legacy.get("splits") != splits:
        raise ValueError(f"legacy.json scores test splits made with {reports.legacy.get('splits')}, "
                         f"but the shipped run trained with {splits}")
    for name, info in reports.legacy.get("drives", {}).items():
        recorded = summary["drives"].get(name)
        if recorded is not None and info.get("label_sha256") != recorded.get("label_sha256"):
            raise ValueError(f"legacy.json scored other labels of drive {name!r} than the run reports")
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
    if report.get("checkpoint_sha256") != digest:
        raise UnusableRunError(f"run {run!r}: its best.pt is another checkpoint than the one its report "
                               "describes; evaluate it again (`sightline eval --all`) and summarise "
                               "again (`sightline report`)")
    if [int(lag) for lag in model.frame_lags] != report["frame_lags"]:
        raise UnusableRunError(f"run {run!r} stacks frame lags {list(model.frame_lags)}, "
                               f"but its report says {report['frame_lags']}")
    return _Model(key=key, run=run, name=history["config"]["model"], model=model.eval(),
                  checkpoint_sha256=digest)


def _outputs(model: torch.nn.Module, inputs: np.ndarray, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Steering degrees (N,) and saliency maps (N, 66, 200) from PyTorch on the CPU, as
    float32: the outputs of the graph the page runs."""
    wrapper = SteeringWithSaliency(model).eval().to("cpu")
    angles, maps = [], []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(np.array(inputs[start:start + batch_size], dtype=np.uint8))
            angle, saliency = wrapper(batch)
            angles.append(angle[:, 0].numpy().astype(np.float32))
            maps.append(saliency.numpy().astype(np.float32))
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
                  step: int = CLIP_STEP) -> dict:
    """The clip window: ``n`` scored frames, with the ``lead_in`` frames before them in the
    same span, whose labels have the largest (population) standard deviation.

    Windows start every ``step`` frames from the first that fits in each ``(start, stop)``
    span; ties go to the earliest. Returns its ``first_frame`` (the first scored frame), its
    ``label_sd_deg`` and how many ``windows`` were compared; a ValueError when none fits.
    """
    angles = np.asarray(angles_deg, dtype=np.float64)
    best_first, best_sd, windows = None, -1.0, 0
    for start, stop in spans:
        for first in range(start + lead_in, stop - n + 1, step):
            windows += 1
            sd = float(np.std(angles[first:first + n]))
            if sd > best_sd:
                best_first, best_sd = first, sd
    if best_first is None:
        longest = max((stop - start for start, stop in spans), default=0)
        raise ValueError(f"no stretch holds {lead_in} history frames and {n} more "
                         f"(the longest has {longest} frames)")
    return {"first_frame": best_first, "label_sd_deg": best_sd, "windows": windows}


def min_gap_to_training(frames: np.ndarray, trained: np.ndarray, minimum: int) -> int:
    """The distance in frames from ``frames`` to the nearest of ``trained``; a ValueError
    when it is below ``minimum``."""
    gap = int(frame_gaps(trained, frames).min())
    if gap < minimum:
        raise ValueError(f"frame {int(frames[np.argmin(frame_gaps(trained, frames))])} is {gap} frames from a "
                         f"training frame; every frame must be at least {minimum} away")
    return gap


def _rule(where: str, n: int, lead_in: int) -> str:
    return (f"the {n}-frame window whose steering labels have the largest standard deviation, among the "
            f"windows that fit with the {lead_in} frames before them inside {where}; windows start every "
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

def _replace_outputs(staging: Path, out_dir: Path) -> None:
    """Remove the previous build's files and move the staged ones into place."""
    for folder, patterns in GENERATED.items():
        for pattern in patterns:
            for old in (out_dir / folder).glob(pattern):
                old.unlink()
    for path in sorted(p for p in staging.rglob("*") if p.is_file()):
        target = out_dir / path.relative_to(staging)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, target)


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

    Refuses, before writing anything: a missing ffmpeg when ``video`` is on; reports that
    disagree on the drive or the split settings; a run whose best.pt is not the checkpoint
    its report describes; a cache that is missing or not built from the drive's labels as
    they are now; a drive whose labels are not the ones the reports were made from; a 2018
    clip closer than 2·firewall + 1 frames to a frame the shipped run trained on.
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
    # the frames the shipped run held out for testing, and the ones it trained on, as it saved them
    shipped_dir = workspace.run_dir(summary["shipped"]["run"])
    held_out, trained = load_split_indices(shipped_dir, "test"), load_split_indices(shipped_dir, "train")
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
        clips[cid] = {"drive": drive, "rule": _rule(where, clip_frames, lead_in), **window,
                      "min_frames_from_training": None}
    # the page says the 2018 clip shows minutes the model never trained on: measured, not assumed
    first = clips["c2018"]["first_frame"]
    try:
        clips["c2018"]["min_frames_from_training"] = min_gap_to_training(
            np.arange(first - lead_in, first + clip_frames), trained, 2 * firewall + 1)
    except ValueError as exc:
        raise ValueError(f"the c2018 clip of {setting['drive']} is too close to training: {exc}") from None

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir.parent, prefix=f".{out_dir.name}-") as tmp:
        staging = Path(tmp)
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

        # the models, kept only if onnxruntime reproduces PyTorch on every input the page feeds them
        check = np.concatenate([stacks["c2018"], stacks["c2017"], selftest_inputs])
        exported = {m.key: export_onnx(m.model, staging / MODEL_FILES[m.key], check, model_name=m.name, run=m.run,
                                       max_bytes=MAX_MODEL_BYTES)
                    for m in models}
        expected = {cid: {m.key: _outputs(m.model, stacks[cid])[0] for m in models} for cid in clips}
        selftest = {m.key: _outputs(m.model, selftest_inputs) for m in models}

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
                "sprite": {"files": sheets[cid], "frames_per_sheet": FRAMES_PER_SHEET, "frame_w": INPUT_WIDTH,
                           "frame_h": INPUT_HEIGHT, "decoded_sha256": clip["decoded_sha256"]},
                "video": None if size is None else "clip.mp4",
                "video_size": None if size is None else list(size),
                "camera_size": list(CAMERA_SIZE),
                "clip_mae_deg": clip["clip_mae_deg"],
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
        _replace_outputs(staging, out_dir)

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
                           "max_saliency_diff": exported[m.key]["max_saliency_diff"]},
            }
            for m in models
        },
        "clips": {
            cid: {"drive": clip["drive"], "first_frame": clip["first_frame"], "n": clip_frames, "lead_in": lead_in,
                  "rule": clip["rule"], "label_sd_deg": clip["label_sd_deg"], "windows": clip["windows"],
                  "min_frames_from_training": clip["min_frames_from_training"],
                  "clip_mae_deg": clip["clip_mae_deg"], "video": clip["video"]}
            for cid, clip in clips.items()
        },
        "selftest": {"clip": "c2018", "frames": [int(f) for f in selftest_frames]},
        "versions": _versions(ffmpeg),
    }
    write_report(reports_dir / "export.json", export)
    return export
