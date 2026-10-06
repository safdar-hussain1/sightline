"""The standard-recipe PilotNet: a model trained the widely copied way, scored like Sightline's own.

A widely copied TensorFlow 1 recipe trains PilotNet on frames shuffled before
an 80/20 split, with MSE on the angle in radians plus L2 on every variable,
behind a ``2*atan`` output head that can never pass +/-180 deg. Sightline
benchmarks one such model, trained on the 2017 drive, from its weights alone:
the twenty TF1 tensors ``Variable`` .. ``Variable_19`` saved as an ``.npz``.
The weights are not part of the repository; the ``legacy`` command takes
their path from ``--weights`` or ``SIGHTLINE_LEGACY_WEIGHTS``.

``reference_forward``
    The TF1 graph in NumPy, written independently of the PyTorch port: NHWC
    activations, HWIO kernels, VALID padding, strides 2, 2, 2, 1, 1, ReLU
    after every layer but the head, the NHWC flatten into the fully-connected
    layers 1152-1164-100-50-10-1, and ``2*atan(z)`` in radians, reported in
    degrees. ``PilotNet.load_legacy_npz``, the port, is checked against it.
``preprocess_legacy`` and ``to_model_input``
    The recipe's own preprocessing: the bottom 150 rows, ``cv2.resize`` to
    200x66 with its default INTER_LINEAR, BGR kept, divided by 255.
    ``to_model_input`` gives the port the same resized pixels as uint8 RGB;
    the port's first layer absorbs the channel order and the scaling.
``evaluate_legacy``
    Scores the port on the 2018 drive's blocked and forward test frames, the
    frames Sightline's own models are tested on (for this model, another
    day), and on the 2017 drive, its own training drive, from frame
    ``CROSS_FIRST_FRAME`` on, the frames the run reports score as another
    day: frames it has seen, the kind its shuffled split scored it on. Frames
    are decoded from the JPEGs, because Sightline's cache resizes with
    INTER_AREA, which this model never saw; targets are the repaired labels
    as the cache stores them (float32), as in every run report.
"""

from __future__ import annotations

import hashlib
import math
import platform
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from sightline.data.labels import DriveLabels, load_labels, repair_dropouts
from sightline.data.preprocess import CROP_BOTTOM, INPUT_HEIGHT, INPUT_WIDTH
from sightline.data.splits import make_split
from sightline.data.verify import label_sha256
from sightline.evaluate import pearson_r, regime_breakdown, regression_metrics
from sightline.models.pilotnet import PilotNet
from sightline.paths import MissingDataError, Workspace
from sightline.reporting import CROSS_FIRST_FRAME
from sightline.train import code_version

__all__ = [
    "CAP_DEG",
    "CONV_STRIDES",
    "OTHER_DAY",
    "PARITY_FRAMES_PER_SET",
    "PARITY_LIMIT_DEG",
    "RECIPE",
    "TENSOR_SHAPES",
    "TRAINED_ON",
    "WEIGHTS_ENV",
    "conv2d_valid",
    "evaluate_legacy",
    "load_weights",
    "predict_frames",
    "preprocess_legacy",
    "read_frame",
    "reference_forward",
    "to_model_input",
]

RECIPE = "PilotNet, TF1, shuffled 80/20 split, MSE in radians + L2, 2·atan head"
#: the drive the model was trained on, and the drive it never saw
TRAINED_ON, OTHER_DAY = "drive2017", "drive2018"
#: the environment variable the ``legacy`` command reads the weights path from
WEIGHTS_ENV = "SIGHTLINE_LEGACY_WEIGHTS"
#: the ``2*atan`` head's outputs lie strictly between -CAP_DEG and +CAP_DEG
CAP_DEG = 180.0
#: the port is checked against the TF1 graph on this many frames of every
#: evaluation set, spread evenly over it (all of a smaller set)
PARITY_FRAMES_PER_SET = 64
#: the evaluation stops, writing no report, when the port and the TF1 graph differ by more
PARITY_LIMIT_DEG = 1e-2

#: the TF1 tensors in creation order: five conv layers, then the four
#: fully-connected layers and the head, each a kernel followed by its bias
TENSOR_SHAPES: dict[str, tuple[int, ...]] = {
    "Variable": (5, 5, 3, 24), "Variable_1": (24,),
    "Variable_2": (5, 5, 24, 36), "Variable_3": (36,),
    "Variable_4": (5, 5, 36, 48), "Variable_5": (48,),
    "Variable_6": (3, 3, 48, 64), "Variable_7": (64,),
    "Variable_8": (3, 3, 64, 64), "Variable_9": (64,),
    "Variable_10": (1152, 1164), "Variable_11": (1164,),
    "Variable_12": (1164, 100), "Variable_13": (100,),
    "Variable_14": (100, 50), "Variable_15": (50,),
    "Variable_16": (50, 10), "Variable_17": (10,),
    "Variable_18": (10, 1), "Variable_19": (1,),
}
CONV_STRIDES = (2, 2, 2, 1, 1)


# --- the weights and the TF1 graph -------------------------------------------------------

def _check_tensors(weights) -> None:
    missing = [name for name in TENSOR_SHAPES if name not in weights]
    if missing:
        raise ValueError(f"the weights lack the TF1 tensor{'s' if len(missing) > 1 else ''} {', '.join(missing)}")
    for name, shape in TENSOR_SHAPES.items():
        found = tuple(np.shape(weights[name]))
        if found != shape:
            raise ValueError(f"tensor {name} has shape {found}, expected {shape}")


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    if not hasattr(data, "files"):
        raise ValueError("it holds a single array")
    with data:
        return {name: np.asarray(data[name], dtype=np.float32) for name in TENSOR_SHAPES if name in data.files}


def load_weights(path: str | Path) -> dict[str, np.ndarray]:
    """The twenty TF1 tensors of an ``.npz`` as float32 arrays, checked by name and shape.

    Any other tensors in the file are left out.
    """
    path = Path(path)
    if not path.is_file():
        raise MissingDataError(f"standard-recipe PilotNet weights not found: {path}")
    try:
        weights = _read_npz(path)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{path} is not an .npz of TF1 tensors ({exc})") from None
    _check_tensors(weights)
    return weights


def conv2d_valid(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray, stride: int) -> np.ndarray:
    """TF1's ``conv2d`` with VALID padding, plus the bias: NHWC input, HWIO kernel and the
    same stride along both axes, so each output axis has ``ceil((size - k + 1) / stride)``
    positions."""
    rows, cols = kernel.shape[:2]
    windows = sliding_window_view(x, (rows, cols), axis=(1, 2))[:, ::stride, ::stride]
    # windows: (N, out rows, out cols, C, k rows, k cols); the kernel is (k rows, k cols, C, O)
    return np.tensordot(windows, kernel, axes=([3, 4, 5], [2, 0, 1])) + bias


def reference_forward(weights: dict[str, np.ndarray], x_bgr01: np.ndarray, chunk: int = 16) -> np.ndarray:
    """Steering angles in degrees from the TF1 graph, for frames prepared as the recipe did.

    ``x_bgr01`` holds (N, 66, 200, 3) frames, or one 66x200x3 frame, in BGR order
    scaled to [0, 1]: ``preprocess_legacy``'s output. Everything is computed in
    float64, ``chunk`` frames at a time, with dropout off as in the recipe's
    inference (keep_prob 1). Returns N angles.
    """
    _check_tensors(weights)
    params = [np.asarray(weights[name], dtype=np.float64) for name in TENSOR_SHAPES]
    x = np.asarray(x_bgr01, dtype=np.float64)
    if x.ndim == 3:
        x = x[None]
    if x.ndim != 4 or x.shape[1:] != (INPUT_HEIGHT, INPUT_WIDTH, 3):
        raise ValueError(f"expected (N, {INPUT_HEIGHT}, {INPUT_WIDTH}, 3) BGR frames, got shape {x.shape}")
    n_conv, n_layers = len(CONV_STRIDES), len(params) // 2
    out = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], chunk):
        h = x[start:start + chunk]
        for layer, stride in enumerate(CONV_STRIDES):
            h = np.maximum(conv2d_valid(h, params[2 * layer], params[2 * layer + 1], stride), 0.0)
        h = h.reshape(h.shape[0], -1)  # tf.reshape of the NHWC map: row, then column, then channel
        for layer in range(n_conv, n_layers - 1):
            h = np.maximum(h @ params[2 * layer] + params[2 * layer + 1], 0.0)
        z = h @ params[-2] + params[-1]
        out[start:start + chunk] = np.degrees(2.0 * np.arctan(z[:, 0]))
    return out


# --- the recipe's preprocessing ----------------------------------------------------------

def _recipe_resize(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
        raise ValueError(f"expected an HxWx3 uint8 BGR frame, got {frame_bgr.dtype} {frame_bgr.shape}")
    if frame_bgr.shape[0] < CROP_BOTTOM:
        raise ValueError(f"frame height {frame_bgr.shape[0]} < crop {CROP_BOTTOM}")
    # INTER_LINEAR is cv2.resize's default, the interpolation the recipe used
    return cv2.resize(frame_bgr[-CROP_BOTTOM:], (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)


def preprocess_legacy(frame_bgr: np.ndarray) -> np.ndarray:
    """The recipe's input, ``cv2.resize(frame[-150:], (200, 66)) / 255.0``: float32 BGR in [0, 1]."""
    return (_recipe_resize(frame_bgr) / 255.0).astype(np.float32)


def to_model_input(frame_bgr: np.ndarray) -> np.ndarray:
    """The same resized pixels as the port takes them: uint8 RGB, 66x200."""
    return cv2.cvtColor(_recipe_resize(frame_bgr), cv2.COLOR_BGR2RGB)


def read_frame(labels: DriveLabels, index: int) -> np.ndarray:
    """Frame ``index`` of a drive, decoded from its JPEG as BGR."""
    path = labels.frame_path(index)
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise MissingDataError(f"unreadable frame: {path}")
    return frame


def predict_frames(model: torch.nn.Module, labels: DriveLabels, indices: np.ndarray,
                   device: str | torch.device = "cpu", batch_size: int = 256, workers: int = 8,
                   desc: str | None = None) -> np.ndarray:
    """The port's angles in degrees for frames ``indices`` of a drive, in order, each
    decoded from its JPEG and prepared with ``to_model_input``."""
    indices = np.asarray(indices, dtype=np.int64)
    device = torch.device(device)
    out = np.empty(indices.size, dtype=np.float64)
    model.eval()

    def load(index) -> np.ndarray:
        return to_model_input(read_frame(labels, int(index)))

    with ThreadPoolExecutor(max_workers=workers) as pool, torch.no_grad(), \
            tqdm(total=indices.size, desc=desc, unit="frame") as bar:
        for start in range(0, indices.size, batch_size):
            chunk = indices[start:start + batch_size]
            batch = torch.from_numpy(np.stack(list(pool.map(load, chunk)))).to(device)
            out[start:start + chunk.size] = model(batch)[:, 0].double().cpu().numpy()
            bar.update(chunk.size)
    return out


# --- the evaluation ----------------------------------------------------------------------

def _split_score(pred: np.ndarray, target: np.ndarray) -> dict:
    r = pearson_r(pred, target)
    return {
        "model": regression_metrics(pred, target),
        "regimes": regime_breakdown(pred, target),
        "pearson_r": r if math.isfinite(r) else None,
        "zero_mae_deg": float(np.abs(target).mean()),
    }


def _drive_info(labels: DriveLabels) -> dict:
    """What the run reports record of a drive, so the reports can be matched."""
    return {
        "frames": len(labels),
        "repaired_labels": int(labels.repaired_indices.size),
        "median_dt_s": labels.median_dt_s() if labels.has_timestamps else None,
        "label_sha256": label_sha256(labels.root / "data.txt"),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _spread(indices: np.ndarray, count: int) -> np.ndarray:
    """``count`` of ``indices`` spread evenly from the first to the last (all, if fewer)."""
    take = min(count, indices.size)
    return indices[np.unique(np.linspace(0, indices.size - 1, take).round().astype(np.int64))]


def evaluate_legacy(weights_path: str | Path, workspace: Workspace, device: str = "cpu",
                    firewall: int = 250, block: int = 1500) -> dict:
    """Score the standard-recipe PilotNet and return its report.

    The evaluation sets are the blocked and forward test frames of ``OTHER_DAY``,
    made with ``make_split`` from the drive's length, ``firewall`` and ``block``
    (Sightline's training settings by default), which share frames, and the
    frames of ``TRAINED_ON`` from ``CROSS_FIRST_FRAME`` on (``seen_first_frame``
    in the report), the frames the run reports score as another day. Every
    frame is predicted once. Targets are the repaired labels rounded to float32,
    as the drive cache stores them, so this report and a run report score a
    shared frame against the same number.

    ``port_parity`` compares the port's scored predictions with
    ``reference_forward`` on ``PARITY_FRAMES_PER_SET`` frames of every set, each
    frame once; a difference above ``PARITY_LIMIT_DEG`` raises ValueError and no
    report is made. ``atan_cap`` gives the largest predicted |angle| and, per
    drive, the labels of the whole drive steered more than ``CAP_DEG`` either
    way, which the head can never reach. The report names the weights by their
    sha256, never by their path; ``device`` is used as given. Every value is a
    plain int, float, str or None.
    """
    weights_path = Path(weights_path)
    weights = load_weights(weights_path)
    labels = {name: repair_dropouts(load_labels(workspace.drive(name))) for name in (OTHER_DAY, TRAINED_ON)}
    n = len(labels[OTHER_DAY])
    try:
        blocked = make_split("blocked", n, firewall=firewall, block=block)
        forward = make_split("forward", n, firewall=firewall, block=block)
    except ValueError as exc:
        raise ValueError(f"cannot split {OTHER_DAY!r}: {exc}") from None
    seen = np.arange(CROSS_FIRST_FRAME, len(labels[TRAINED_ON]), dtype=np.int64)
    if seen.size == 0:
        raise ValueError(f"{TRAINED_ON!r} has no frames from frame {CROSS_FIRST_FRAME} on")
    sets = {
        f"{OTHER_DAY}_blocked_test": (OTHER_DAY, blocked.test),
        f"{OTHER_DAY}_forward_test": (OTHER_DAY, forward.test),
        f"{TRAINED_ON}_all_seen": (TRAINED_ON, seen),
    }
    # the cache stores the angles as float32, and the run reports score against those
    targets = {drive: lab.angles_deg.astype(np.float32).astype(np.float64) for drive, lab in labels.items()}

    torch_device = torch.device(device)
    model = PilotNet()
    model.load_legacy_npz(weights_path)
    model.eval().to(torch_device)

    # every frame is predicted once, though the two test sets of OTHER_DAY share some
    predicted = {}
    for drive in labels:
        frames = np.unique(np.concatenate([idx for d, idx in sets.values() if d == drive]))
        predicted[drive] = (frames, predict_frames(model, labels[drive], frames, torch_device, desc=drive))

    def predictions(drive: str, idx: np.ndarray) -> np.ndarray:
        frames, values = predicted[drive]
        pos = np.minimum(np.searchsorted(frames, idx), frames.size - 1)
        if not np.array_equal(frames[pos], idx):
            raise RuntimeError(f"frames of {drive} were looked up that were never predicted")
        return values[pos]

    diffs = []
    for drive in labels:
        frames = np.unique(np.concatenate(
            [_spread(idx, PARITY_FRAMES_PER_SET) for d, idx in sets.values() if d == drive]))
        x = np.stack([preprocess_legacy(read_frame(labels[drive], int(i))) for i in frames])
        diffs.append(np.abs(predictions(drive, frames) - reference_forward(weights, x)))
    diffs = np.concatenate(diffs)
    worst = float(diffs.max())
    if not worst <= PARITY_LIMIT_DEG:  # a NaN fails too
        raise ValueError(f"the PyTorch port differs from the TF1 graph by up to {worst:.3g} deg "
                         f"on {diffs.size} frames (limit {PARITY_LIMIT_DEG:g} deg)")

    beyond = {}
    for drive, angles in targets.items():
        count = int((np.abs(angles) > CAP_DEG).sum())
        beyond[drive] = {"n": count, "frac": count / angles.size}
    every_prediction = np.concatenate([values for _, values in predicted.values()])

    return {
        "recipe": RECIPE,
        "trained_on": TRAINED_ON,
        "weights_sha256": _file_sha256(weights_path),
        "device": str(torch_device),
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
                     "python": platform.python_version()},
        "eval_code_version": code_version(),
        "splits": {"firewall": int(firewall), "block": int(block)},
        "seen_first_frame": int(CROSS_FIRST_FRAME),
        "drives": {drive: _drive_info(drive_labels) for drive, drive_labels in labels.items()},
        "port_parity": {"frames": int(diffs.size), "max_abs_diff_deg": worst},
        "eval": {key: _split_score(predictions(drive, idx), targets[drive][idx]) for key, (drive, idx) in sets.items()},
        "atan_cap": {"max_abs_pred_deg": float(np.abs(every_prediction).max()), "labels_beyond_180": beyond},
    }
