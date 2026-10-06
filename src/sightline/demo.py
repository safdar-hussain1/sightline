"""Cockpit demo renderer: a stretch of a drive as a video of what the model sees and steers.

Each video frame shows the camera frame and, beside it, the same frame with the model's
VisualBackProp saliency laid over the rows the model sees (the bottom ``CROP_BOTTOM`` rows
at full width; the rows above them never reach the model and are dimmed). Under them, the
instrument panel: the readouts (the model's angle, the human's, the error and the mean
error so far), a steering wheel turned to the model's angle with the human's angle as a
ghost ring on its rim, and both angles over the last few seconds.

The model gets exactly the input evaluation gives it: each frame from the drive cache,
stacked with the history frames its run records (``frame_lags``), scored on the CPU, the
device every published number comes from. The demo reads the cache and never builds one
(``sightline cache`` does). It runs headless: the output is an mp4 file (OpenCV ``mp4v``),
written under a temporary name and moved into place once every frame is in it.

The wheel shows the model's angle through an exponential display filter: each frame the
displayed angle moves the share ``DISPLAY_ALPHA`` of the way to the prediction. It is
bounded and defined when the two are equal, unlike the ``|delta|^(2/3)`` update common in
steering demos, which divides by ``|delta|`` the moment the prediction matches the display.
Only the wheel is smoothed; the readouts and the angle strip show the model's own output.
"""

from __future__ import annotations

import contextlib
import math
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from sightline.data.frames import stack_frames
from sightline.data.preprocess import CROP_BOTTOM
from sightline.paths import MissingDataError, Workspace
from sightline.saliency import SteeringWithSaliency

__all__ = [
    "DISPLAY_ALPHA",
    "HUMAN_BGR",
    "MODEL_BGR",
    "PANEL",
    "DemoResult",
    "draw_wheel",
    "render_demo",
    "saliency_overlay",
    "smooth_toward",
]

#: share of the remaining gap the wheel's displayed angle closes each frame
DISPLAY_ALPHA = 0.35
#: height of the instrument panel under the camera frames, and the wheel's size, in pixels
PANEL = 256
#: seconds of driving the angle strip shows
STRIP_SECONDS = 6.0
#: frames scored per forward pass
BATCH = 64
#: weight of the saliency heat map where it is blended over the camera frame
SALIENCY_WEIGHT = 0.55

# colours, BGR
MODEL_BGR = (214, 120, 42)    # the model: blue
HUMAN_BGR = (52, 177, 242)    # the human driver: amber
INK_BGR = (235, 235, 235)
MUTED_BGR = (150, 150, 150)
WHEEL_BGR = (120, 120, 120)
SURFACE_BGR = (18, 18, 18)
STRIP_BGR = (34, 34, 34)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def smooth_toward(display: float, target: float, alpha: float = DISPLAY_ALPHA) -> float:
    """One step of the display filter: ``display`` moved the share ``alpha`` of the way to
    ``target``. Never past the target, and ``display`` itself when the two are equal."""
    if not 0.0 < alpha <= 1.0:  # a NaN fails too
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    return display + alpha * (target - display)


def draw_wheel(size: int, angle_deg: float, ghost_deg: float | None = None) -> np.ndarray:
    """A steering wheel ``size`` pixels square, turned to ``angle_deg`` (clockwise positive).

    The spoke that points straight up when the wheel is centred, and the dot where it meets
    the rim, are the model's; ``ghost_deg`` (the human's angle) is a ring on the rim. Any
    angle draws, past half a turn too, and equal angles draw the ring around the dot.
    """
    img = np.full((size, size, 3), SURFACE_BGR, dtype=np.uint8)
    c = size // 2
    r = int(size * 0.42)

    def on_rim(theta_deg: float) -> tuple[int, int]:
        t = math.radians(theta_deg)
        return c + int(round(r * math.sin(t))), c - int(round(r * math.cos(t)))

    cv2.circle(img, (c, c), r, WHEEL_BGR, max(2, size // 40), cv2.LINE_AA)
    cv2.circle(img, (c, c), int(r * 0.16), WHEEL_BGR, -1, cv2.LINE_AA)
    for offset in (120.0, 240.0):
        cv2.line(img, (c, c), on_rim(angle_deg + offset), WHEEL_BGR, max(2, size // 40), cv2.LINE_AA)
    cv2.line(img, (c, c), on_rim(angle_deg), MODEL_BGR, max(3, size // 32), cv2.LINE_AA)
    cv2.circle(img, on_rim(angle_deg), max(4, size // 23), MODEL_BGR, -1, cv2.LINE_AA)
    if ghost_deg is not None:
        cv2.circle(img, on_rim(ghost_deg), max(6, size // 16), HUMAN_BGR, max(2, size // 64), cv2.LINE_AA)
    return img


def saliency_overlay(frame_bgr: np.ndarray, saliency: np.ndarray, weight: float = SALIENCY_WEIGHT) -> np.ndarray:
    """A copy of ``frame_bgr`` with ``saliency`` laid over the rows the model sees.

    ``saliency`` is a map in [0, 1] over the model's input (66 x 200), which is the bottom
    ``CROP_BOTTOM`` rows of the frame at full width; it is stretched back over them as an
    inferno heat map blended at ``weight``. The rows above are left as they are.
    """
    height, width = frame_bgr.shape[:2]
    if height < CROP_BOTTOM:
        raise ValueError(f"frame height {height} is less than the {CROP_BOTTOM}-row crop the model sees")
    heat = np.clip(np.rint(np.asarray(saliency, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)
    heat = cv2.resize(heat, (width, CROP_BOTTOM), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    out = frame_bgr.copy()
    out[height - CROP_BOTTOM:] = cv2.addWeighted(frame_bgr[height - CROP_BOTTOM:], 1.0 - weight, heat, weight, 0.0)
    return out


@dataclass(frozen=True, eq=False)
class DemoResult:
    """What a render wrote: the file, the stretch of the drive and both angles per frame."""

    path: Path
    drive: str
    run: str
    first_frame: int
    fps: float
    size: tuple[int, int]
    device: str
    #: the model's angle on each rendered frame, unfiltered (degrees)
    model_deg: np.ndarray
    #: the human's angle on each rendered frame, from the cache's repaired labels (degrees)
    human_deg: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.model_deg.size)

    @property
    def mae_deg(self) -> float:
        return float(np.mean(np.abs(self.model_deg - self.human_deg)))

    @property
    def zero_mae_deg(self) -> float:
        """What always steering straight ahead scores on the same frames."""
        return float(np.mean(np.abs(self.human_deg)))


def _label(img: np.ndarray, text: str, origin: tuple[int, int], colour, scale: float = 0.6,
           thickness: int = 1) -> None:
    cv2.putText(img, text, origin, _FONT, scale, colour, thickness, cv2.LINE_AA)


def _tag(img: np.ndarray, text: str, x: int, y: int) -> None:
    """A caption on a dark box, legible over any camera frame."""
    (w, h), base = cv2.getTextSize(text, _FONT, 0.5, 1)
    cv2.rectangle(img, (x - 6, y - h - 6), (x + w + 6, y + base + 4), SURFACE_BGR, -1)
    _label(img, text, (x, y), INK_BGR, 0.5)


def _strip(img: np.ndarray, box: tuple[int, int, int, int], human: list[float], model: list[float],
           length: int, scale: float) -> None:
    """Both angles over the last ``length`` frames, the newest at the right edge, between
    -``scale`` and +``scale`` degrees (values beyond are drawn at the edge)."""
    x0, y0, x1, y1 = box
    cv2.rectangle(img, (x0, y0), (x1, y1), STRIP_BGR, -1)
    mid = (y0 + y1) / 2.0
    cv2.line(img, (x0, int(round(mid))), (x1, int(round(mid))), WHEEL_BGR, 1, cv2.LINE_AA)

    def points(values: list[float]) -> np.ndarray:
        shown = np.asarray(values[-length:], dtype=np.float64)
        xs = x1 - (shown.size - 1 - np.arange(shown.size)) * (x1 - x0) / max(length - 1, 1)
        ys = mid - np.clip(shown / scale, -1.0, 1.0) * (y1 - y0) / 2.0
        return np.rint(np.stack([xs, ys], axis=1) * 16).astype(np.int32)  # 4 fractional bits

    for values, colour in ((human, HUMAN_BGR), (model, MODEL_BGR)):
        if len(values) > 1:
            cv2.polylines(img, [points(values)], False, colour, 2, cv2.LINE_AA, shift=4)
    # the labels sit outside the box, where no trace can cross them
    _label(img, f"+{scale:.0f} deg", (x0, y0 - 8), MUTED_BGR, 0.45)
    _label(img, f"-{scale:.0f} deg", (x0, y1 + 20), MUTED_BGR, 0.45)
    caption = f"last {STRIP_SECONDS:.0f} s"
    (w, _), _ = cv2.getTextSize(caption, _FONT, 0.45, 1)
    _label(img, caption, (x1 - w, y1 + 20), MUTED_BGR, 0.45)


def _compose(frame: np.ndarray, saliency: np.ndarray, wheel_deg: float, model: list[float], human: list[float],
             abs_errors: float, drive: str, index: int, strip_length: int, strip_scale: float,
             size: tuple[int, int]) -> np.ndarray:
    """One video frame: the camera, the saliency view and the instrument panel."""
    fh, fw = frame.shape[:2]
    width, height = size
    canvas = np.full((height, width, 3), SURFACE_BGR, dtype=np.uint8)
    canvas[:fh, :fw] = frame
    view = saliency_overlay(frame, saliency)
    top = fh - CROP_BOTTOM
    view[:top] = (view[:top] * 0.45).astype(np.uint8)  # the model never sees these rows
    cv2.line(view, (0, top), (fw - 1, top), INK_BGR, 1, cv2.LINE_AA)
    canvas[:fh, fw:2 * fw] = view
    _tag(canvas, "camera", 12, 24)
    _tag(canvas, "where the model looks (it sees below the line)", fw + 12, 24)

    x_wheel = width // 2 - PANEL // 2
    canvas[fh:fh + PANEL, x_wheel:x_wheel + PANEL] = draw_wheel(PANEL, wheel_deg, ghost_deg=human[-1])

    angle, label, error = model[-1], human[-1], abs(model[-1] - human[-1])
    _label(canvas, f"model {angle:+8.2f} deg", (24, fh + 46), MODEL_BGR, 0.75, 2)
    _label(canvas, f"human {label:+8.2f} deg", (24, fh + 90), HUMAN_BGR, 0.75, 2)
    _label(canvas, f"error {error:8.2f} deg", (24, fh + 134), INK_BGR, 0.75, 2)
    _label(canvas, f"mean error so far {abs_errors / len(model):6.2f} deg", (24, fh + 178), MUTED_BGR, 0.55)
    _label(canvas, f"{drive}, frame {index}", (24, fh + 222), MUTED_BGR, 0.55)

    x_strip = x_wheel + PANEL + 24
    _strip(canvas, (x_strip, fh + 34, width - 24, fh + PANEL - 42), human, model, strip_length, strip_scale)
    return canvas


def _even(n: int) -> int:
    return n + (n % 2)


def _read_frame(labels, drive: str, index: int) -> np.ndarray:
    path = labels.frame_path(index)
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise MissingDataError(f"frame {index} of drive {drive!r} cannot be read: {path}")
    return frame


def render_demo(
    drive_name: str,
    run_dir: Path,
    out_path: Path,
    workspace: Workspace,
    start: int = 0,
    num_frames: int = 600,
    fps: float = 17.5,
) -> DemoResult:
    """Render frames ``start .. start + num_frames - 1`` of ``drive_name`` with the model of
    ``run_dir`` into the mp4 ``out_path`` (see the module docstring)."""
    from sightline.export.onnx_export import open_drive_cache
    from sightline.reporting import load_run_model

    run_dir, out_path = Path(run_dir), Path(out_path)
    if start < 0 or num_frames < 1:
        raise ValueError(f"cannot render {num_frames} frames from frame {start}")
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"fps must be a positive number, got {fps}")
    model, _ = load_run_model(run_dir)  # a missing or unusable run is the first thing reported
    workspace.drive(drive_name)  # then a missing drive, before the cache it would have
    labels, cache = open_drive_cache(workspace, drive_name)
    n = len(cache)
    end = start + num_frames
    if end > n:
        raise ValueError(f"drive {drive_name!r} has {n} frames (0..{n - 1}); "
                         f"frames {start}..{end - 1} run past its end")

    device = torch.device("cpu")
    wrapper = SteeringWithSaliency(model).to(device).eval()
    lags = tuple(model.frame_lags)
    human_all = np.asarray(cache.angles_deg[start:end], dtype=np.float32)
    strip_length = max(2, int(round(STRIP_SECONDS * fps)))
    strip_scale = max(10.0, 1.1 * float(np.abs(human_all).max()))

    first = _read_frame(labels, drive_name, start)
    fh, fw = first.shape[:2]
    size = (_even(2 * fw), _even(fh + PANEL))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(f".{out_path.stem}-{os.getpid()}-{secrets.token_hex(4)}.partial{out_path.suffix}")
    writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    model_deg: list[float] = []
    human_deg: list[float] = []
    try:
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV cannot write an mp4v video to {out_path}")
        wheel_deg = None
        abs_errors = 0.0
        for batch_start in range(start, end, BATCH):
            indices = np.arange(batch_start, min(batch_start + BATCH, end), dtype=np.int64)
            stacks = torch.from_numpy(stack_frames(cache.inputs, indices, lags)).to(device)
            with torch.inference_mode():
                angles, saliency = wrapper(stacks)
            angles = angles[:, 0].numpy()
            saliency = saliency.numpy()
            for k, index in enumerate(indices):
                frame = first if index == start else _read_frame(labels, drive_name, int(index))
                if frame.shape[:2] != (fh, fw):
                    raise ValueError(f"frame {index} of drive {drive_name!r} is {frame.shape[1]}x{frame.shape[0]}, "
                                     f"not {fw}x{fh} like frame {start}")
                angle = float(angles[k])
                model_deg.append(angle)
                human_deg.append(float(human_all[index - start]))
                abs_errors += abs(angle - human_deg[-1])
                wheel_deg = angle if wheel_deg is None else smooth_toward(wheel_deg, angle)
                writer.write(_compose(frame, saliency[k], wheel_deg, model_deg, human_deg, abs_errors,
                                      drive_name, int(index), strip_length, strip_scale, size))
        writer.release()
        os.replace(partial, out_path)
    finally:
        writer.release()
        with contextlib.suppress(OSError):
            partial.unlink(missing_ok=True)

    return DemoResult(path=out_path, drive=drive_name, run=run_dir.name, first_frame=start, fps=float(fps),
                      size=size, device=device.type, model_deg=np.asarray(model_deg, dtype=np.float32),
                      human_deg=human_all)
