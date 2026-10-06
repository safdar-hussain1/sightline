"""Cockpit demo renderer.

Renders a drive segment to video: camera frame, saliency heat overlay, a
drawn steering wheel showing the model's angle with the human's angle as a
ghost needle, and a live error readout. Runs headless — output is a video
file, no GUI required.

The wheel is drawn with OpenCV primitives (no bitmap assets), and the angle
display is smoothed with a plain exponential filter — bounded, and defined
at zero error, unlike the classic ``|delta|^(2/3)`` update, which divides by
zero the moment the prediction matches the display.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch

from sightline.data.cache import build_cache
from sightline.data.frames import stack_frames
from sightline.data.labels import load_labels, repair_dropouts
from sightline.paths import Workspace
from sightline.saliency import SteeringWithSaliency
from sightline.train import pick_device

__all__ = ["render_demo", "draw_wheel"]


def draw_wheel(size: int, angle_deg: float, ghost_deg: float | None = None) -> np.ndarray:
    """Draw a steering wheel at ``angle_deg`` (clockwise positive)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    c = size // 2
    r = int(size * 0.42)
    cv2.circle(img, (c, c), r, (235, 235, 235), max(2, size // 40), cv2.LINE_AA)
    cv2.circle(img, (c, c), int(r * 0.16), (235, 235, 235), -1, cv2.LINE_AA)

    def spoke(theta_deg: float, color: tuple, thickness: int) -> None:
        t = math.radians(theta_deg)
        x = c + int(r * math.sin(t))
        y = c - int(r * math.cos(t))
        cv2.line(img, (c, c), (x, y), color, thickness, cv2.LINE_AA)

    if ghost_deg is not None:
        spoke(ghost_deg, (90, 90, 90), max(2, size // 60))
    for offset in (0.0, 120.0, 240.0):
        spoke(angle_deg + offset, (235, 235, 235), max(2, size // 40))
    spoke(angle_deg, (80, 190, 255), max(3, size // 30))
    return img


def render_demo(
    drive_name: str,
    run_dir: Path,
    out_path: Path,
    workspace: Workspace,
    start: int = 0,
    num_frames: int = 600,
    fps: float = 17.5,
) -> Path:
    from sightline.reporting import load_run_model

    labels = repair_dropouts(load_labels(workspace.drive(drive_name)))
    cache = build_cache(labels, workspace.cache_dir(drive_name))
    model, _ = load_run_model(run_dir)
    device = pick_device()
    wrapper = SteeringWithSaliency(model).to(device).eval()

    end = min(start + num_frames, len(cache))
    if end <= start:
        raise ValueError(f"empty segment {start}..{end}")

    frame0 = cv2.imread(str(labels.frame_path(start)))
    fh, fw = frame0.shape[:2]
    panel = 256
    out_w, out_h = fw * 2, fh + panel
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {out_path}")

    smoothed = 0.0
    for i in range(start, end):
        frame = cv2.imread(str(labels.frame_path(i)))
        model_in = torch.from_numpy(stack_frames(cache.inputs, np.array([i]), model.frame_lags)).to(device)
        with torch.no_grad():
            angle_t, sal_t = wrapper(model_in)
        angle = float(angle_t[0, 0])
        human = float(cache.angles_deg[i])
        smoothed += 0.35 * (angle - smoothed)

        sal = (sal_t[0].cpu().numpy() * 255).astype(np.uint8)
        sal_color = cv2.applyColorMap(sal, cv2.COLORMAP_INFERNO)
        sal_big = cv2.resize(sal_color, (fw, 150), interpolation=cv2.INTER_LINEAR)
        overlay = frame.copy()
        roi = overlay[-150:]
        overlay[-150:] = cv2.addWeighted(roi, 0.45, sal_big, 0.55, 0)

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:fh, :fw] = frame
        canvas[:fh, fw:] = overlay
        wheel = draw_wheel(panel, smoothed, ghost_deg=human)
        x0 = out_w // 2 - panel // 2
        canvas[fh:, x0 : x0 + panel] = wheel
        cv2.putText(canvas, f"model {angle:+7.2f} deg", (24, fh + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 190, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"human {human:+7.2f} deg", (24, fh + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 160, 160), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"error {abs(angle - human):6.2f} deg", (24, fh + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
        writer.write(canvas)

    writer.release()
    return out_path
