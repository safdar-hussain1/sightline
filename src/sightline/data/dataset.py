"""Torch dataset over the preprocessed cache, with training augmentation.

Each sample is the model input for one frame: the frame itself plus any
history frames (``frame_lags``), stacked along channels by ``stack_frames``.
Every geometric and photometric draw is made once per sample and applied to
all stacked frames alike, so augmentation never fakes motion between them;
only Gaussian noise is drawn per pixel. Augmentation works in float32 and
clips to [0, 255] and casts back to uint8 once, at the end.

Recipes (``augment``):

* ``none`` -- evaluation: the cached bytes, untouched.
* ``flip-only`` -- p=0.5 horizontal flip of all frames, angle negated (a
  mirrored road steers the other way).
* ``standard`` (PilotNet) -- the flip; p=0.8 brightness/contrast jitter,
  gain U(0.7, 1.3) and bias U(-25, 25); p=0.3 soft vertical shadow band, as
  cast by roadside structures (left edge U(0, 0.75 W), width U(0.15 W,
  0.5 W), shade U(0.5, 0.8)).
* ``strong`` (SightlineNet) -- everything in ``standard``, then p=0.5
  translation by ``np.roll`` with dy in [-4, 4] and dx in [-8, 8] pixels
  (label unchanged); p=0.5 colour cast, one U(0.85, 1.15) factor per RGB
  channel; p=0.3 Gaussian noise, sigma 6; p=0.3 cutout of a rectangle with
  height in [8, 24) and width in [16, 60) pixels, filled with 128.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from sightline.data.cache import DriveCache
from sightline.data.frames import stack_frames

__all__ = ["SteeringDataset", "AUGMENT_RECIPES"]

AUGMENT_RECIPES = ("none", "flip-only", "standard", "strong")


class SteeringDataset(Dataset):
    def __init__(
        self,
        cache: DriveCache,
        indices: np.ndarray,
        augment: str = "none",
        seed: int = 0,
        frame_lags: tuple[int, ...] = (),
    ):
        if augment not in AUGMENT_RECIPES:
            raise ValueError(f"unknown augmentation recipe {augment!r}; expected one of {AUGMENT_RECIPES}")
        indices = np.asarray(indices)
        if indices.min() < 0 or indices.max() >= len(cache):
            raise IndexError("split indices out of cache range")
        self.cache = cache
        self.indices = indices.astype(np.int64)
        self.augment = augment
        self.frame_lags = tuple(frame_lags)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[i])
        frames = stack_frames(self.cache.inputs, [idx], self.frame_lags)[0]  # uint8 HWC, 3 per frame
        angle = float(self.cache.angles_deg[idx])

        if self.augment != "none":
            frames, angle = self._augment(frames, angle)

        return (
            torch.from_numpy(np.ascontiguousarray(frames)),
            torch.tensor([angle], dtype=torch.float32),
        )

    def _augment(self, frames: np.ndarray, angle: float) -> tuple[np.ndarray, float]:
        rng = self.rng
        x = frames.astype(np.float32)  # a fresh copy: the in-place edits below never reach the cache
        h, w, channels = x.shape

        if rng.random() < 0.5:
            x = x[:, ::-1]
            angle = -angle

        if self.augment in ("standard", "strong"):
            if rng.random() < 0.8:
                x = x * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
            if rng.random() < 0.3:
                x0 = int(rng.uniform(0, w * 0.75))
                x1 = int(x0 + rng.uniform(w * 0.15, w * 0.5))
                x[:, x0:x1] *= rng.uniform(0.5, 0.8)

        if self.augment == "strong":
            if rng.random() < 0.5:
                dy, dx = int(rng.integers(-4, 5)), int(rng.integers(-8, 9))
                x = np.roll(x, (dy, dx), axis=(0, 1))
            if rng.random() < 0.5:
                cast = rng.uniform(0.85, 1.15, size=3).astype(np.float32)
                x = x * np.tile(cast, channels // 3)
            if rng.random() < 0.3:
                x = x + 6.0 * rng.standard_normal(x.shape, dtype=np.float32)
            if rng.random() < 0.3:
                ch, cw = int(rng.integers(8, 24)), int(rng.integers(16, 60))
                y0, x0 = int(rng.integers(0, h - ch)), int(rng.integers(0, w - cw))
                x[y0 : y0 + ch, x0 : x0 + cw] = 128.0

        return np.clip(x, 0, 255).astype(np.uint8), angle
