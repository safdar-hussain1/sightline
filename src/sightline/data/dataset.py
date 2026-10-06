"""Torch dataset over the preprocessed cache, with training augmentation.

Augmentations are steering-aware: a horizontal flip mirrors the road, so the
label must be negated. Photometric jitter (brightness/contrast, a soft random
shadow band) leaves the label untouched. All augmentation happens on uint8
numpy arrays before tensor conversion; evaluation mode returns frames
untouched.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from sightline.data.cache import DriveCache

__all__ = ["SteeringDataset"]


class SteeringDataset(Dataset):
    def __init__(
        self,
        cache: DriveCache,
        indices: np.ndarray,
        augment: bool = False,
        seed: int = 0,
    ):
        if indices.min() < 0 or indices.max() >= len(cache):
            raise IndexError("split indices out of cache range")
        self.cache = cache
        self.indices = indices.astype(np.int64)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.indices[i]
        frame = np.asarray(self.cache.inputs[idx])  # uint8 HWC RGB copy
        angle = float(self.cache.angles_deg[idx])

        if self.augment:
            frame, angle = self._augment(frame, angle)

        return (
            torch.from_numpy(np.ascontiguousarray(frame)),
            torch.tensor([angle], dtype=torch.float32),
        )

    def _augment(self, frame: np.ndarray, angle: float) -> tuple[np.ndarray, float]:
        rng = self.rng
        if rng.random() < 0.5:
            frame = frame[:, ::-1]
            angle = -angle
        # brightness/contrast jitter in float, clipped back to uint8
        if rng.random() < 0.8:
            gain = rng.uniform(0.7, 1.3)
            bias = rng.uniform(-25, 25)
            frame = np.clip(frame.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        # soft vertical shadow band, as cast by roadside structures
        if rng.random() < 0.3:
            w = frame.shape[1]
            x0 = int(rng.uniform(0, w * 0.75))
            x1 = int(x0 + rng.uniform(w * 0.15, w * 0.5))
            shade = rng.uniform(0.5, 0.8)
            band = frame[:, x0:x1].astype(np.float32) * shade
            frame = frame.copy()
            frame[:, x0:x1] = band.astype(np.uint8)
        return frame, angle
