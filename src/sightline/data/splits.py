"""Split protocols for video-frame datasets.

Consecutive dashcam frames are near-duplicates: same scene, same label, a
fraction of a second apart. A random split therefore places frames from the
same moment on both sides of the train/validation boundary, and validation
error measures memorization distance instead of generalization. Sightline's
honest protocol splits chronologically and additionally discards a firewall
of frames at every boundary so that no validation or test frame is within
``firewall`` frames of anything the model trained on.

``random_split`` is kept, deliberately, as the standard failure mode the
benchmark reproduces and measures. It is not used for any published headline
number other than the leakage measurement itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SplitIndices", "chronological_split", "random_split"]


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        for name in ("train", "val", "test"):
            arr = getattr(self, name)
            if arr.size == 0:
                raise ValueError(f"{name} split is empty")
        overlap = (
            set(self.train.tolist()) & set(self.val.tolist())
            | set(self.train.tolist()) & set(self.test.tolist())
            | set(self.val.tolist()) & set(self.test.tolist())
        )
        if overlap:
            raise ValueError(f"splits overlap on {len(overlap)} indices")

    def min_train_distance(self, split: str) -> int:
        """Smallest frame-index distance from ``split`` to any train frame."""
        other = getattr(self, split)
        train_sorted = np.sort(self.train)
        pos = np.searchsorted(train_sorted, other)
        best = np.full(other.shape, np.iinfo(np.int64).max, dtype=np.int64)
        left_ok = pos > 0
        best[left_ok] = np.abs(other[left_ok] - train_sorted[pos[left_ok] - 1])
        right_ok = pos < train_sorted.size
        best[right_ok] = np.minimum(
            best[right_ok], np.abs(train_sorted[pos[right_ok]] - other[right_ok])
        )
        return int(best.min())


def chronological_split(
    n: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    firewall: int = 250,
) -> SplitIndices:
    """Contiguous train | firewall | val | firewall | test split.

    ``firewall`` frames are dropped on each side of every boundary, so the
    minimum temporal distance between train and evaluation frames is at least
    ``firewall`` frames by construction.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("fractions must be positive and sum below 1")
    if firewall < 0:
        raise ValueError("firewall must be non-negative")
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    if train_end <= 0 or val_end - train_end <= 2 * firewall or n - val_end <= firewall:
        raise ValueError(f"n={n} too small for fractions with firewall={firewall}")
    return SplitIndices(
        train=np.arange(0, train_end - firewall, dtype=np.int64),
        val=np.arange(train_end + firewall, val_end - firewall, dtype=np.int64),
        test=np.arange(val_end + firewall, n, dtype=np.int64),
    )


def random_split(
    n: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 0,
) -> SplitIndices:
    """The standard failure mode: shuffle video frames, then slice.

    Reproduced faithfully (including its statistical sin) so the benchmark
    can measure exactly how much a random split flatters the numbers.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("fractions must be positive and sum below 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n).astype(np.int64)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return SplitIndices(
        train=order[:train_end],
        val=order[train_end:val_end],
        test=order[val_end:],
    )
