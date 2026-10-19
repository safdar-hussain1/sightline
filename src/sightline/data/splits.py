"""Split protocols for video-frame datasets.

Consecutive dashcam frames are near-duplicates: same scene, same label, a
fraction of a second apart. A split that puts neighbouring frames on both
sides of the train/evaluation boundary measures memorization distance instead
of generalization. Three protocols are provided:

``blocked``
    Interleaved contiguous blocks (train, validation and test stretches
    repeating across the whole drive) with a firewall of dropped frames at
    every train/evaluation boundary. Both sides see the same mix of roads and
    every evaluation frame is at least ``2 * firewall + 1`` frames from a
    training frame. This is the headline honest protocol.
``forward``
    Chronological train | firewall | validation | firewall | test, i.e.
    training on the earlier part of the drive and testing on the later part
    (see ``chronological_split``). The same distance guarantee as ``blocked``.
``leaky``
    Shuffle the frames, then slice. This is the standard failure mode,
    reproduced on purpose so its optimism can be measured on exactly the same
    frames the honest protocol scores (see ``paired_frames``). It is never
    used for a headline number other than the leakage measurement itself.

The rules every split is made with unless a caller names others are the module
constants ``FIREWALL``, ``BLOCK``, ``BLOCK_PATTERN``, ``TRAIN_FRAC`` and
``VAL_FRAC``; the functions below and the commands take their defaults from
them. The site data states the split rules: the pattern and the fractions are
these constants, which no command changes, and the block and the firewall are
the values the runs recorded, which the commands can change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BLOCK",
    "BLOCK_PATTERN",
    "FIREWALL",
    "PROTOCOLS",
    "PUBLIC_NAMES",
    "SplitIndices",
    "TRAIN_FRAC",
    "VAL_FRAC",
    "blocked_split",
    "chronological_split",
    "frames_sha256",
    "make_split",
    "paired_frames",
    "random_split",
]

PROTOCOLS = ("leaky", "blocked", "forward")
PUBLIC_NAMES = {"leaky": "shuffled", "blocked": "blocked", "forward": "later in the drive"}

#: frames dropped on each side of every boundary between a training stretch and an
#: evaluation stretch (``blocked`` and ``forward``)
FIREWALL = 250
#: frames per block of the ``blocked`` split
BLOCK = 1500
#: the ``blocked`` split's repeating block pattern: T train, V validation, E test
BLOCK_PATTERN = "TTVTTET"
#: the share of a drive the ``leaky`` and ``forward`` splits train on, and the share they
#: validate on; the rest of the drive tests
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


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
            np.intersect1d(self.train, self.val).size
            + np.intersect1d(self.train, self.test).size
            + np.intersect1d(self.val, self.test).size
        )
        if overlap:
            raise ValueError(f"splits overlap on {overlap} indices")

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
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    firewall: int = FIREWALL,
) -> SplitIndices:
    """Contiguous train | firewall | val | firewall | test split.

    ``firewall`` frames are dropped on each side of every boundary, so every
    validation and test frame is at least ``2 * firewall + 1`` frames from a
    training frame by construction.
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
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
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


def blocked_split(n: int, block: int = BLOCK, firewall: int = FIREWALL,
                  pattern: str = BLOCK_PATTERN) -> SplitIndices:
    """Interleaved contiguous blocks with a firewall at every train/eval boundary.

    The drive is cut into ``block``-frame blocks labelled by repeating
    ``pattern`` (T train, V validation, E test). ``firewall`` frames are
    dropped on BOTH sides of every boundary between a training block and an
    evaluation block, so every evaluation frame is at least
    ``2 * firewall + 1`` frames from a training frame — the same guarantee as
    the forward split — while both sides of the split see the same mix of
    roads. Validation and test blocks are not firewalled from each other; the
    default pattern never places them adjacent.
    """
    if set(pattern) - set("TVE") or not set("TVE") <= set(pattern):
        raise ValueError(f"pattern must use T, V and E, each at least once: {pattern!r}")
    if firewall < 0:
        raise ValueError("firewall must be non-negative")
    if block <= 2 * firewall:
        raise ValueError(f"block ({block}) must exceed twice the firewall ({firewall})")
    labels = np.array(list(pattern))[(np.arange(n) // block) % len(pattern)]
    keep = np.ones(n, dtype=bool)
    for edge in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
        if "T" in (labels[edge - 1], labels[edge]):
            keep[max(0, edge - firewall):min(n, edge + firewall)] = False
    idx = np.arange(n, dtype=np.int64)
    parts = {name: idx[keep & (labels == code)]
             for name, code in (("train", "T"), ("val", "V"), ("test", "E"))}
    empty = [name for name, part in parts.items() if part.size == 0]
    if empty:
        raise ValueError(
            f"n={n} too small for the blocked split with block={block}, firewall={firewall}: "
            f"no {' or '.join(empty)} frames"
        )
    return SplitIndices(**parts)


def make_split(protocol: str, n: int, seed: int = 0, firewall: int = FIREWALL,
               block: int = BLOCK) -> SplitIndices:
    """Build the split for one of ``PROTOCOLS`` over ``n`` frames."""
    if protocol == "leaky":
        return random_split(n, seed=seed)
    if protocol == "blocked":
        return blocked_split(n, block=block, firewall=firewall)
    if protocol == "forward":
        return chronological_split(n, firewall=firewall)
    raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")


def paired_frames(blocked: SplitIndices, leaky: SplitIndices) -> np.ndarray:
    """Frames held out by BOTH protocols: the paired leakage measurement's set."""
    return np.intersect1d(blocked.test, leaky.test).astype(np.int64)


def frames_sha256(indices: np.ndarray) -> str:
    """sha256 of frame indices as little-endian int64: equal digests mean identical frames."""
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
