"""Preprocessed-input cache.

Decoding 45k+ JPEGs per epoch is the training bottleneck, so the whole drive
is preprocessed once into a uint8 memmap (N x 66 x 200 x 3 RGB) plus a
float32 angle array. The cache is fingerprinted by the preprocessing
constants, the number of dropout repairs and the sha256 of the label file; a
cache whose fingerprint differs is regenerated automatically.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from sightline.data.labels import DriveLabels
from sightline.data.verify import label_sha256
from sightline.data.preprocess import (
    CROP_BOTTOM,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    preprocess_frame,
)

__all__ = ["DriveCache", "build_cache"]

_CACHE_VERSION = 2


class DriveCache:
    """Read side of the cache: memmapped inputs and in-memory angles."""

    def __init__(self, cache_dir: str | Path):
        cache_dir = Path(cache_dir)
        meta_path = cache_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"no cache at {cache_dir}; run build_cache first")
        self.meta = json.loads(meta_path.read_text())
        n = self.meta["num_frames"]
        self.inputs = np.memmap(
            cache_dir / "inputs.u8",
            dtype=np.uint8,
            mode="r",
            shape=(n, INPUT_HEIGHT, INPUT_WIDTH, 3),
        )
        self.angles_deg = np.load(cache_dir / "angles.npy")
        if len(self.angles_deg) != n:
            raise ValueError("cache angle count does not match input count")

    def __len__(self) -> int:
        return int(self.meta["num_frames"])


def _fingerprint(labels: DriveLabels) -> dict:
    label_file = labels.root / "data.txt"
    digest = label_sha256(label_file) if label_file.is_file() else None
    return {
        "cache_version": _CACHE_VERSION,
        "input": [INPUT_HEIGHT, INPUT_WIDTH, CROP_BOTTOM],
        "num_frames": len(labels),
        "label_sha256": digest,
        "repaired": int(labels.repaired_indices.size),
    }


def build_cache(
    labels: DriveLabels,
    cache_dir: str | Path,
    workers: int = 8,
    force: bool = False,
) -> DriveCache:
    """Build (or reuse) the preprocessed cache for one drive."""
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    fingerprint = _fingerprint(labels)

    if not force and meta_path.is_file():
        existing = json.loads(meta_path.read_text())
        if {k: existing.get(k) for k in fingerprint} == fingerprint:
            return DriveCache(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    n = len(labels)
    mm = np.memmap(
        cache_dir / "inputs.u8",
        dtype=np.uint8,
        mode="w+",
        shape=(n, INPUT_HEIGHT, INPUT_WIDTH, 3),
    )

    def _one(i: int) -> None:
        frame = cv2.imread(str(labels.frame_path(i)), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"unreadable frame: {labels.frame_path(i)}")
        mm[i] = preprocess_frame(frame)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(tqdm(pool.map(_one, range(n)), total=n, desc="cache", unit="frame"))
    mm.flush()

    np.save(cache_dir / "angles.npy", labels.angles_deg.astype(np.float32))
    if labels.timestamps_s is not None:
        np.save(cache_dir / "timestamps.npy", labels.timestamps_s)
    meta_path.write_text(json.dumps(fingerprint, indent=2))
    return DriveCache(cache_dir)
