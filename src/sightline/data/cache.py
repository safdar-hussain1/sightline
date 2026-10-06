"""Preprocessed-input cache.

Decoding 45k+ JPEGs per epoch is the training bottleneck, so the whole drive
is preprocessed once into a uint8 memmap (N x 66 x 200 x 3 RGB) plus a
float32 angle array. The cache is fingerprinted by the preprocessing
constants, the number of dropout repairs and the sha256 of the label file; a
cache whose fingerprint differs is regenerated automatically.

A cache is never rewritten in place. Training processes keep ``inputs.u8``
memory-mapped for hours, and a mapped file that is rewritten under them
hands them other frames, or kills them (SIGBUS) once they read past a
shortened end. So a build writes each file under a temporary name in the
cache folder and flushes it, deletes the old ``meta.json``, swaps each file
into place with ``os.replace`` and writes ``meta.json`` last, the same way.
A process that mapped the old ``inputs.u8`` keeps the old file, whole, until
it lets go of it, and a process opening the cache finds either no
``meta.json`` or one that describes the files beside it, even when a build
is killed half-way: a cache never looks complete unless it is. A build that
fails with an error deletes its temporary files and leaves no ``meta.json``,
so the next build starts again.

``open_cache`` is the read-only way in, for commands that publish what they compute from
a cache: it opens a cache only when its fingerprint is the one the drive's labels give
now, and never builds one.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
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
from sightline.paths import MissingDataError

__all__ = ["DriveCache", "build_cache", "open_cache"]

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


def _write_inputs(labels: DriveLabels, path: Path, workers: int) -> None:
    """Preprocess every frame into a new memmap at ``path``, flushed to disk and unmapped."""
    n = len(labels)
    mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=(n, INPUT_HEIGHT, INPUT_WIDTH, 3))

    def _one(i: int) -> None:
        frame = cv2.imread(str(labels.frame_path(i)), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"unreadable frame: {labels.frame_path(i)}")
        mm[i] = preprocess_frame(frame)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(tqdm(pool.map(_one, range(n)), total=n, desc="cache", unit="frame"))
        mm.flush()
    finally:
        del mm  # the last reference: unmaps the file, also when a frame failed


def _write_npy(path: Path, array: np.ndarray) -> None:
    with open(path, "wb") as f:  # a file object, so np.save adds no ".npy" to the temporary name
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())


def _write_text(path: Path, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def open_cache(labels: DriveLabels, cache_dir: str | Path) -> DriveCache:
    """The cache at ``cache_dir``, opened only if it was built from ``labels`` as they are now.

    Raises MissingDataError, naming the command that builds the cache, when there is no
    cache or its fingerprint differs from the one ``labels`` give (another label file,
    other repairs or other preprocessing). Nothing is ever written.
    """
    cache_dir = Path(cache_dir)
    drive = labels.root.name
    if not (cache_dir / "meta.json").is_file():
        raise MissingDataError(f"no cache for drive {drive!r} at {cache_dir}; "
                               f"run `sightline cache --drive {drive}` first")
    cache = DriveCache(cache_dir)
    fingerprint = _fingerprint(labels)
    if {k: cache.meta.get(k) for k in fingerprint} != fingerprint:
        raise MissingDataError(f"the cache at {cache_dir} was built from other labels or settings than "
                               f"drive {drive!r} has now; run `sightline cache --drive {drive}` to build it again")
    return cache


def build_cache(
    labels: DriveLabels,
    cache_dir: str | Path,
    workers: int = 8,
    force: bool = False,
) -> DriveCache:
    """Build (or reuse) the preprocessed cache for one drive, swapping each file into place."""
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    fingerprint = _fingerprint(labels)

    if not force and meta_path.is_file():
        existing = json.loads(meta_path.read_text())
        if {k: existing.get(k) for k in fingerprint} == fingerprint:
            return DriveCache(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    data_files = ["inputs.u8", "angles.npy"] + (["timestamps.npy"] if labels.timestamps_s is not None else [])
    suffix = f".tmp-{os.getpid()}-{secrets.token_hex(4)}"  # unique to this build
    temporary = {name: cache_dir / (name + suffix) for name in [*data_files, "meta.json"]}
    try:
        _write_inputs(labels, temporary["inputs.u8"], workers)
        _write_npy(temporary["angles.npy"], labels.angles_deg.astype(np.float32))
        if labels.timestamps_s is not None:
            _write_npy(temporary["timestamps.npy"], labels.timestamps_s)
        # from here until the new meta.json is in place an opener finds no cache, never a
        # meta.json written for other files than the ones beside it
        meta_path.unlink(missing_ok=True)
        for name in data_files:
            os.replace(temporary[name], cache_dir / name)
        if labels.timestamps_s is None:
            (cache_dir / "timestamps.npy").unlink(missing_ok=True)  # an earlier build's
        _write_text(temporary["meta.json"], json.dumps(fingerprint, indent=2))
        os.replace(temporary["meta.json"], meta_path)
    except BaseException:
        # whatever failed, nothing that looks like a cache remains; the next build starts again
        for path in [*temporary.values(), meta_path]:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
        raise
    return DriveCache(cache_dir)
