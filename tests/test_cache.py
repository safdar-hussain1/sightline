"""The drive cache: reuse, and regeneration that never disturbs a reader.

Training keeps ``inputs.u8`` memory-mapped for hours, so regenerating a cache must leave the
file a reader opened whole and must never leave a ``meta.json`` beside files it does not
describe.
"""
import hashlib
import io
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

import sightline.data.cache
from sightline.data.cache import DriveCache, build_cache
from sightline.data.labels import load_labels
from sightline.data.preprocess import preprocess_frame
from conftest import make_drive

DATA_FILES = {"inputs.u8", "angles.npy", "timestamps.npy"}


def frames_of(labels) -> np.ndarray:
    """Every frame's model input, decoded and preprocessed one at a time."""
    return np.stack([preprocess_frame(cv2.imread(str(labels.frame_path(i)))) for i in range(len(labels))])


def test_unchanged_drive_reuses_the_cache(tmp_path):
    drive = make_drive(tmp_path, "drivetest", n=10)
    cache_dir = tmp_path / "cache"
    build_cache(load_labels(drive), cache_dir)
    # a byte that differs from what preprocessing wrote; regenerating would overwrite it
    with open(cache_dir / "inputs.u8", "r+b") as f:
        sentinel = (f.read(1)[0] + 128) % 256
        f.seek(0)
        f.write(bytes([sentinel]))
    cache = build_cache(load_labels(drive), cache_dir)
    assert (cache_dir / "inputs.u8").read_bytes()[0] == sentinel
    assert cache.inputs[0, 0, 0, 0] == sentinel


def test_same_size_label_edit_regenerates_the_cache(tmp_path):
    drive = make_drive(tmp_path, "drivetest", n=10, timestamps=False)
    cache_dir = tmp_path / "cache"
    first = float(build_cache(load_labels(drive), cache_dir).angles_deg[0])
    text = (drive / "data.txt").read_text()
    first_line, rest = text.split("\n", 1)
    name, value = first_line.split()
    edited = f"{name} {float(value) + 1.0:.6f}\n{rest}"
    assert len(edited) == len(text)
    (drive / "data.txt").write_text(edited)
    second = float(build_cache(load_labels(drive), cache_dir).angles_deg[0])
    assert second != first and np.isclose(second, first + 1.0, atol=1e-3)


# --- regenerating a cache in use ------------------------------------------------------------

def test_a_reader_keeps_the_frames_it_opened_while_the_cache_is_regenerated(tmp_path):
    cache_dir = tmp_path / "cache"
    first = load_labels(make_drive(tmp_path / "first", "drivetest", n=30))
    reader = build_cache(first, cache_dir)
    opened = os.stat(cache_dir / "inputs.u8")
    # other labels and other frames, and fewer of them: the new inputs file is shorter
    second = load_labels(make_drive(tmp_path / "second", "drivetest", n=20, scenery=True))
    assert build_cache(second, cache_dir).meta != reader.meta
    # rewritten in place, the file would now end inside the reader's map, and reading past
    # its end kills the reading process (SIGBUS); checked first, so that fails, not crashes
    assert os.stat(cache_dir / "inputs.u8").st_ino != opened.st_ino, "inputs.u8 was rewritten in place"
    np.testing.assert_array_equal(reader.inputs, frames_of(first))  # every frame, as first built
    fresh = DriveCache(cache_dir)
    assert len(fresh) == 20
    np.testing.assert_array_equal(fresh.inputs, frames_of(second))
    np.testing.assert_array_equal(fresh.angles_deg, second.angles_deg.astype(np.float32))


def test_regenerating_drops_the_old_meta_before_it_swaps_a_file_and_writes_the_new_one_last(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    build_cache(load_labels(make_drive(tmp_path / "a", "drivetest", n=10)), cache_dir)
    swaps = []
    replace = os.replace

    def watched(src, dst):
        swaps.append((Path(dst).name, (cache_dir / "meta.json").exists()))
        replace(src, dst)

    monkeypatch.setattr(os, "replace", watched)
    build_cache(load_labels(make_drive(tmp_path / "b", "drivetest", n=12)), cache_dir)
    # every file arrives whole under its own name, and while any of them is swapped an
    # opener finds no meta.json, never one that describes the files beside it
    assert sorted(name for name, _ in swaps[:-1]) == sorted(DATA_FILES)
    assert swaps[-1][0] == "meta.json"
    assert not any(meta_present for _, meta_present in swaps)
    assert len(DriveCache(cache_dir)) == 12


def test_a_build_that_fails_part_way_leaves_no_cache_to_open(tmp_path):
    cache_dir = tmp_path / "cache"
    build_cache(load_labels(make_drive(tmp_path / "old", "drivetest", n=12)), cache_dir)
    new = load_labels(make_drive(tmp_path / "new", "drivetest", n=16, scenery=True))
    calls = []

    def fails_on_the_fifth_frame(frame):
        calls.append(1)
        if len(calls) == 5:
            raise RuntimeError("cannot preprocess frame 4")
        return preprocess_frame(frame)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sightline.data.cache, "preprocess_frame", fails_on_the_fifth_frame)
        with pytest.raises(RuntimeError, match="frame 4"):
            build_cache(new, cache_dir, workers=1)
    # no temporary file is left, and without meta.json nothing opens what remains
    assert {p.name for p in cache_dir.iterdir()} <= DATA_FILES
    with pytest.raises(FileNotFoundError, match="no cache"):
        DriveCache(cache_dir)

    cache = build_cache(new, cache_dir)
    assert len(cache) == 16
    np.testing.assert_array_equal(cache.inputs, frames_of(new))


def test_regenerating_from_labels_without_timestamps_keeps_no_earlier_timestamps(tmp_path):
    cache_dir = tmp_path / "cache"
    build_cache(load_labels(make_drive(tmp_path / "a", "drivetest", n=10)), cache_dir)
    assert (cache_dir / "timestamps.npy").is_file()
    build_cache(load_labels(make_drive(tmp_path / "b", "drivetest", n=10, timestamps=False)), cache_dir)
    assert sorted(p.name for p in cache_dir.iterdir()) == ["angles.npy", "inputs.u8", "meta.json"]


# --- the bytes on disk ----------------------------------------------------------------------

def block_colours(i: int, rows: np.ndarray, cols: np.ndarray, channels: np.ndarray) -> np.ndarray:
    """Frame ``i``'s colour of each (row block, column block, channel)."""
    return ((37 * i + 53 * rows[:, None, None] + 71 * cols[None, :, None] + 101 * channels[None, None, :])
            % 256).astype(np.uint8)


def exact_drive(root: Path, n: int) -> Path:
    """A timestamped drive whose cached bytes are the same on every platform: lossless PNG
    frames whose bottom 150 rows are blocks of 25 rows x 91 columns, one colour each, which
    the resize to 66 x 200 maps exactly onto blocks of 11 x 40 (no output pixel straddles two
    blocks, so no rounding can differ). The rows above, cropped away, are noise."""
    drive = root / "exactdrive"
    drive.mkdir(parents=True)
    lines = []
    for i in range(n):
        frame = np.random.default_rng(i).integers(0, 256, (256, 455, 3), dtype=np.uint8)
        frame[-150:] = block_colours(i, np.arange(150) // 25, np.arange(455) // 91, np.arange(3))
        assert cv2.imwrite(str(drive / f"{i}.png"), frame)
        lines.append(f"{i}.png {2.5 * (i - 5):.6f},2018-07-01 17:00:{i:02d}:{57 * i % 1000}")
    (drive / "data.txt").write_text("\n".join(lines) + "\n")
    return drive


#: sha256 of the files a build of ``exact_drive(n=12)`` writes: a cache already on disk stays
#: valid only while every build writes the same bytes, however it writes them
EXACT_SHA256 = {
    "inputs.u8": "e63aa178a39e08eaecee1d3951b2482665a4eb91d179560c5675e9058d9e4a59",
    "angles.npy": "f5552617b763b259907d57855c7df9232ac19d1825cf1f9f7d3a7a0292645e6d",
    "meta.json": "0cdac0ee0971687821f4395500a0d387a33f14ddc06707af05b9c19335c7150b",
}


def test_the_cache_files_keep_their_exact_bytes(tmp_path):
    labels = load_labels(exact_drive(tmp_path, n=12))
    cache_dir = tmp_path / "cache"
    cache = build_cache(labels, cache_dir)
    rows, cols, rgb = np.arange(66) // 11, np.arange(200) // 40, np.arange(3)[::-1]
    np.testing.assert_array_equal(cache.inputs, np.stack([block_colours(i, rows, cols, rgb) for i in range(12)]))
    assert sorted(p.name for p in cache_dir.iterdir()) == ["angles.npy", "inputs.u8", "meta.json", "timestamps.npy"]
    for name, digest in EXACT_SHA256.items():
        assert hashlib.sha256((cache_dir / name).read_bytes()).hexdigest() == digest, name
    # epoch seconds follow the local time zone, so these bytes are compared, not pinned
    saved = io.BytesIO()
    np.save(saved, labels.timestamps_s)
    assert (cache_dir / "timestamps.npy").read_bytes() == saved.getvalue()
