import numpy as np

from sightline.data.cache import build_cache
from sightline.data.labels import load_labels
from conftest import make_drive


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
