"""Is the dataset on disk the one every published number was measured on?"""
from __future__ import annotations

import hashlib
from pathlib import Path

KNOWN_DRIVES = {
    "drive2018": {"frames": 63825,
                  "label_sha256": "f7804a17185534831be517dccad1f24a1fbcf34ff5d961f03f83c5b9b8e57434"},
    "drive2017": {"frames": 45406,
                  "label_sha256": "235c70454d460b86edf1ddb4afba4354529687ad9ca08cb5d7ddfd8f8d4a8bb7"},
}


def verify_drive(path: Path, name: str) -> dict:
    label_file = Path(path) / "data.txt"
    digest = hashlib.sha256(label_file.read_bytes()).hexdigest()
    rows = [ln.split()[0] for ln in label_file.read_text().splitlines() if ln.strip()]
    missing = [r for r in rows if not (Path(path) / r).is_file()]
    expected = KNOWN_DRIVES.get(name)
    problems = []
    if expected is None:
        problems.append(f"{name!r} is not a known drive; nothing to compare against")
    else:
        if digest != expected["label_sha256"]:
            problems.append("label file differs from the published version (sha256 mismatch)")
        if len(rows) != expected["frames"]:
            problems.append(f"{len(rows)} labelled frames, expected {expected['frames']}")
    if missing:
        problems.append(f"{len(missing)} labelled frames have no image (first: {missing[0]})")
    return {"name": name, "frames": len(rows), "label_sha256": digest,
            "ok": not problems, "problems": problems}
