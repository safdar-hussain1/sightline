"""Structural guards over the repository itself.

These tests police the repo's public surface: what is tracked, what is
ignored, and what words and paths may never appear in tracked text. Every
banned word and pattern is assembled from fragments, so this file never
matches its own guards and is scanned like every other file.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: tracked files nobody writes by hand: the npm lock file and the bundles a web build emits
NOT_AUTHORED = ("package-lock.json", "docs/assets/")
#: formats that are never text, fonts among them
BINARY_SUFFIXES = {
    ".bin", ".gif", ".ico", ".jpeg", ".jpg", ".mp4", ".npy", ".npz", ".onnx", ".otf", ".pdf", ".png",
    ".ttf", ".wasm", ".webm", ".webp", ".woff", ".woff2",
}
#: an inline base64 payload (``data:...;base64,...``): its letters are data, not words
BASE64_PAYLOAD = re.compile(r";base64,[A-Za-z0-9+/=\s]+")


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return [REPO / line for line in out.stdout.splitlines() if line]


def _is_text(path: Path) -> bool:
    """A file to read as text: not a binary format, and no NUL byte (git's own test)."""
    return path.suffix.lower() not in BINARY_SUFFIXES and path.is_file() and b"\0" not in path.read_bytes()[:8000]


def _authored_files() -> list[Path]:
    """Every tracked text file written by hand or by this repository's own code."""
    return [path for path in _tracked_files()
            if not any(part in path.relative_to(REPO).as_posix() for part in NOT_AUTHORED) and _is_text(path)]


def _strings(tree, where: str):
    """Every key and string value of a parsed JSON file, with where it sits; a value under a key that
    ends in ``_b64`` is a base64 payload and is left out."""
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield f"{where}.{key}", key
            if not key.endswith("_b64"):
                yield from _strings(value, f"{where}.{key}")
    elif isinstance(tree, list):
        for i, value in enumerate(tree):
            yield from _strings(value, f"{where}[{i}]")
    elif isinstance(tree, str):
        yield where, tree


def _readable_text(path: Path) -> list[tuple[str, str]]:
    """What a reader of ``path`` reads, as ``(where, text)`` pieces: a notebook's cells and the text
    its outputs show (never an image's base64 bytes), a JSON file's keys and strings (never a base64
    payload), any other file whole with inline base64 payloads blanked."""
    rel = path.relative_to(REPO).as_posix()
    text = path.read_text(errors="replace")
    if path.suffix == ".ipynb":
        pieces = []
        for i, cell in enumerate(json.loads(text)["cells"]):
            pieces.append((f"{rel} cell {i}", "".join(cell["source"])))
            for output in cell.get("outputs", []):
                if output.get("output_type") == "stream":
                    pieces.append((f"{rel} cell {i} output", "".join(output["text"])))
                for kind, value in output.get("data", {}).items():
                    if kind.startswith("text/"):
                        pieces.append((f"{rel} cell {i} {kind}", "".join(value)))
        return pieces
    if path.suffix == ".json":
        try:
            return list(_strings(json.loads(text), rel))
        except ValueError:  # JSON with comments, say: read it as it stands
            pass
    return [(rel, BASE64_PAYLOAD.sub(";base64,", text))]


def _line(text: str, start: int) -> int:
    return text.count("\n", 0, start) + 1


# --- words that tell a history, and names that never ship ----------------------------------------

#: words that tell the project's history or name the tools behind it, each its own word in any case
HISTORY_WORDS = [
    "re" + "build", "re" + "built", "re" + "building",
    "col" + "lege", "course" + "work", "orig" + "inally",
    "up" + "grade", "up" + "graded", "up" + "grades", "up" + "grading",
    "v" + "1", "v" + "2",  # version markers
    "cla" + "ude", "anthro" + "pic", "co-auth" + "ored",
    "chat" + "gpt", "open" + "ai",
]
BANNED = re.compile(r"\b(" + "|".join(re.escape(word) for word in HISTORY_WORDS) + r")\b", re.IGNORECASE)
#: the one use of those words that stays: pip's flag in a command line (``pip install --upgrade pip``),
#: and only as a token of its own; a longer token that starts with it is banned like the word itself
PIP_FLAG = re.compile(r"(?<![\w-])--(up" + r"grade)(?![\w-])")


def banned_words(text: str) -> list[tuple[int, str]]:
    """Each banned word in ``text``, with its offset; pip's flag excepted."""
    allowed = {match.span(1) for match in PIP_FLAG.finditer(text)}
    return [(match.start(), match.group(0)) for match in BANNED.finditer(text) if match.span() not in allowed]


@pytest.mark.parametrize("text, found", [
    ("pip install --up" + "grade pip", []),
    ("`--up" + "grade`", []),
    ("an up" + "grade of the page", ["up" + "grade"]),
    ("--up" + "grade-strategy eager", ["up" + "grade"]),
    ("x--up" + "grade", ["up" + "grade"]),
    ("--Up" + "grade", ["Up" + "grade"]),  # the flag is lower case
    ("the V" + "2 model and v" + "1.2", ["V" + "2", "v" + "1"]),
    ("drive2018, mv2, env1 and uv1x", []),
    ("Re" + "built from scratch", ["Re" + "built"]),
])
def test_the_rule_for_banned_words(text, found):
    assert [word for _, word in banned_words(text)] == found


def test_no_banned_words_in_tracked_files():
    offenders = []
    for path in _authored_files():
        for where, text in _readable_text(path):
            offenders += [f"{where}:{_line(text, start)}:{word}" for start, word in banned_words(text)]
    assert offenders == [], f"banned words in tracked files: {offenders}"


def test_the_scan_reads_notebooks_and_json_but_not_their_base64_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    payload = "AAAA+v" + "2/BBBB"  # the bytes of an image: a banned word by chance, between + and /
    notebook = {"cells": [{"cell_type": "code", "source": ["print(1)"], "outputs": [
        {"output_type": "display_data", "data": {"image/png": payload, "text/plain": ["the v" + "2 table"]}}]}]}
    (tmp_path / "n.ipynb").write_text(json.dumps(notebook))
    (tmp_path / "r.json").write_text(json.dumps({"saliency_f32_b64": {"m": payload}, "title": "an up" + "grade"}))
    (tmp_path / "p.html").write_text(f'<img src="data:image/png;base64,{payload}"> the original page')
    found = {p.name: [word for where, text in _readable_text(p) for _, word in banned_words(text)]
             for p in tmp_path.iterdir()}
    assert found == {"n.ipynb": ["v" + "2"], "r.json": ["up" + "grade"], "p.html": []}


def test_no_absolute_user_paths_in_tracked_files():
    pattern = re.compile(r"/Use" + r"rs/[A-Za-z]")
    offenders = [where for path in _authored_files() for where, text in _readable_text(path) if pattern.search(text)]
    assert offenders == [], f"absolute user paths leaked into: {offenders}"


#: the repository's hygiene sweep for build-process traces, the alternatives of its ``git grep -i -P``
#: pattern in order, assembled from fragments so that neither this file nor the sweep finds them here
TRACE_PARTS = (
    r"\.super" + "powers", "super" + "powers", "sub" + "agent", r"\brul" + r"ing\b", r"\bta" + r"sk \d+\b",
    r"\bta" + r"sk brief\b", r"\bthe br" + r"ief\b", r"\bper the pl" + r"an\b", "implementation pl" + "an",
    "design sp" + "ec", "sp" + "ec §", "pl" + "an §", "CLAU" + r"DE\.md", r"\bthe contr" + r"oller\b",
    r"\bthis ta" + r"sk\b",
)
TRACES = re.compile("|".join(TRACE_PARTS), re.IGNORECASE)
#: sha256 of the sweep's pattern as the hygiene check types it, so the two cannot drift apart
TRACE_SHA256 = "ee7c3653d5b4725c2984eef6e458abf0d737c08404fecc044d298fb4c228176a"


def test_the_trace_pattern_is_the_hygiene_checks():
    assert hashlib.sha256(TRACES.pattern.encode()).hexdigest() == TRACE_SHA256


def test_no_build_process_traces_in_tracked_files():
    scanned = [path for path in _tracked_files()
               if not any(part in path.relative_to(REPO).as_posix() for part in (*NOT_AUTHORED, "node_modules/"))
               and _is_text(path)]
    assert Path(__file__).resolve() in scanned  # this file is scanned too: no exclusion hides anything
    offenders = []
    for path in scanned:
        text = path.read_text(errors="replace")
        offenders += [f"{path.relative_to(REPO)}:{_line(text, m.start())}:{m.group(0)}" for m in TRACES.finditer(text)]
    assert offenders == [], f"build-process traces in tracked files: {offenders}"


def test_no_private_notes_folder_or_working_notes_file_is_tracked():
    private_name, private_folder = "CLAU" + "DE.md", ".super" + "powers"
    offenders = [p.relative_to(REPO).as_posix() for p in _tracked_files()
                 if p.name == private_name or private_folder in p.relative_to(REPO).parts]
    assert offenders == [], f"private files tracked: {offenders}"


def test_no_process_documents_tracked():
    banned_names = re.compile(
        r"(spec|plan|handbook|session|ledger|numbers|todo|prompt)[-_a-z0-9]*\.(md|txt|pdf)$",
        re.IGNORECASE,
    )
    allowed = {"README.md"}
    offenders = [
        str(p.relative_to(REPO))
        for p in _tracked_files()
        if p.name not in allowed and banned_names.search(p.name)
    ]
    assert offenders == [], f"process documents tracked: {offenders}"


@pytest.mark.parametrize(
    "path",
    [
        "data/drive2018/data.txt",
        "data/drive2017/0.jpg",
        "runs/cache/drive2018/meta.json",
        "runs/anything/best.pt",
        "weights.npz",
        "model.ckpt.index",
        "web/.public-k2x9q1/data/results.json",
        "reports/.export.json.4242-1a2b3c4d.new",
        "reports/.export.json.4242-1a2b3c4d.previous",
        "demo.mp4",  # what `sightline demo` writes by default
        "demo-c2018.mp4",
        "notebooks/.ipynb_checkpoints/02-results-checkpoint.ipynb",
    ],
)
def test_private_paths_are_ignored(path: str):
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", path], cwd=REPO
    )
    assert result.returncode == 0, f"{path} is NOT git-ignored"


def _matched_by_an_ignore_rule(path: str) -> bool:
    """Whether the ignore rules alone match ``path``: ``--no-index`` asks the rules even about
    a tracked file, which ``git check-ignore`` otherwise leaves out."""
    return subprocess.run(["git", "check-ignore", "--no-index", "--quiet", path], cwd=REPO).returncode == 0


def test_the_demo_clip_videos_are_tracked_and_no_ignore_rule_matches_them():
    tracked = {p.relative_to(REPO).as_posix() for p in _tracked_files()}
    for clip in ("c2017", "c2018"):
        path = f"web/public/data/clips/{clip}/clip.mp4"
        assert path in tracked, f"{path} is not tracked"
        assert not _matched_by_an_ignore_rule(path), f"an ignore rule matches {path}"
    # the rule for the demo command's output is anchored to the repository root
    assert _matched_by_an_ignore_rule("demo.mp4")
    assert not _matched_by_an_ignore_rule("web/public/demo.mp4"), "the demo.mp4 rule is not root-only"
    assert not _matched_by_an_ignore_rule("notebooks/demo.mp4"), "the demo.mp4 rule is not root-only"


def test_the_export_record_is_tracked_and_no_ignore_rule_matches_it():
    assert "reports/export.json" in {p.relative_to(REPO).as_posix() for p in _tracked_files()}
    assert not _matched_by_an_ignore_rule("reports/export.json")


def test_no_dataset_or_weight_files_tracked():
    heavy = [
        str(p.relative_to(REPO))
        for p in _tracked_files()
        if p.suffix in {".pt", ".pth", ".npz", ".ckpt"} or ".ckpt." in p.name
    ]
    assert heavy == [], f"weight/dataset files tracked: {heavy}"


def test_commit_authorship_is_clean():
    log = subprocess.run(
        ["git", "log", "--all", "--format=%an <%ae> %cn <%ce>%n%B"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    if not log.strip():
        pytest.skip("no commits yet")
    banned = re.compile("(cla" + "ude|anthro" + "pic|co-auth" + "ored)", re.IGNORECASE)
    assert not banned.search(log), "attribution or banned text found in git history"
    for line in log.splitlines():
        if line.endswith(">") and "<" in line and "@" in line:
            assert "safdar.eryx@gmail.com" in line, f"unexpected author line: {line}"
