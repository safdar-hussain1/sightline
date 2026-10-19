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
from pathlib import Path, PurePosixPath

import pytest

REPO = Path(__file__).resolve().parents[1]

#: formats that are never text, fonts among them
BINARY_SUFFIXES = {
    ".bin", ".gif", ".ico", ".jpeg", ".jpg", ".mp4", ".npy", ".npz", ".onnx", ".otf", ".pdf", ".png",
    ".ttf", ".wasm", ".webm", ".webp", ".woff", ".woff2",
}
#: tracked files without a binary suffix that may hold a NUL byte, each with the reason it may; a
#: NUL byte in any other of them fails the guards, as the text scans could not read the file
NUL_ALLOWED: dict[str, str] = {}
#: an inline base64 payload (``data:...;base64,...``), up to the first character base64 never uses
BASE64_PAYLOAD = re.compile(r";base64,[A-Za-z0-9+/]+=*")
#: the keys of a JSON tree that hold base64 payloads: a name that ends in ``_b64``, and a notebook
#: output's binary image types
PAYLOAD_KEY = re.compile(r".*_b64|image/(?:png|jpeg|gif|webp)")


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return [REPO / line for line in out.stdout.splitlines() if line]


def not_authored(rel: str) -> bool:
    """Whether the tracked file at ``rel`` (from the repository root) is one nobody writes by hand,
    matched exactly: an npm lock file, a bundle the web build emits into ``docs/assets/``, or a file
    of an installed node module."""
    path = PurePosixPath(rel)
    return path.name == "package-lock.json" or rel.startswith("docs/assets/") or "node_modules" in path.parts


def _scanned_files() -> list[Path]:
    """Every tracked file the text scans read: all but the ones nobody writes by hand and the
    binary formats."""
    return [path for path in _tracked_files()
            if path.is_file() and not not_authored(path.relative_to(REPO).as_posix())
            and path.suffix.lower() not in BINARY_SUFFIXES]


def is_base64(text: str) -> bool:
    """Whether ``text`` is base64 and nothing else (line breaks aside): its letters, then at most two
    ``=``, in a whole number of four-character groups."""
    packed = text.replace("\n", "")
    return len(packed) % 4 == 0 and re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", packed) is not None


def _strings(tree, where: str, payload: bool = False):
    """Every key and every string of a parsed JSON tree, with where it sits. Under a key that names a
    base64 payload (``PAYLOAD_KEY``) a string that is base64 and nothing else is left out; any other
    string there is read, and the walk goes into every mapping and list whatever its key."""
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield f"{where}.{key}", key
            yield from _strings(value, f"{where}.{key}", payload or PAYLOAD_KEY.fullmatch(key) is not None)
    elif isinstance(tree, list):
        for i, value in enumerate(tree):
            yield from _strings(value, f"{where}[{i}]", payload)
    elif isinstance(tree, str) and not (payload and is_base64(tree)):
        yield where, tree


def _readable_text(path: Path) -> list[tuple[str, str]]:
    """What a reader of ``path`` can read, as ``(where, text)`` pieces: every key and string of a
    notebook or a JSON file (a notebook's metadata, every output and every traceback included), but
    no base64 payload; any other file whole, its inline base64 payloads blanked."""
    rel = path.relative_to(REPO).as_posix()
    text = path.read_text(errors="replace")
    if path.suffix in (".ipynb", ".json"):
        try:
            return list(_strings(json.loads(text), rel))
        except ValueError:  # not JSON as it stands (JSON with comments, say): read as text below
            pass
    return [(rel, BASE64_PAYLOAD.sub(";base64,", text))]


def _line(text: str, start: int) -> int:
    return text.count("\n", 0, start) + 1


#: files every scan must read, so that a scan of nothing cannot pass
SCANNED_FOR_SURE = {"LICENSE", ".gitignore", "pyproject.toml", "src/sightline/cli.py", "tests/test_guards.py",
                    "notebooks/02-results.ipynb", "reports/summary.json", "web/public/data/results.json"}


def test_the_scans_read_every_file_but_the_generated_ones_and_the_binary_formats():
    assert SCANNED_FOR_SURE <= {path.relative_to(REPO).as_posix() for path in _scanned_files()}


@pytest.mark.parametrize("rel, skipped", [
    ("package-lock.json", True), ("web/package-lock.json", True),
    ("notes/package-lock.json.md", False), ("web/package-lock.json.orig", False),
    ("docs/assets/index-3f2a.js", True), ("web/docs/assets/index.js", False), ("docs/assets-notes.md", False),
    ("web/node_modules/vite/index.js", True), ("web/my_node_modules.md", False), ("notes/node_modules.md", False),
])
def test_only_the_files_nobody_writes_are_left_out_and_exactly(rel, skipped):
    assert not_authored(rel) is skipped


def test_no_tracked_file_holds_a_nul_byte_unless_its_format_is_binary():
    allowed = {rel for rel, reason in NUL_ALLOWED.items() if reason.strip()}
    holding = [path.relative_to(REPO).as_posix() for path in _scanned_files() if b"\0" in path.read_bytes()]
    assert [rel for rel in holding if rel not in allowed] == [], "files the text scans cannot read"


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
    files = _scanned_files()
    assert SCANNED_FOR_SURE <= {path.relative_to(REPO).as_posix() for path in files}
    offenders = []
    for path in files:
        for where, text in _readable_text(path):
            offenders += [f"{where}:{_line(text, start)}:{word}" for start, word in banned_words(text)]
    assert offenders == [], f"banned words in tracked files: {offenders}"


def _found_in(path: Path) -> list[str]:
    return [word for _, text in _readable_text(path) for _, word in banned_words(text)]


def test_the_scan_reads_everything_a_notebook_holds_but_its_images(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    payload = "AAAA+v" + "2/BBBB"  # an image's bytes: a banned word by chance, between + and /
    notebook = {"metadata": {"title": "the v" + "1 notes"}, "cells": [{
        "cell_type": "code", "metadata": {"tags": ["re" + "build"]}, "source": ["print(1)"], "outputs": [
            {"output_type": "display_data", "data": {"image/png": payload, "text/plain": ["the v" + "2 table"],
                                                     "application/json": {"note": "an up" + "grade"}}},
            {"output_type": "error", "ename": "ValueError", "evalue": "orig" + "inally",
             "traceback": ["line one", "col" + "lege"]}]}]}
    (tmp_path / "n.ipynb").write_text(json.dumps(notebook))
    assert sorted(_found_in(tmp_path / "n.ipynb")) == sorted(
        ["v" + "1", "re" + "build", "v" + "2", "up" + "grade", "orig" + "inally", "col" + "lege"])


def test_the_scan_leaves_out_only_strings_that_are_base64_under_a_payload_key(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    payload = "AAAA+v" + "2/BBBB"
    cases = {
        "skipped.json": {"saliency_f32_b64": {"m": payload}, "frames_b64": payload},
        "not-base64.json": {"saliency_f32_b64": "an up" + "grade, not a payload"},
        "nested.json": {"saliency_f32_b64": {"note": ["an up" + "grade"]}},
        "plain-key.json": {"title": payload},
    }
    for name, tree in cases.items():
        (tmp_path / name).write_text(json.dumps(tree))
    assert {name: _found_in(tmp_path / name) for name in cases} == {
        "skipped.json": [], "not-base64.json": ["up" + "grade"], "nested.json": ["up" + "grade"],
        "plain-key.json": ["v" + "2"]}
    assert is_base64(payload) and is_base64("QUJD\nREVG") and not is_base64("up" + "grade") \
        and not is_base64("QUJDRA=") and not is_base64("QU JD")


def test_the_scan_blanks_an_inline_payload_and_reads_what_follows_it(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    payload = "AAAA+v" + "2/BBBB"
    (tmp_path / "p.html").write_text(f'<img src="data:image/png;base64,{payload}">an up' + "grade")
    (tmp_path / "q.md").write_text(f"data:text/plain;base64,{payload}\nthe v" + "1 page")
    (tmp_path / "r.css").write_text(f"a {{ background: url(data:image/png;base64,{payload}) }} /* the original */")
    assert {p.name: _found_in(p) for p in sorted(tmp_path.iterdir())} == \
        {"p.html": ["up" + "grade"], "q.md": ["v" + "1"], "r.css": []}


def test_no_absolute_user_paths_in_tracked_files():
    pattern = re.compile(r"/Use" + r"rs/[A-Za-z]")
    files = _scanned_files()
    assert SCANNED_FOR_SURE <= {path.relative_to(REPO).as_posix() for path in files}
    offenders = [where for path in files for where, text in _readable_text(path) if pattern.search(text)]
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
    scanned = _scanned_files()
    # this file is scanned too, so no exclusion hides anything, and so is every file a scan must read
    assert SCANNED_FOR_SURE <= {path.relative_to(REPO).as_posix() for path in scanned}
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
