"""Structural guards over the repository itself.

These tests police the repo's public surface: what is tracked, what is
ignored, and what words and paths may never appear in tracked text. The
banned-word list is assembled from fragments so this file cannot trip its
own guard.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    ".py", ".md", ".toml", ".json", ".txt", ".html", ".css", ".ts", ".tsx",
    ".js", ".svg", ".yml", ".yaml", ".xml", ".cfg", ".ini", ".ipynb",
}


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return [REPO / line for line in out.stdout.splitlines() if line]


def _tracked_text() -> list[tuple[Path, str]]:
    pairs = []
    for path in _tracked_files():
        if path.suffix.lower() in TEXT_SUFFIXES and path.is_file():
            pairs.append((path, path.read_text(errors="replace")))
    return pairs


def test_no_banned_words_in_tracked_files():
    # assembled from fragments so the guard never matches itself
    fragments = [
        "re" + "build", "re" + "built", "re" + "building",
        "col" + "lege", "course" + "work", "orig" + "inally",
        "cla" + "ude", "anthro" + "pic", "co-auth" + "ored",
        "chat" + "gpt", "open" + "ai",
    ]
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(f) for f in fragments) + r")\b", re.IGNORECASE
    )
    offenders = []
    for path, text in _tracked_text():
        if path.name == Path(__file__).name:
            continue
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(REPO)}:{line}:{match.group(0)}")
    assert offenders == [], f"banned words in tracked files: {offenders}"


def test_no_absolute_user_paths_in_tracked_files():
    pattern = re.compile(r"/Use" + r"rs/[A-Za-z]")
    offenders = [
        str(path.relative_to(REPO))
        for path, text in _tracked_text()
        if path.name != Path(__file__).name and pattern.search(text)
    ]
    assert offenders == [], f"absolute user paths leaked into: {offenders}"


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
