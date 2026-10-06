"""No test can reach the repository's own data, caches or reports.

A relative default (``runs``, ``data``, ``reports``) resolves against the working
directory, so every test, and every module- or session-scoped fixture, runs from a tmp
folder of its own (conftest.py). Behind that, a tripwire records the repository's drive
caches and label files when the session starts and fails the test after which any of them
changed. The tripwire is checked here on stand-in repositories.
"""
import os
import re
import shutil
from pathlib import Path

import pytest

import conftest
from sightline.paths import Workspace

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def module_cwd() -> Path:
    """The working directory of a module-scoped fixture, set up before any test's own folder."""
    return Path.cwd().resolve()


def test_every_test_runs_in_its_own_tmp_folder(tmp_path):
    cwd = Path.cwd().resolve()
    assert cwd.is_relative_to(tmp_path.resolve())
    assert not cwd.is_relative_to(REPO)


def test_module_fixtures_run_outside_the_repository_too(module_cwd):
    assert not module_cwd.is_relative_to(REPO)


def test_a_relative_workspace_default_resolves_under_the_tests_folder(tmp_path):
    ws = Workspace()
    for path in (ws.data_dir, ws.runs_dir, ws.cache_dir("drive2018")):
        assert path.resolve().is_relative_to(tmp_path.resolve()), path


def test_the_tripwire_watches_this_repository_after_every_test(request, workspace_before):
    assert "real_workspace_untouched" in request.fixturenames
    assert (workspace_before is None) == (not (REPO / "runs" / "cache").is_dir())
    assert workspace_before == conftest.workspace_state(REPO)


# --- the tripwire on stand-in repositories --------------------------------------------------

def stand_in(root: Path) -> Path:
    """A repository with two drives' label files and caches, and one run folder."""
    for drive in ("drive2018", "drive2017"):
        (root / "data" / drive).mkdir(parents=True)
        (root / "data" / drive / "data.txt").write_text("0.jpg 1.5\n")
        (root / "runs" / "cache" / drive).mkdir(parents=True)
        for name in ("inputs.u8", "angles.npy", "meta.json"):
            (root / "runs" / "cache" / drive / name).write_bytes(f"{drive} {name}".encode())
    (root / "runs" / "drive2018_pilotnet_blocked_s0").mkdir()
    return root


def test_an_untouched_repository_passes(tmp_path):
    root = stand_in(tmp_path / "repo")
    before = conftest.workspace_state(root)
    assert sorted(before) == ["data/drive2017/data.txt", "data/drive2018/data.txt",
                              *(f"runs/cache/{d}/{f}" for d in ("drive2017", "drive2018")
                                for f in ("angles.npy", "inputs.u8", "meta.json"))]
    (root / "runs" / "cache" / "drive2018" / "inputs.u8").read_bytes()  # reading is no change
    # run folders, logs and reports are written by training and evaluation: not watched
    (root / "runs" / "drive2018_pilotnet_blocked_s0" / "history.json").write_text("{}")
    (root / "runs" / "logs").mkdir()
    (root / "reports").mkdir()
    (root / "reports" / "summary.json").write_text("{}")
    conftest.check_workspace(before, root)


@pytest.mark.parametrize("change", ["touched", "appended", "replaced"])
def test_the_tripwire_names_a_changed_file(tmp_path, change):
    root = stand_in(tmp_path / "repo")
    before = conftest.workspace_state(root)
    target = root / "runs" / "cache" / "drive2018" / "inputs.u8"
    st = target.stat()
    if change == "touched":  # only the modification time moves
        os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    elif change == "appended":  # only the size moves
        with open(target, "ab") as f:
            f.write(b"x")
        os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))
    else:  # another file of the same size and time takes the name: only the inode moves
        shutil.copy2(target, target.with_name("copy"))
        os.replace(target.with_name("copy"), target)
        assert (target.stat().st_size, target.stat().st_mtime_ns) == (st.st_size, st.st_mtime_ns)
    with pytest.raises(pytest.fail.Exception, match=r"runs/cache/drive2018/inputs\.u8 \(changed\)"):
        conftest.check_workspace(before, root)


@pytest.mark.parametrize("path, what", [
    ("runs/cache/drive2017/meta.json", "deleted"),
    ("data/drive2018/data.txt", "deleted"),
    ("runs/cache/drive2017/inputs.u8.tmp-1", "new"),
    ("runs/cache/drivetest/inputs.u8", "new"),
])
def test_the_tripwire_names_a_deleted_or_new_file(tmp_path, path, what):
    root = stand_in(tmp_path / "repo")
    before = conftest.workspace_state(root)
    if what == "deleted":
        (root / path).unlink()
    else:
        (root / path).parent.mkdir(exist_ok=True)
        (root / path).write_bytes(b"")
    with pytest.raises(pytest.fail.Exception, match=rf"{re.escape(path)} \({what}\)") as exc:
        conftest.check_workspace(before, root)
    assert "\n" not in str(exc.value)  # one line, naming the file


def test_a_removed_cache_folder_names_every_file_it_held(tmp_path):
    root = stand_in(tmp_path / "repo")
    before = conftest.workspace_state(root)
    shutil.rmtree(root / "runs" / "cache")
    assert conftest.workspace_changes(before, root) == sorted(
        f"runs/cache/{d}/{f} (deleted)" for d in ("drive2017", "drive2018")
        for f in ("angles.npy", "inputs.u8", "meta.json"))


def test_without_a_cache_folder_the_tripwire_does_nothing(tmp_path):
    root = tmp_path / "fresh-clone"
    (root / "data" / "drive2018").mkdir(parents=True)
    (root / "data" / "drive2018" / "data.txt").write_text("0.jpg 1.5\n")
    before = conftest.workspace_state(root)
    assert before is None
    (root / "data" / "drive2018" / "data.txt").unlink()
    (root / "runs" / "cache" / "drivetest").mkdir(parents=True)
    (root / "runs" / "cache" / "drivetest" / "inputs.u8").write_bytes(b"x")
    assert conftest.workspace_changes(before, root) == []
    conftest.check_workspace(before, root)
