"""No test can reach the repository's own data, caches or reports.

A relative default (``runs``, ``data``, ``reports``) resolves against the working
directory, so every test, and every module- or session-scoped fixture, runs from a tmp
folder of its own (conftest.py). Behind that, a tripwire records the repository's drive
caches and label files when the session starts and fails the collection or the test phase
in which any of them changed. The tripwire is checked here on stand-in repositories: its
functions directly, and its hooks by running a pytest session of their own on a stand-in
repository that holds a copy of conftest.py.
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


def test_the_tripwire_holds_this_repositorys_caches_and_labels_as_they_are(request):
    watched = request.session.stash[conftest.TRIPWIRE]["state"]
    assert (watched is None) == (not (REPO / "runs" / "cache").is_dir())
    assert watched == conftest.workspace_state(REPO)


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


# --- the tripwire's hooks, in a pytest session of their own ------------------------------------

def session_on_a_stand_in(pytester, tests: dict[str, str]):
    """Run pytest on a stand-in repository with a copy of conftest.py and the test modules
    ``tests`` (file name to source), in a process of its own; return its result."""
    root = stand_in(pytester.path)
    (root / "tests").mkdir()
    (root / "tests" / "conftest.py").write_text(Path(conftest.__file__).read_text())
    header = ("from pathlib import Path\nimport pytest\nROOT = Path(__file__).resolve().parents[1]\n"
              "CACHE = ROOT / 'runs' / 'cache' / 'drive2018'\n")
    for name, source in tests.items():
        (root / "tests" / name).write_text(header + source)
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", "tests")


def _reported(result, nodeid: str, when: str, changes: str) -> bool:
    """Whether the session's output names ``nodeid`` with the tripwire's message for ``when``."""
    return f"the repository's drive caches or label files changed while {nodeid}{when}: {changes}; " \
        "a test writes only under its own tmp folder" in result.stdout.str()


def test_the_tripwire_fails_the_phase_a_change_happens_in_and_reports_each_change_once(pytester):
    result = session_on_a_stand_in(pytester, {
        "test_a.py": (
            "def test_reads_only():\n"
            "    (CACHE / 'inputs.u8').read_bytes()\n"
            "def test_writes_into_the_cache():\n"
            "    (CACHE / 'inputs.u8').write_bytes(b'other frames')\n"
            "def test_after_it():\n"
            "    pass\n"
            "@pytest.fixture\n"
            "def writes_while_set_up():\n"
            "    (CACHE / 'angles.npy').write_bytes(b'other angles')\n"
            "def test_with_a_fixture_that_writes(writes_while_set_up):\n"
            "    pass\n"),
        "test_b.py": (
            "@pytest.fixture(scope='module')\n"
            "def leaves_a_file():\n"
            "    yield\n"
            "    (CACHE / 'inputs.u8.tmp-1').write_bytes(b'')\n"
            "def test_one(leaves_a_file):\n"
            "    pass\n"
            "def test_two(leaves_a_file):\n"
            "    pass\n"),
        "test_c.py": (
            "@pytest.fixture(scope='session')\n"
            "def edits_the_labels_at_the_end():\n"
            "    yield\n"
            "    (ROOT / 'data' / 'drive2017' / 'data.txt').write_text('0.jpg 2.5\\n')\n"
            "def test_next(edits_the_labels_at_the_end):\n"
            "    pass\n"),
    })
    # a failure in the call phase, and errors at a setup and at two teardowns (both of those tests
    # passed first); every other test passes, so each change was reported once, where it happened
    result.assert_outcomes(passed=5, failed=1, errors=3)
    assert _reported(result, "tests/test_a.py::test_writes_into_the_cache", " ran",
                     "runs/cache/drive2018/inputs.u8 (changed)")
    assert _reported(result, "tests/test_a.py::test_with_a_fixture_that_writes", " was set up",
                     "runs/cache/drive2018/angles.npy (changed)")
    # a module fixture's teardown is blamed on the test whose teardown ran it, and says so
    assert _reported(result, "tests/test_b.py::test_two", " and the fixtures of tests/test_b.py were torn down",
                     "runs/cache/drive2018/inputs.u8.tmp-1 (new)")
    assert _reported(result, "tests/test_c.py::test_next",
                     " and the fixtures of tests/test_c.py and of the session were torn down",
                     "data/drive2017/data.txt (changed)")


def test_the_tripwire_fails_the_collection_of_a_module_that_writes_when_it_is_imported(pytester):
    result = session_on_a_stand_in(pytester, {
        "test_a.py": "def test_first():\n    pass\n",
        "test_b.py": "(CACHE / 'meta.json').unlink()\ndef test_never_runs():\n    pass\n",
    })
    # the record from the start of the session sees the write; a record taken at the first
    # test's setup would have made it the baseline
    assert result.ret != 0
    result.assert_outcomes(errors=1)
    assert _reported(result, "tests/test_b.py", " was imported", "runs/cache/drive2018/meta.json (deleted)")
