"""Shared fixtures: every test runs outside the repository, and synthetic drives.

A command's relative default (``runs``, ``data``, ``reports``) resolves against the working
directory, so every test runs from a tmp folder of its own and the session itself from
another, and a test that needs a repository file names it by an absolute path under
``REPO``. Behind that, a tripwire records the repository's drive caches and label files
when the session starts, before any test module is imported, and checks them again after
each test module is collected and after each test's setup, call and teardown: a cache that
training processes have memory-mapped must never change under them. A change fails the
collection of the module that made it, or the phase of the test it happened in; a change
made while a module's or the session's fixtures were torn down is reported at the teardown
of the test after which they were, and says so. When that collection or phase failed on its
own as well, the message names its error too, so the tripwire never hides a test's own
failure. Each change is reported once. Every check also compares with the record from the
start of the session, which nothing changes, and a repository without caches is watched as
well: a cache folder deleted or made is a change.

Synthetic drives are tiny, deterministic stand-ins for the real dataset. Each frame is a
455x256 BGR image with a bright vertical bar whose horizontal position follows the
steering angle, so a model can learn the mapping in one epoch and a broken pipeline is
visible. Labels use the timestamped format.

With ``scenery=True`` the bar stands in front of a grey background that drifts
slowly from frame to frame, as roadside scenery does: neighbouring frames look
almost alike and frames far apart do not. Without it, every frame showing the
same bar position is pixel-identical, however far apart in time.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

#: runs a pytest session of its own, for the tripwire's tests
pytest_plugins = ("pytester",)

#: the repository; tests reach its files by absolute paths built from this, never through
#: the working directory
REPO = Path(__file__).resolve().parents[1]

W, H = 455, 256


# --- every test runs outside the repository -------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def session_outside_the_repository(tmp_path_factory):
    """Module- and session-scoped fixtures are set up before a test enters its own folder,
    so the session runs from a tmp folder of its own as well."""
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path_factory.mktemp("session"))
        yield


@pytest.fixture(autouse=True)
def inside_tmp(tmp_path, monkeypatch):
    """Every test runs from its own tmp folder, so a relative default folder (runs/, data/,
    reports/) lands there and never reaches the repository's."""
    monkeypatch.chdir(tmp_path)


# --- the tripwire -----------------------------------------------------------------------------

def _watched_files(root: Path) -> dict[str, tuple[int, int, int]]:
    """``(size, mtime_ns, inode)`` of every file under ``root/runs/cache`` and of every
    ``root/data/<drive>/data.txt`` there is, keyed by the path relative to ``root``. Stat
    calls only, so it is cheap enough to run after every test."""
    paths = [Path(folder) / name for folder, _, names in os.walk(root / "runs" / "cache") for name in names]
    data = root / "data"
    if data.is_dir():
        paths += [Path(entry.path) / "data.txt" for entry in os.scandir(data) if entry.is_dir()]
    state = {}
    for path in paths:
        try:
            st = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:  # a drive without labels, or a file deleted while listed
            continue
        state[path.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns, st.st_ino)
    return state


def workspace_state(root: Path) -> dict[str, tuple[int, int, int]]:
    """The drive caches and label files of the repository at ``root`` (see ``_watched_files``).
    A repository without them gives an empty record, never none: a cache folder deleted or made
    later is a change like any other."""
    return _watched_files(root)


def state_changes(before: dict, after: dict) -> list[str]:
    """Each file of ``before`` changed in ``after`` or gone from it, and each file new in ``after``."""
    return sorted([f"{path} (deleted)" for path in before.keys() - after.keys()]
                  + [f"{path} (new)" for path in after.keys() - before.keys()]
                  + [f"{path} (changed)" for path in before.keys() & after.keys() if after[path] != before[path]])


def workspace_changes(before: dict, root: Path) -> list[str]:
    """Each file of ``before`` changed or deleted since, and each new one."""
    return state_changes(before, workspace_state(root))


def _fail(changes: list[str], when: str, also: str = "") -> None:
    pytest.fail(f"the repository's drive caches or label files changed{when}: {', '.join(changes)}; "
                f"a test writes only under its own tmp folder{also}", pytrace=False)


def check_workspace(before: dict, root: Path, when: str = "") -> None:
    """Fail, naming every changed file, when the workspace at ``root`` is not as ``before``;
    ``when`` says when it changed."""
    changes = workspace_changes(before, root)
    if changes:
        _fail(changes, when)


#: what the tripwire holds of the repository's drive caches and label files: ``origin``, recorded
#: when the session starts and never changed; ``state``, as the last check found them; and
#: ``reported``, every difference from the origin a check has reported. Each change is reported
#: once, where it happened, and a difference from the origin that no check reported is reported too.
TRIPWIRE = pytest.StashKey[dict]()


def start_tripwire(session, root: Path = REPO) -> None:
    state = workspace_state(root)
    session.stash[TRIPWIRE] = {"origin": dict(state), "state": state, "reported": set()}


def pytest_sessionstart(session):
    start_tripwire(session)


def tripwire(session, when: str, root: Path = REPO, also: str = "") -> None:
    """Fail with every change since the last check, and every difference from the start of the
    session that no check has reported yet; ``when`` says when it happened, and ``also`` is added
    to the message (the phase's own error, when it failed as well)."""
    watched = session.stash.get(TRIPWIRE, None)
    if watched is None:
        pytest.fail("the tripwire has no record from the start of the session; run the tests from the "
                    "repository or its tests folder", pytrace=False)
    after = workspace_state(root)
    from_origin = state_changes(watched["origin"], after)
    found = sorted(set(state_changes(watched["state"], after)) | (set(from_origin) - watched["reported"]))
    watched["state"] = after
    watched["reported"] |= set(from_origin)
    if found:
        _fail(found, when, also)


def error_summary(error: BaseException) -> str:
    """One line naming ``error``: its type and the first line of its message."""
    lines = str(error).strip().splitlines()
    return f"{type(error).__name__}: {lines[0]}" if lines else type(error).__name__


def report_summary(text: str) -> str:
    """One line from a failed report's text: its last error line (``E   ...``), else its last line."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    errors = [line[1:].strip() for line in lines if line.startswith("E ")]
    return (errors or lines or [""])[-1]


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    report = yield
    if isinstance(collector, pytest.Module):  # collecting a test module imports it: its top level runs
        # a module that failed to import keeps its own error: named in the message, in full below it
        own = report.longreprtext if report.failed else ""
        try:
            tripwire(collector.session, f" while {collector.nodeid} was imported",
                     also=f"; its collection also failed: {report_summary(own)}" if own else "")
        except pytest.fail.Exception as exc:
            report.outcome, report.longrepr = "failed", str(exc) + (f"\n\n{own}" if own else "")
    return report


def _watched(session, when: str):
    """The body of a test phase's hook wrapper: the phase runs, then the tripwire checks the caches
    and labels. When the phase raised too, the tripwire's failure names that error rather than
    hiding it; an interrupted session is left to stop."""
    try:
        result = yield
    except (KeyboardInterrupt, pytest.exit.Exception):
        raise
    except BaseException as error:
        tripwire(session, when, also=f"; it also raised {error_summary(error)}")
        raise
    tripwire(session, when)
    return result


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    return (yield from _watched(item.session, f" while {item.nodeid} was set up"))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    return (yield from _watched(item.session, f" while {item.nodeid} ran"))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    # a test's teardown also tears down its module's fixtures when the next test is in another
    # module, and the session's after the last test
    module = item.getparent(pytest.Module)
    name = item.nodeid if module is None else module.nodeid
    if nextitem is None:
        also = f" and the fixtures of {name} and of the session were"
    elif nextitem.getparent(pytest.Module) is not module:
        also = f" and the fixtures of {name} were"
    else:
        also = " was"
    return (yield from _watched(item.session, f" while {item.nodeid}{also} torn down"))


# --- synthetic drives -------------------------------------------------------------------------

def _scenery(seed: int, grid: tuple[int, int] = (6, 16), rho: float = 0.99):
    """Grey backgrounds, one per frame: a coarse random field that keeps a correlation
    of ``rho`` from one frame to the next, upsampled smoothly to the frame size."""
    rng = np.random.default_rng(seed)
    field = rng.normal(0, 1, grid)
    while True:
        smooth = cv2.resize(field.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        grey = np.clip(110.0 + 50.0 * smooth, 0, 255).astype(np.uint8)
        yield np.repeat(grey[:, :, None], 3, axis=2)
        field = rho * field + np.sqrt(1 - rho * rho) * rng.normal(0, 1, grid)


def make_drive(root: Path, name: str, n: int, timestamps: bool = True,
               dropouts: tuple[int, ...] = (), scenery: bool = False,
               intervals_ms: tuple[int, ...] = (57,)) -> Path:
    """``intervals_ms`` are the gaps between consecutive timestamps, repeated in turn."""
    drive = root / name
    drive.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(len(name) * 1000 + n)
    t = np.arange(n)
    angles = 40.0 * np.sin(t / 25.0) + rng.normal(0, 1.0, n)
    for i in dropouts:
        angles[i] = 0.0
    backgrounds = _scenery(len(name) * 1000 + n + 1) if scenery else None
    offsets_ms = np.concatenate([[0], np.cumsum(np.resize(np.asarray(intervals_ms), max(n - 1, 0)))])
    start = datetime(2018, 7, 1, 17, 0, 0)
    lines = []
    for i in range(n):
        img = np.full((H, W, 3), 60, np.uint8) if backgrounds is None else next(backgrounds)
        x = int(W / 2 + angles[i] * 2.0)
        cv2.rectangle(img, (max(0, x - 6), 120), (min(W - 1, x + 6), H - 1), (220, 220, 220), -1)
        cv2.imwrite(str(drive / f"{i}.jpg"), img)
        if timestamps:
            ts = start + timedelta(milliseconds=int(offsets_ms[i]))
            lines.append(f"{i}.jpg {angles[i]:.6f},{ts:%Y-%m-%d %H:%M:%S}:{ts.microsecond // 1000}")
        else:
            lines.append(f"{i}.jpg {angles[i]:.6f}")
    (drive / "data.txt").write_text("\n".join(lines) + "\n")
    return drive


@pytest.fixture
def tiny_drive(tmp_path: Path) -> Path:
    return make_drive(tmp_path / "data", "drivetest", n=900)
