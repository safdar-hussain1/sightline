"""The mutation battery's own test: the battery must be able to fail.

``scripts/mutation_battery.py`` says whether each published claim has a test behind it. "No survivor,
as every claim is defended" and "no survivor, as nothing was mutated" print the same line, so what is
tested here is that the battery reports what it should when it should. Handed a claim no test
defends, it exits 1 and names the claim. Handed a stale or a duplicated anchor, or a named test that
is red before anything is mutated, it stops with a hard error and writes nothing. After a row that
raised, or one interrupted by Ctrl-C or SIGTERM, the file is back byte for byte. A control that goes
red fails the run.

Each test drives the battery as a command against a throwaway git repository built in ``tmp_path``,
holding a two-rule engine where one rule has a test and the other has none. Showing that the battery
can find a survivor in this repository would mean keeping an undefended claim here for it to find.
The tests at the end read the real table without mutating anything: it holds the rows and the
controls it should, every anchor is still in its file exactly once, every whole-file rewrite keeps
its file's value and every named test still exists.
"""

import hashlib
import importlib.util
import json
import os
import py_compile
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "scripts" / "mutation_battery.py"
GUARD_WORDS_ENV = "SIGHTLINE_GUARD_WORDS"


@pytest.fixture(scope="module")
def battery():
    """The battery imported by its path (``scripts/`` is not a package), registered in
    ``sys.modules`` first, as its dataclasses look their module up there."""
    spec = importlib.util.spec_from_file_location("_mutation_battery", BATTERY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ENGINE = '''"""A two-rule engine: one rule has a test, the other has none."""


def defended_sum(a, b):
    return a + b


def undefended_product(a, b):
    return a * b
'''

ENGINE_TEST = '''import json
import time
from pathlib import Path

import pytest

from engine import defended_sum


def test_defended_sum_adds():
    assert defended_sum(2, 3) == 5


def test_red_on_the_clean_engine():
    assert defended_sum(2, 2) == 5


@pytest.mark.skip(reason="collected, never run")
def test_never_runs():
    assert defended_sum(2, 3) == 5


def test_waits_while_the_sum_is_wrong():
    """Passes at once on the clean engine; under a mutation it waits to be interrupted."""
    deadline = time.monotonic() + 120
    while defended_sum(2, 3) != 5 and time.monotonic() < deadline:
        time.sleep(0.1)
    assert defended_sum(2, 3) == 5


def test_the_recorded_sum_is_the_sum():
    recorded = json.loads(Path(__file__).with_name("data.json").read_text())
    assert recorded == {"sum": defended_sum(2, 3)}
'''

#: a made-up word the sandbox's own guard refuses, held there as its digest like the real guard's
WORD = "heliotrope"
WORD_DIGEST = hashlib.sha256(WORD.encode()).hexdigest()
NOTES = "The engine adds and multiplies.\n"
SANDBOX_GUARD = f'''import hashlib
import re
from pathlib import Path

REFUSED = {{"{WORD_DIGEST}"}}


def test_no_refused_word():
    text = Path(__file__).resolve().parents[1].joinpath("NOTES.txt").read_text()
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    assert [t for t in tokens if hashlib.sha256(t.encode()).hexdigest() in REFUSED] == []
'''
#: the readable list behind the sandbox guard's digest, kept outside the sandbox as the real one is
WORD_LIST = ("| entry | normal form | kind | match | sha256 of the normal form |\n|---|---|---|---|---|\n"
             f"| {WORD} | {WORD} | probe | exact | `{WORD_DIGEST}` |\n")

#: the mutation that breaks the DEFENDED rule; its named test must go red
DEFENDED = {
    "id": "defended",
    "claim": "the defended rule adds its arguments",
    "path": "engine.py",
    "find": "    return a + b\n",
    "replace": "    return a - b\n",
    "runner": "pytest",
    "tests": ["test_engine.py::test_defended_sum_adds"],
    "expect": "must_fail",
    "note": "subtracts instead of adding",
}
#: a mutation of code no test reads, whose row claims the defended rule's test defends it
CANARY = dict(DEFENDED, id="canary", claim="the undefended rule multiplies its arguments",
              find="    return a * b\n", replace="    return a / b\n", note="divides, and nothing looks")
#: the defended mutation filed as a control: the control goes red, which must fail the run
CONTROL = dict(DEFENDED, id="control", claim="control that breaks the rule", expect="must_survive",
               note="axis: none at all, it changes what the rule does")
#: a mutation that leaves the engine unable to load, so pytest cannot run the named test at all
RAISES = dict(DEFENDED, id="raises", replace="    return a +\n")
#: a mutation whose named test waits to be interrupted
WAITS = dict(DEFENDED, id="waits", tests=["test_engine.py::test_waits_while_the_sum_is_wrong"])
#: a row that plants the sandbox guard's word, which only the list outside the sandbox spells
PLANTED = {
    "id": "planted",
    "claim": "the sandbox guard refuses its word",
    "path": "NOTES.txt",
    "find": NOTES,
    "replace": NOTES + "It was checked with a <<plant>>.\n",
    "runner": "pytest",
    "tests": ["tests/test_guards.py::test_no_refused_word"],
    "expect": "must_fail",
    "note": "the refused word goes into the notes",
    "plant_entry": WORD_DIGEST,
    "plant_rule": WORD_DIGEST,
    "plant_match": "exact",
}
#: whole-file rewrites of the recorded sum: one changes only the layout, one changes nothing
REINDENTED = {"id": "reindented", "claim": "control: the recorded sum written with an indent of four",
              "path": "data.json", "kind": "json-indent-4", "runner": "pytest",
              "tests": ["test_engine.py::test_the_recorded_sum_is_the_sum"], "expect": "must_survive",
              "note": "axis: whitespace"}
REORDERED = dict(REINDENTED, id="reordered", kind="json-keys-reversed",
                 claim="control: the recorded sum's keys in reverse order (it has one key)")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _commit(root: Path, message: str) -> None:
    # -f so that a global excludes file cannot leave a fixture file out, and an explicit identity
    _git(root, "add", "-f", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false",
         "commit", "--no-verify", "-q", "-m", message)


def _porcelain(root: Path) -> str:
    return subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True,
                          check=True).stdout


@pytest.fixture
def sandbox(tmp_path) -> Path:
    """A committed throwaway repository: the engine, its tests, notes and a guard that reads them."""
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    files = {"engine.py": ENGINE, "test_engine.py": ENGINE_TEST, "tests/test_guards.py": SANDBOX_GUARD,
             "NOTES.txt": NOTES, "data.json": json.dumps({"sum": 5}, indent=2) + "\n",
             "pytest.ini": "[pytest]\n", ".gitignore": "__pycache__/\n"}
    for rel, text in files.items():
        (root / rel).write_text(text)
    _git(root, "init", "-q")
    _commit(root, "sandbox")
    assert _porcelain(root) == "", "the sandbox did not start clean"
    return root


def _command(root: Path, rows: list[dict], *extra: str) -> list[str]:
    # the table, the results and the runs' temporary folders all live OUTSIDE the sandbox: inside it
    # they would make the tree dirty, which the battery refuses
    table = root.parent / "table.json"
    table.write_text(json.dumps(rows))
    return [sys.executable, str(BATTERY), "--root", str(root), "--table", str(table),
            "--basetemp", str(root.parent / "basetemp"), "--results", str(root.parent / "results.json"),
            *extra]


def _env(**extra: str) -> dict:
    env = {key: value for key, value in os.environ.items() if key != GUARD_WORDS_ENV}
    env.update(extra)
    return env


def _run(root: Path, rows: list[dict], *extra: str, env: dict | None = None):
    """The battery's exit, its output and the results it wrote (None if it wrote none)."""
    proc = subprocess.run(_command(root, rows, *extra), capture_output=True, text=True,
                          env=env or _env(), timeout=600)
    out = root.parent / "results.json"
    return proc, json.loads(out.read_text()) if out.is_file() else None


# --- what the battery must report ---------------------------------------------------------------


def test_a_claim_no_named_test_notices_is_reported_as_a_survivor(sandbox):
    """The finding the battery exists for: exit 1, the claim, the file's byte change and the test
    that stayed green, named, so the reader can act on it."""
    proc, results = _run(sandbox, [CANARY])

    assert proc.returncode == 1, f"{proc.stdout}\n{proc.stderr}"
    assert "SURVIVORS" in proc.stdout and CANARY["claim"] in proc.stdout
    assert "test_engine.py::test_defended_sum_adds" in proc.stdout and "bytes" in proc.stdout
    (row,) = results["rows"]
    assert (row["outcome"], row["ok"], row["red_tests"]) == ("survived", False, [])
    assert (sandbox / "engine.py").read_text() == ENGINE and _porcelain(sandbox) == ""


def test_a_defended_claim_is_killed_and_the_results_name_the_test_that_went_red(sandbox):
    """The other half: a battery that always exited 1 would pass the test above."""
    proc, results = _run(sandbox, [DEFENDED])

    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "PASSED" in proc.stdout and "SURVIVORS" not in proc.stdout
    (row,) = results["rows"]
    assert (row["outcome"], row["ok"], row["red_tests"]) == ("killed", True, DEFENDED["tests"])
    assert row["mutated"][0]["failed"] == 1 and row["clean_again"][0]["passed"] == 1
    engine = hashlib.sha256(ENGINE.encode()).hexdigest()
    assert row["sha256_before"] == row["sha256_restored"] == engine
    plus = ENGINE.index("a + b") + 2
    assert (row["changed_at"], row["removed_bytes"], row["inserted_bytes"]) == (plus, 1, 1)
    assert results["exit_code"] == 0 and results["error"] == ""


def test_a_stale_anchor_is_a_hard_error_and_nothing_is_written(sandbox):
    """A bare find-and-replace that matched nothing would run green while mutating nothing."""
    proc, results = _run(sandbox, [dict(DEFENDED, find="    return a + b + 1\n")])

    assert proc.returncode == 2, f"{proc.stdout}\n{proc.stderr}"
    assert "NOT PRESENT" in proc.stderr and "SURVIVORS" not in proc.stdout
    assert results["rows"] == [] and results["exit_code"] == 2
    assert (sandbox / "engine.py").read_text() == ENGINE and _porcelain(sandbox) == ""


def test_a_duplicated_anchor_is_a_hard_error(sandbox):
    """An anchor found twice would mutate more than its row says."""
    proc, _ = _run(sandbox, [dict(DEFENDED, find="    return a ")])

    assert proc.returncode == 2, f"{proc.stdout}\n{proc.stderr}"
    assert "occurs 2 times" in proc.stderr
    assert (sandbox / "engine.py").read_text() == ENGINE and _porcelain(sandbox) == ""


def test_a_named_test_red_on_the_clean_tree_is_a_hard_error(sandbox):
    """A test that is red before the mutation would score the mutation as caught."""
    proc, results = _run(sandbox, [dict(DEFENDED, tests=["test_engine.py::test_red_on_the_clean_engine"])])

    assert proc.returncode == 2, f"{proc.stdout}\n{proc.stderr}"
    assert "already red on the clean tree" in proc.stderr
    assert results["rows"] == []
    assert (sandbox / "engine.py").read_text() == ENGINE and _porcelain(sandbox) == ""


@pytest.mark.parametrize("test, message", [
    ("test_engine.py::test_never_runs", "passes no case on the clean tree"),  # collected, only skips
    ("test_engine.py::test_absent", "could not run"),  # collects nothing: pytest exits 4
])
def test_a_named_test_that_cannot_go_red_is_a_hard_error(sandbox, test, message):
    proc, _ = _run(sandbox, [dict(DEFENDED, tests=[test])])

    assert proc.returncode == 2, f"{proc.stdout}\n{proc.stderr}"
    assert message in proc.stderr
    assert (sandbox / "engine.py").read_text() == ENGINE and _porcelain(sandbox) == ""


def test_the_file_is_restored_byte_for_byte_after_a_row_that_raised(sandbox):
    """The mutation leaves the engine unable to load, so the mutated run raises; the file must be
    back all the same, and the error is a hard one, never a kill."""
    proc, _ = _run(sandbox, [RAISES])

    assert proc.returncode == 2, f"{proc.stdout}\n{proc.stderr}"
    # pytest exits 2 or 4 when a named test's module fails to import
    assert "HARD ERROR" in proc.stderr and "(mutated): pytest exited" in proc.stderr
    assert "could not run" in proc.stderr
    assert (sandbox / "engine.py").read_bytes() == ENGINE.encode() and _porcelain(sandbox) == ""


def _ignore_interrupts() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["ctrl-c", "sigterm"])
def test_an_interrupt_in_the_middle_of_a_row_restores_the_file(sandbox, signum):
    """The signal lands while the named test runs on the mutated engine. The battery is started
    with both signals ignored, as a shell starts a background job of a script, and must still
    stop and restore the file."""
    proc = subprocess.Popen(_command(sandbox, [WAITS]), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=_env(), preexec_fn=_ignore_interrupts)
    try:
        deadline = time.monotonic() + 120
        while (sandbox / "engine.py").read_text() == ENGINE:
            assert proc.poll() is None, proc.communicate()
            assert time.monotonic() < deadline, "the battery never wrote the mutation"
            time.sleep(0.05)
        time.sleep(1.0)  # into the run of the named test
        proc.send_signal(signum)
        out, err = proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == 130, f"{out}\n{err}"
    assert "INTERRUPTED" in err
    assert (sandbox / "engine.py").read_bytes() == ENGINE.encode() and _porcelain(sandbox) == ""


def test_a_control_that_goes_red_fails_the_run(sandbox):
    """A control is an assertion too: a change that keeps the meaning must keep its test green."""
    proc, results = _run(sandbox, [CONTROL])

    assert proc.returncode == 1, f"{proc.stdout}\n{proc.stderr}"
    assert "CONTROLS THAT WENT RED" in proc.stdout and CONTROL["claim"] in proc.stdout
    (row,) = results["rows"]
    assert (row["outcome"], row["ok"]) == ("killed", False)


def test_a_whole_file_rewrite_must_change_the_bytes_and_keep_the_value(sandbox):
    proc, results = _run(sandbox, [REINDENTED])
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert results["rows"][0]["outcome"] == "survived"

    # one key in reverse order is the same file: a rewrite that changes nothing is an error
    proc, _ = _run(sandbox, [REORDERED])
    assert proc.returncode == 2 and "changed NO bytes" in proc.stderr, proc.stderr
    assert _porcelain(sandbox) == ""


@pytest.fixture
def sandbox_with_stale_bytecode(sandbox: Path) -> Path:
    """The sandbox with bytecode of the clean engine beside it, which CPython trusts unchecked.

    A run that read bytecode from beside the sources would score a defended claim as a survivor:
    CPython checks a cached ``.pyc`` against the source's size and its mtime in whole seconds, so a
    mutation of the same length written within the same second runs the clean code. An
    unchecked-hash ``.pyc`` is that defect without the stopwatch.
    """
    cached = sandbox / "__pycache__" / f"engine.{sys.implementation.cache_tag}.pyc"
    py_compile.compile(str(sandbox / "engine.py"), cfile=str(cached), doraise=True,
                       invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
    (sandbox / "engine.py").write_text(ENGINE.replace("return a + b", "return a - b"))
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPYCACHEPREFIX"}
    proof = subprocess.run([sys.executable, "-c", "import engine; print(engine.defended_sum(2, 3))"],
                           cwd=sandbox, capture_output=True, text=True, env={**env, "PYTHONDONTWRITEBYTECODE": "1"})
    (sandbox / "engine.py").write_text(ENGINE)
    assert proof.stdout.strip() == "5", f"the stale bytecode does not mask the source: {proof.stderr}"
    assert _porcelain(sandbox) == ""
    return sandbox


def test_the_mutated_bytes_are_run_not_bytecode_left_beside_them(sandbox_with_stale_bytecode):
    proc, results = _run(sandbox_with_stale_bytecode, [DEFENDED])

    assert proc.returncode == 0, f"a defended claim scored as a survivor: stale bytecode ran\n{proc.stdout}"
    assert results["rows"][0]["outcome"] == "killed"


def test_a_dirty_tree_is_refused_and_an_untracked_path_only_when_allowed(sandbox):
    """The battery edits tracked files in place, so uncommitted work stops it; the refusal says
    that git-ignored paths are outside what it can see."""
    (sandbox / "engine.py").write_text(ENGINE + "# an uncommitted edit\n")
    proc, _ = _run(sandbox, [DEFENDED])
    assert proc.returncode == 2 and "REFUSING TO RUN" in proc.stderr and "git-ignored" in proc.stderr
    assert (sandbox / "engine.py").read_text() == ENGINE + "# an uncommitted edit\n"

    (sandbox / "engine.py").write_text(ENGINE)
    (sandbox / "scratch.txt").write_text("untracked work\n")
    proc, _ = _run(sandbox, [DEFENDED])
    assert proc.returncode == 2 and "REFUSING TO RUN" in proc.stderr
    proc, _ = _run(sandbox, [DEFENDED], "--allow-untracked", "scratch.txt")
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert (sandbox / "scratch.txt").read_text() == "untracked work\n"
    assert _porcelain(sandbox) == "?? scratch.txt\n"


def test_a_root_that_is_not_a_git_repository_is_a_hard_error(tmp_path):
    """Exit 1 means survivors; a failed `git status` must not be mistaken for that."""
    root = tmp_path / "plain"
    root.mkdir()
    (root / "engine.py").write_text(ENGINE)
    (root / "test_engine.py").write_text(ENGINE_TEST)
    proc, _ = _run(root, [DEFENDED])

    assert proc.returncode == 2, f"{proc.stdout}\n{proc.stderr}"
    assert "not a usable git repository" in proc.stderr and "Traceback" not in proc.stderr


def test_several_rows_leave_the_tree_byte_identical(sandbox):
    before = {name: (sandbox / name).read_bytes() for name in ("engine.py", "NOTES.txt", "data.json")}
    proc, results = _run(sandbox, [DEFENDED, CANARY, REINDENTED, DEFENDED | {"id": "again"}])

    assert proc.returncode == 1  # the canary survives, as it must
    assert [row["outcome"] for row in results["rows"]] == ["killed", "survived", "survived", "killed"]
    assert {name: (sandbox / name).read_bytes() for name in before} == before
    assert _porcelain(sandbox) == ""


@pytest.mark.parametrize("before, after, span", [
    (b"abcdef", b"abXdef", (2, 1, 1)),
    (b"abcdef", b"abdef", (2, 1, 0)),
    (b"abcdef", b"abcXYdef", (3, 0, 2)),
    (b"aaaa", b"aaaaaa", (4, 0, 2)),  # a repeated byte: the common start is taken first
    (b"abc", b"xyz", (0, 3, 3)),
])
def test_the_byte_proof_names_the_one_stretch_that_changed(battery, before, after, span):
    assert battery.changed_span(before, after) == span


# --- planted words ------------------------------------------------------------------------------


def test_a_planted_row_is_skipped_with_its_reason_when_the_list_is_not_given(sandbox, battery):
    assert PLANTED["replace"].count(battery.PLANT) == 1 and battery.GUARD_WORDS_ENV == GUARD_WORDS_ENV
    proc, results = _run(sandbox, [PLANTED, DEFENDED])

    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "SKIPPED" in proc.stdout and f"{GUARD_WORDS_ENV} is not set" in proc.stdout
    assert [row["outcome"] for row in results["rows"]] == ["skipped", "killed"]
    assert (sandbox / "NOTES.txt").read_text() == NOTES and _porcelain(sandbox) == ""


def test_a_planted_row_plants_the_listed_word_and_never_prints_it(sandbox, tmp_path):
    words = tmp_path / "words.md"
    words.write_text(WORD_LIST)
    proc, results = _run(sandbox, [PLANTED], env=_env(**{GUARD_WORDS_ENV: str(words)}))

    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert results["rows"][0]["outcome"] == "killed"
    assert results["rows"][0]["red_tests"] == PLANTED["tests"]
    written = proc.stdout + proc.stderr + (sandbox.parent / "results.json").read_text()
    assert WORD not in written
    assert (sandbox / "NOTES.txt").read_text() == NOTES and _porcelain(sandbox) == ""


def test_a_hard_error_never_prints_the_planted_word_though_the_test_output_does(sandbox, tmp_path):
    """Planted into the engine, the word breaks its syntax, and pytest's own report of that quotes
    the line; the battery's error message must carry it redacted."""
    words = tmp_path / "words.md"
    words.write_text(WORD_LIST)
    row = dict(PLANTED, path="engine.py", find="    return a + b\n", replace="    return a + b <<plant>>\n",
               tests=DEFENDED["tests"])
    proc, results = _run(sandbox, [row], env=_env(**{GUARD_WORDS_ENV: str(words)}))

    assert proc.returncode == 2, f"{proc.stdout}\n{proc.stderr}"
    assert "<the planted word>" in proc.stderr
    assert WORD not in proc.stdout + proc.stderr + (sandbox.parent / "results.json").read_text()
    assert (sandbox / "engine.py").read_text() == ENGINE and _porcelain(sandbox) == ""


def test_a_list_whose_entry_does_not_hash_to_its_digest_is_a_hard_error(sandbox, tmp_path):
    words = tmp_path / "words.md"
    words.write_text(WORD_LIST.replace(f"| {WORD} |", "| lavender |", 1))
    proc, _ = _run(sandbox, [PLANTED], env=_env(**{GUARD_WORDS_ENV: str(words)}))

    assert proc.returncode == 2 and "does not hash to it" in proc.stderr, proc.stderr
    assert (sandbox / "NOTES.txt").read_text() == NOTES


# --- the real table, read without mutating anything ---------------------------------------------

#: every claim the table must hold a row for, and the controls; a deleted row fails here
REQUIRED_IDS = {*map(str, range(1, 21)), "4b", "a", "b", "c", "d", "e"}


def test_the_real_table_holds_every_claim_and_every_control(battery):
    ids = [entry.id for entry in battery.CLAIMS]
    assert len(ids) == len(set(ids)) and REQUIRED_IDS <= set(ids)
    controls = [entry for entry in battery.CLAIMS if entry.expect == battery.MUST_SURVIVE]
    assert len(controls) >= 5 and all("axis:" in entry.note for entry in controls)
    assert sum(entry.expect == battery.MUST_FAIL for entry in battery.CLAIMS) >= 21


def test_every_anchor_of_the_real_table_is_in_its_file_once_and_every_rewrite_keeps_its_value(battery):
    """A refactor that moves a mutated line fails here, in the suite, before anyone runs the
    battery. Planted rows are checked for their anchor alone: the list behind them is not given."""
    plans = battery.plan_rows(ROOT, battery.CLAIMS, env={})
    for plan in plans:
        if plan.skipped:
            assert plan.entry.planted and GUARD_WORDS_ENV in plan.skipped
        else:
            assert plan.after != plan.before, plan.entry.id
    assert sum(not plan.skipped for plan in plans) >= len(plans) - 3


def test_every_named_test_of_the_real_table_exists(battery):
    missing = []
    for entry in battery.CLAIMS:
        for test in entry.tests:
            file_part, _, name = test.partition("::")
            if entry.runner == "pytest":
                source = ROOT / file_part
                needle = f"def {name.partition('[')[0]}("
            else:
                source = ROOT / "web" / file_part
                needle = name
            if not source.is_file() or needle not in source.read_text():
                missing.append(test)
    assert missing == [], f"the table names tests that no longer exist: {missing}"


def test_every_planted_row_exercises_a_rule_the_guard_holds(battery):
    guard = (ROOT / "tests" / "test_guards.py").read_text()
    planted = [entry for entry in battery.CLAIMS if entry.planted]
    assert {entry.id for entry in planted} == {"18", "19", "e"}
    assert all(entry.plant_rule in guard for entry in planted)
