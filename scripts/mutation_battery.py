#!/usr/bin/env python3
"""Break each claim Sightline publishes, one at a time, and require the tests named for it to notice.

What it does
------------
Each row of ``CLAIMS`` names a claim, one single-file mutation that removes the mechanism behind
it, and the tests that exist to defend it. The battery writes the mutation, runs those tests, puts
the file back and proves it is back. A must-fail row is KILLED when every test it names goes red;
when one of them stays green the row is a SURVIVOR, a claim with no test behind it. Must-survive
rows are controls: a change that keeps the meaning (prose order, whitespace, a local name, key
order, a word used as a command-line flag), under which the same tests must stay green, so that a
kill means a test reads what the claim says and not how a file happens to spell it.

Why it distrusts itself
-----------------------
A battery that mutates nothing reports "every claim defended" in the same words as one that works.
So every step proves itself:

1. An anchor must occur exactly once in its file (overlapping matches count), and the mutation must
   change the file's bytes. A whole-file rewrite (a JSON file written out again) must parse back to
   the same value and still change the bytes. All of it is checked before any file is written.
2. Before any mutation, the tests of every row run on the clean tree and must pass, each named
   test with at least one case that passes. A named test that is red already, collects nothing or
   only skips is a hard error: "it failed" and "it never ran" must never be confused.
3. A row is killed only when its named tests run and fail. A mutation that stops a test module
   loading at all (pytest exit 2, a vitest file that does not load) is a hard error, as it tells
   nothing about the claim.
4. The file is restored from a byte copy held in memory, never from git, and the restored bytes
   and their sha256 are checked against the copy. The restore sits in a ``finally``, so an
   exception, Ctrl-C or SIGTERM restores the file too, and both writes run with those signals held.
   The named tests then run once more on the restored tree and must pass again.
5. ``git status --porcelain`` must be identical before and after the run, and the run refuses to
   start on a dirty tree, apart from untracked paths named with ``--allow-untracked``. That status
   does not see git-ignored paths, and no row touches one.
6. Every pytest run gets ``-p no:cacheprovider``, a ``--basetemp`` of its own outside the
   repository, bytecode writing off and an empty bytecode cache of its own, so a mutated source is
   always compiled from the bytes on disk, never read from a stale ``.pyc`` whose size and mtime
   happen to match it.
7. ``tests/test_mutation_battery_smoke.py`` shows on a throwaway git repository that the battery
   reports a survivor, errors on a stale or a duplicated anchor and on a named test that is red on
   the clean tree, restores a file byte for byte after a row that raised or was interrupted, and
   fails a control that goes red.

Planted words
-------------
Two rows plant a word the public-text guard refuses, and one control plants a word the guard
allows only as a command-line flag. ``tests/test_guards.py`` keeps its words as sha256 digests,
and so does this file: each plant names the digest of an entry in the readable list kept outside
the repository, at the path the ``SIGHTLINE_GUARD_WORDS`` environment variable gives. At run time
the battery finds that entry, checks that its normal form has the digest named, checks that the
guard rule the row exercises (a digest that must be in ``tests/test_guards.py``) matches the word,
and only then writes the word into the file, from where the restore removes it. The word is never
printed and never written to the results. Without the variable those rows are skipped, and the
report says why.

Exit codes
----------
0    every must-fail row killed and every control green (skipped rows are listed as skipped);
1    a survivor, or a control that went red;
2    a hard error: a dirty tree, a stale or ambiguous anchor, a named test that is not green on the
     clean tree, a mutation that changed nothing, a failed restore, a tree left changed;
130  interrupted; the file being mutated was restored.

Usage
-----
    python scripts/mutation_battery.py --list
    python scripts/mutation_battery.py --results OUT.json [--basetemp DIR] [--only ID,ID]
    python scripts/mutation_battery.py --root CHECKOUT --table ROWS.json   (the smoke test)

The pytest rows run with the interpreter that runs this script, and the vitest rows with
``npx vitest`` from ``web/``. Nothing is imported from ``src/``: the battery must not depend on the
code it mutates.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

REPO_ROOT = Path(__file__).resolve().parents[1]

MUST_FAIL = "must_fail"
MUST_SURVIVE = "must_survive"
EXPECTATIONS = (MUST_FAIL, MUST_SURVIVE)
RUNNERS = ("pytest", "vitest")
#: where a row's replacement takes its planted word
PLANT = "<<plant>>"
#: the environment variable naming the readable list behind the guard's digests
GUARD_WORDS_ENV = "SIGHTLINE_GUARD_WORDS"
#: characters a vitest name filter may not hold: ``-t`` reads a regular expression
REGEX_MARKS = set(".^$*+?()[]{}|\\")


class HardError(RuntimeError):
    """The battery cannot trust its own run. Never a skip, never a pass."""


class Interrupted(BaseException):
    """SIGTERM, raised where it lands like Ctrl-C, so that every ``finally`` restores its file."""


# --- whole-file rewrites ----------------------------------------------------------------------


def _reversed_keys(value):
    if isinstance(value, dict):
        return {key: _reversed_keys(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reversed_keys(item) for item in value]
    return value


def json_indent_4(data: bytes) -> bytes:
    """The JSON file written again with an indent of four, keys sorted as the reports write them."""
    return (json.dumps(json.loads(data), indent=4, sort_keys=True) + "\n").encode()


def json_keys_reversed(data: bytes) -> bytes:
    """The JSON file written again with the keys of every object in reverse order, indent two."""
    return (json.dumps(_reversed_keys(json.loads(data)), indent=2) + "\n").encode()


#: the rewrites a row can name as its ``kind``; each must leave the parsed value as it was
TRANSFORMS = {"json-indent-4": json_indent_4, "json-keys-reversed": json_keys_reversed}


# --- the rows ---------------------------------------------------------------------------------


def _dotted(node_id: str) -> str:
    """A pytest node id as its JUnit report spells it, ``classname.name``."""
    path, bracket, params = node_id.partition("[")
    names = path.split("::")
    names[0] = re.sub(r"\.py$", "", names[0].replace("/", "."))
    return ".".join(names) + bracket + params


def _belongs(case: str, test: str) -> bool:
    return case == test or case.startswith(test + "[") or case.startswith(test + ".")


@dataclass(frozen=True)
class Entry:
    """One claim, one single-file mutation, the tests named to defend it."""

    id: str
    claim: str
    path: str
    runner: str
    #: pytest node ids, or vitest ``<file under web/>::<name filter>``
    tests: tuple[str, ...]
    expect: str
    #: for a must-fail row, what the mutation removes; for a control, ``axis:`` and the axis it varies
    note: str
    #: the anchor and what replaces it: UTF-8 text (``kind="text"``) or hex bytes (``kind="hex"``);
    #: a ``kind`` naming one of ``TRANSFORMS`` rewrites the whole file and takes neither
    find: str = ""
    replace: str = ""
    kind: str = "text"
    #: a planted word: the sha256 of its entry in the readable list, the sha256 of the guard rule
    #: the row exercises, how that rule matches the word ("exact" or "stem"), the file holding it
    plant_entry: str = ""
    plant_rule: str = ""
    plant_match: str = ""
    plant_guard: str = "tests/test_guards.py"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tests", tuple(self.tests))
        where = f"row {self.id!r}"
        if self.expect not in EXPECTATIONS:
            raise HardError(f"{where}: unknown expectation {self.expect!r}")
        if self.runner not in RUNNERS:
            raise HardError(f"{where}: unknown runner {self.runner!r}")
        if not self.tests or len(set(self.tests)) != len(self.tests):
            raise HardError(f"{where}: name each test once, at least one")
        if self.expect == MUST_SURVIVE and "axis:" not in self.note:
            raise HardError(f"{where}: a control must say which axis it varies (``axis:`` in its note)")
        if self.kind in ("text", "hex"):
            if not self.find:
                raise HardError(f"{where}: empty anchor")
            if self.find == self.replace:
                raise HardError(f"{where}: the anchor and its replacement are the same")
            if self.kind == "hex":
                try:
                    bytes.fromhex(self.find), bytes.fromhex(self.replace)
                except ValueError as exc:
                    raise HardError(f"{where}: a hex anchor or replacement is not hex ({exc})") from exc
        elif self.kind in TRANSFORMS:
            if self.find or self.replace:
                raise HardError(f"{where}: a whole-file rewrite takes no anchor")
        else:
            raise HardError(f"{where}: unknown kind {self.kind!r}")
        if (PLANT in self.replace) != bool(self.plant_entry):
            raise HardError(f"{where}: a planted word needs both the {PLANT} mark and the digest of its entry")
        if self.plant_entry and (self.kind != "text" or self.plant_match not in ("exact", "stem")
                                 or not re.fullmatch(r"[0-9a-f]{64}", self.plant_entry)
                                 or not re.fullmatch(r"[0-9a-f]{64}", self.plant_rule)):
            raise HardError(f"{where}: a plant is a text row with two sha256 digests and a match")
        for test in self.tests:
            if self.runner == "vitest":
                file_part, sep, name = test.partition("::")
                if not (sep and file_part and name.strip()) or REGEX_MARKS & set(name):
                    raise HardError(f"{where}: a vitest test is '<file>::<name>', with no regex marks in "
                                    f"the name: {test!r}")
            elif test.startswith(("/", "./", "-")):
                raise HardError(f"{where}: a pytest node id is relative to the root: {test!r}")
        dotted = [_dotted(test) for test in self.tests] if self.runner == "pytest" else []
        for a in dotted:
            if any(_belongs(b, a) for b in dotted if b != a):
                raise HardError(f"{where}: one named test contains another: {self.tests}")

    @property
    def planted(self) -> bool:
        return bool(self.plant_entry)


SPLITS = "src/sightline/data/splits.py"
HONEST = "tests/test_splits.py::test_honest_protocols_keep_every_eval_frame_501_frames_from_training"
BLOCKED_SIZES = "tests/test_splits.py::test_blocked_split_sizes_on_drive2018_length"
HISTORY = "tests/test_frames.py::test_history_never_looks_into_the_future"
DOCS_NUMBERS = "tests/test_docs_numbers.py::test_every_printed_figure_is_the_reports_figure_in_its_phrase"
RESULTS_SYNC = ("tests/test_site_data.py::"
                "test_the_committed_results_json_matches_the_committed_reports_and_the_drives_when_they_are_here")
WORD_GUARD = "tests/test_guards.py::test_no_banned_words_in_tracked_files"
#: the guard's digests the plants exercise, and the entries of the readable list they plant
PROCESS_WORD = "94a168d2da574b00ec6c787377278465271cf2b97e68af076fc3eee87eba8664"  # a stem, kind process
HISTORY_STEM = "76cfec1c272a6aa7f2e559a0d72bf106db3e74ab77f8b70c32955769422edf0b"  # a stem, kind history
FLAG_WORD = "7fef9479c8bc5a3c86891cb82969cbf2ffc73c7350366c82286345d8292eebf5"  # a word allowed as a flag

#: the body of blocked_split's docstring as it stands, and with its first and last sentences
#: swapped, which say independent things
BLOCKED_DOC = (
    "    The drive is cut into ``block``-frame blocks labelled by repeating\n"
    "    ``pattern`` (T train, V validation, E test). ``firewall`` frames are\n"
    "    dropped on BOTH sides of every boundary between a training block and an\n"
    "    evaluation block, so every evaluation frame is at least\n"
    "    ``2 * firewall + 1`` frames from a training frame — the same guarantee as\n"
    "    the forward split — while both sides of the split see the same mix of\n"
    "    roads. Validation and test blocks are not firewalled from each other; the\n"
    "    default pattern never places them adjacent.\n"
)
BLOCKED_DOC_REORDERED = (
    "    Validation and test blocks are not firewalled from each other; the\n"
    "    default pattern never places them adjacent. ``firewall`` frames are\n"
    "    dropped on BOTH sides of every boundary between a training block and an\n"
    "    evaluation block, so every evaluation frame is at least\n"
    "    ``2 * firewall + 1`` frames from a training frame — the same guarantee as\n"
    "    the forward split — while both sides of the split see the same mix of\n"
    "    roads. The drive is cut into ``block``-frame blocks labelled by repeating\n"
    "    ``pattern`` (T train, V validation, E test).\n"
)

CLAIMS: list[Entry] = [
    # -- the protocols ---------------------------------------------------------------------------
    Entry(
        id="1",
        claim="the blocked protocol keeps every evaluation frame at least 501 frames from training",
        path=SPLITS,
        find="        return blocked_split(n, block=block, firewall=firewall)\n",
        replace="        return random_split(n, seed=seed)\n",
        runner="pytest",
        tests=(f"{HONEST}[blocked]",),
        expect=MUST_FAIL,
        note="make_split's blocked branch hands back the shuffled split",
    ),
    Entry(
        id="2",
        claim="the blocked split drops a firewall of frames at every train/evaluation boundary",
        path=SPLITS,
        find="            keep[max(0, edge - firewall):min(n, edge + firewall)] = False\n",
        replace="            pass\n",
        runner="pytest",
        tests=(f"{HONEST}[blocked]", BLOCKED_SIZES),
        expect=MUST_FAIL,
        note="the firewall removal becomes a no-op: neighbouring frames stay on both sides",
    ),
    Entry(
        id="3",
        claim="the forward protocol keeps every evaluation frame at least 501 frames from training",
        path=SPLITS,
        find=("        train=np.arange(0, train_end - firewall, dtype=np.int64),\n"
              "        val=np.arange(train_end + firewall, val_end - firewall, dtype=np.int64),\n"
              "        test=np.arange(val_end + firewall, n, dtype=np.int64),\n"),
        replace=("        train=np.arange(0, train_end, dtype=np.int64),\n"
                 "        val=np.arange(train_end, val_end, dtype=np.int64),\n"
                 "        test=np.arange(val_end, n, dtype=np.int64),\n"),
        runner="pytest",
        tests=(f"{HONEST}[forward]",),
        expect=MUST_FAIL,
        note="chronological_split loses its firewall offsets on both boundaries",
    ),
    Entry(
        id="4",
        claim="the paired set is exactly the frames both protocols hold out",
        path=SPLITS,
        find="    return np.intersect1d(blocked.test, leaky.test).astype(np.int64)\n",
        replace="    return np.asarray(leaky.test, dtype=np.int64)\n",
        runner="pytest",
        tests=(
            "tests/test_splits.py::test_paired_frames_are_in_both_test_sets_and_nowhere_in_either_training_set",
            "tests/test_splits.py::test_the_default_rules_make_the_paired_frames_the_committed_reports_name",
        ),
        expect=MUST_FAIL,
        note=(
            "paired_frames returns the shuffled test set alone. The paired-frames test of "
            "test_reporting.py is not named here: it builds its expected frames with paired_frames "
            "itself, so it agrees with any definition of the set; row 4b holds it to what it checks"
        ),
    ),
    Entry(
        id="4b",
        claim="the blocked and the shuffled run reports score the same paired frames",
        path="src/sightline/reporting.py",
        find='            make_split("leaky", n, seed=seed),\n',
        replace='            make_split("leaky", n, seed=seed + (protocol == "blocked")),\n',
        runner="pytest",
        tests=("tests/test_reporting.py::test_paired_scores_use_identical_frames_for_both_protocols",),
        expect=MUST_FAIL,
        note="the blocked run pairs against the next seed's shuffled split, so the two reports score other frames",
    ),
    # -- the inputs and the labels ---------------------------------------------------------------
    Entry(
        id="5",
        claim="a history frame never comes from the future",
        path="src/sightline/data/frames.py",
        find="np.maximum(idx - lag, 0)",
        replace="np.maximum(idx + lag, 0)",
        runner="pytest",
        tests=(HISTORY,),
        expect=MUST_FAIL,
        note="history indices reach forward by the lag instead of back",
    ),
    Entry(
        id="6",
        claim="a dropout is repaired only between two large angles of one sign",
        path="src/sightline/data/labels.py",
        find="    mask = is_zero & prev_ok & next_ok & same_sign\n",
        replace="    mask = is_zero & prev_ok & next_ok\n",
        runner="pytest",
        tests=("tests/test_labels.py::test_genuine_zeros_are_left_alone",),
        expect=MUST_FAIL,
        note="drops the same-sign condition, so a real zero crossing is interpolated away",
    ),
    # -- the evaluation and the choice of the shipped model ---------------------------------------
    Entry(
        id="7",
        claim="the zero-order-hold baseline copies the previous frame's label",
        path="src/sightline/evaluate.py",
        find='        "zoh": angles[idx - 1],\n',
        replace='        "zoh": angles[idx],\n',
        runner="pytest",
        tests=("tests/test_evaluate.py::test_zoh_and_extrapolation_read_only_the_past",),
        expect=MUST_FAIL,
        note="the baseline reads the label of the frame it predicts",
    ),
    Entry(
        id="8",
        claim="the shipped model is chosen on validation error, never on test error",
        path="src/sightline/reporting.py",
        find='SELECTED_BY = "val_mae_deg"',
        replace='SELECTED_BY = "test_mae_deg"',
        runner="pytest",
        tests=("tests/test_reporting.py::test_shipped_model_is_chosen_on_validation_not_test",),
        expect=MUST_FAIL,
        note="the seed is picked by its test error",
    ),
    # -- the models ------------------------------------------------------------------------------
    Entry(
        id="9",
        claim="PilotNet's atan head cannot steer past half a turn",
        path="src/sightline/models/pilotnet.py",
        find="2.0 * torch.atan(",
        replace="(",
        runner="pytest",
        tests=("tests/test_models.py::test_pilotnet_cannot_steer_past_half_a_turn",),
        expect=MUST_FAIL,
        note="the output head returns its raw value in radians, unbounded",
    ),
    Entry(
        id="10",
        claim="each saliency map is scaled onto 0 to 1, sample by sample",
        path="src/sightline/saliency.py",
        find="    normed = (flat - lo) / (hi - lo + 1e-12)\n",
        replace="    normed = flat\n",
        runner="pytest",
        tests=("tests/test_models.py::"
               "test_visual_backprop_scales_each_map_onto_zero_to_one_whatever_the_activations_scale",),
        expect=MUST_FAIL,
        note=(
            "the relevance leaves unnormalised. The wrapper test's 0-to-1 bounds are not named: they "
            "catch this only because a randomly initialised SightlineNet's relevance dips about "
            "3.5e-13 below zero, and PilotNet's raw relevance already lies inside them"
        ),
    ),
    # -- the published numbers and the site data --------------------------------------------------
    Entry(
        id="11",
        claim="every figure the README and the model card print is the reports' figure",
        path="reports/summary.json",
        find='"mean": 3.0465626341557503,',
        replace='"mean": 3.0565626341557503,',
        runner="pytest",
        tests=(DOCS_NUMBERS,),
        expect=MUST_FAIL,
        note="SightlineNet's paired error ratio moves by one printed unit, from 3.05 to 3.06",
    ),
    Entry(
        id="12",
        claim="the site's results.json holds the reports' values",
        path="web/public/data/results.json",
        find='"mean": 3.0465626341557503,',
        replace='"mean": 3.0565626341557503,',
        runner="pytest",
        tests=(RESULTS_SYNC,),
        expect=MUST_FAIL,
        note="the page's copy of SightlineNet's paired error ratio drifts from summary.json",
    ),
    Entry(
        id="13",
        claim="every committed sprite decodes to the frame hashes its clip records",
        path="web/public/data/clips/c2018/sprite-07.jpg",
        kind="hex",
        find="ffdb00430002",
        replace="ffdb00430003",
        runner="pytest",
        tests=("tests/test_site_data.py::test_the_committed_sprites_decode_to_their_recorded_hashes",),
        expect=MUST_FAIL,
        note=(
            "one byte: the luminance quantisation table's DC step goes from 2 to 3, so the file still "
            "decodes, to other pixels"
        ),
    ),
    Entry(
        id="14",
        claim="the page credits its author visibly in the footer",
        path="web/index.html",
        find='          <a href="https://github.com/safdar-hussain1">Built by Safdar Hussain</a>\n',
        replace="",
        runner="pytest",
        tests=("tests/test_authorship_marks.py::test_the_footer_credits_the_author_visibly[template]",),
        expect=MUST_FAIL,
        note="the footer credit leaves the page template",
    ),
    Entry(
        id="15",
        claim="the self-test's verdict starts with PASS only when every check passed",
        path="web/src/selftest.ts",
        find="  if (total > 0 && failed === undefined && threads === 1) {",
        replace="  if (!(total > 0 && failed === undefined && threads === 1)) {",
        runner="vitest",
        tests=("src/selftest.test.ts::never starts a failing verdict with PASS",),
        expect=MUST_FAIL,
        note="the PASS branch is taken exactly when the checks did not all pass",
    ),
    Entry(
        id="16",
        claim="the drive profile is drawn from the repaired labels",
        path="src/sightline/export/site_data.py",
        find="    profiles = {name: drive_profile(labels) for name, labels in sorted(drives.items())}\n",
        replace=(
            '    profiles = {name: drive_profile(__import__("sightline.data.labels", fromlist=["load_labels"])'
            ".load_labels(labels.root))\n"
            "                for name, labels in sorted(drives.items())}\n"
        ),
        runner="pytest",
        tests=("tests/test_site_data.py::test_the_drive_profile_is_the_repaired_labels_averaged_over_buckets",),
        expect=MUST_FAIL,
        note="the profile reads each label file again without the dropout repair",
    ),
    Entry(
        id="17",
        claim="the split rules the site states are the splits module's",
        path=SPLITS,
        find='BLOCK_PATTERN = "TTVTTET"',
        replace='BLOCK_PATTERN = "TTETTVT"',
        runner="pytest",
        tests=("tests/test_site_data.py::"
               "test_the_committed_split_rules_and_sizes_are_the_splits_modules_and_the_run_reports",),
        expect=MUST_FAIL,
        note=(
            "the validation and test blocks swap places: every split size stays the same, so only the "
            "stated pattern can tell"
        ),
    ),
    # -- the public-text guard -------------------------------------------------------------------
    Entry(
        id="18",
        claim="the guard refuses a build-process word in a tracked document",
        path="web/public/model-card.md",
        find="# Sightline model card\n\n",
        replace=f"# Sightline model card\n\nThis card was checked by a {PLANT} before it shipped.\n\n",
        runner="pytest",
        tests=(WORD_GUARD,),
        expect=MUST_FAIL,
        note="a process word from the private list goes into the model card",
        plant_entry=PROCESS_WORD,
        plant_rule=PROCESS_WORD,
        plant_match="stem",
    ),
    Entry(
        id="19",
        claim="the guard refuses a word that tells the project's history in README prose",
        path="README.md",
        find="This is research software. It must never be used to steer, assist or control a vehicle.\n",
        replace=(
            "This is research software. It must never be used to steer, assist or control a vehicle.\n"
            f"Each release is an {PLANT} on the last.\n"
        ),
        runner="pytest",
        tests=(WORD_GUARD,),
        expect=MUST_FAIL,
        note="a history word from the private list, an inflection its stem refuses, goes into the README",
        plant_entry=FLAG_WORD,
        plant_rule=HISTORY_STEM,
        plant_match="stem",
    ),
    Entry(
        id="20",
        claim="the page's markup types no number: every figure comes from the data",
        path="web/src/sections/markup.ts",
        find="It does not drive.</p>",
        replace="It does not drive, at 30 frames a second.</p>",
        runner="vitest",
        tests=("src/sections/markup.test.ts::prints no number of its own",),
        expect=MUST_FAIL,
        note="a visible figure is typed into How it works",
    ),
    # -- the controls ----------------------------------------------------------------------------
    Entry(
        id="a",
        claim="control: blocked_split's docstring with two sentences swapped",
        path=SPLITS,
        find=BLOCKED_DOC,
        replace=BLOCKED_DOC_REORDERED,
        runner="pytest",
        tests=(f"{HONEST}[blocked]", BLOCKED_SIZES),
        expect=MUST_SURVIVE,
        note="axis: prose. Rows 1 and 2 change what the split does; this changes only how it is described",
    ),
    Entry(
        id="b",
        claim="control: summary.json written with an indent of four",
        path="reports/summary.json",
        kind="json-indent-4",
        runner="pytest",
        tests=(DOCS_NUMBERS,),
        expect=MUST_SURVIVE,
        note="axis: whitespace. Row 11 moves a value; this moves only the layout the value is written in",
    ),
    Entry(
        id="c",
        claim="control: a local variable of history_indices renamed",
        path="src/sightline/data/frames.py",
        find=("    cols = [idx] + [np.maximum(idx - lag, 0) for lag in lags]\n"
              "    return np.concatenate(cols, axis=1)\n"),
        replace=("    columns = [idx] + [np.maximum(idx - lag, 0) for lag in lags]\n"
                 "    return np.concatenate(columns, axis=1)\n"),
        runner="pytest",
        tests=(HISTORY,),
        expect=MUST_SURVIVE,
        note="axis: an identifier. Row 5 changes which frame is read; this changes only a name",
    ),
    Entry(
        id="d",
        claim="control: results.json written with every object's keys in reverse order",
        path="web/public/data/results.json",
        kind="json-keys-reversed",
        runner="pytest",
        tests=(RESULTS_SYNC,),
        expect=MUST_SURVIVE,
        note=(
            "axis: key order. Row 12 changes a value; this changes only the order the keys are written "
            "in (the file is written with sorted keys, so sorting them again would change nothing)"
        ),
    ),
    Entry(
        id="e",
        claim="control: the history word of row 19 written as a command-line flag",
        path="README.md",
        find=".venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu\n",
        replace=f".venv/bin/pip install --{PLANT} torch --index-url https://download.pytorch.org/whl/cpu\n",
        runner="pytest",
        tests=(WORD_GUARD,),
        expect=MUST_SURVIVE,
        note=(
            "axis: the word's role. Row 19 puts the word in prose; here it stands as a pip flag, which "
            "the guard allows, so a guard that refused the word everywhere would go red"
        ),
        plant_entry=FLAG_WORD,
        plant_rule=FLAG_WORD,
        plant_match="exact",
    ),
]


# --- planted words ------------------------------------------------------------------------------


def normal_form(text: str) -> str:
    """The guard's normal form: ASCII letter-and-digit tokens (and section signs) in lower case, a
    run of digits with up to two letters after it written ``#``, joined by single spaces."""
    return " ".join("#" if re.fullmatch(r"[0-9]+[A-Za-z]{0,2}", token) else token.lower()
                    for token in re.findall(r"[A-Za-z0-9]+|§", text))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def list_entries(text: str) -> dict[str, str]:
    """The readable list's entries by their digest: every table row ``| entry | ... | `<sha256>` |``."""
    entries = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and re.fullmatch(r"`?[0-9a-f]{64}`?", cells[-1]):
            entries[cells[-1].strip("`")] = cells[0].strip("`")
    return entries


def resolve_plant(root: Path, entry: Entry, env: dict) -> tuple[str | None, str]:
    """``(word, "")`` for a planted row, or ``(None, why it is skipped)``. Raises a HardError when
    the list is given but cannot back the plant: the entry is missing, does not hash to its digest,
    or the guard rule the row exercises does not match it or is not in the guard file."""
    source = env.get(GUARD_WORDS_ENV, "")
    if not source:
        return None, (f"{GUARD_WORDS_ENV} is not set: this row plants a word the guard keeps as a digest, "
                      "and only the readable list kept outside the repository spells it")
    path = Path(source).expanduser()
    if not path.is_file():
        raise HardError(f"row {entry.id!r}: {GUARD_WORDS_ENV} names no file")
    word = list_entries(path.read_text(encoding="utf-8")).get(entry.plant_entry)
    if word is None:
        raise HardError(f"row {entry.id!r}: the list has no entry with digest {entry.plant_entry[:12]}")
    normal = normal_form(word)
    if not normal or _sha(normal) != entry.plant_entry:
        raise HardError(f"row {entry.id!r}: the list's entry for {entry.plant_entry[:12]} does not hash to it")
    if entry.plant_match == "exact":
        matched = _sha(normal) == entry.plant_rule
    else:
        matched = any(_sha(normal[:k]) == entry.plant_rule for k in range(4, len(normal) + 1))
    if not matched:
        raise HardError(f"row {entry.id!r}: the guard rule {entry.plant_rule[:12]} does not match the "
                        "planted word, so the row would test nothing")
    guard = root / entry.plant_guard
    if not guard.is_file() or entry.plant_rule not in guard.read_text(encoding="utf-8"):
        raise HardError(f"row {entry.id!r}: {entry.plant_guard} does not hold the rule "
                        f"{entry.plant_rule[:12]} this row exercises")
    return word, ""


# --- the mutation -------------------------------------------------------------------------------


@dataclass
class Plan:
    """A row made ready: the file's bytes now, the mutated bytes, or why the row is skipped."""

    entry: Entry
    before: bytes = b""
    after: bytes = b""
    skipped: str = ""
    #: the word a planted row writes; never printed
    word: str | None = field(default=None, repr=False)


def _anchor(entry: Entry, word: str | None) -> tuple[bytes, bytes]:
    if entry.kind == "hex":
        return bytes.fromhex(entry.find), bytes.fromhex(entry.replace)
    replace = entry.replace if word is None else entry.replace.replace(PLANT, word)
    return entry.find.encode("utf-8"), replace.encode("utf-8")


def check_anchor(entry: Entry, data: bytes) -> int:
    """The offset of the row's anchor, which must occur exactly once (overlapping matches count)."""
    find, _ = _anchor(entry, None)
    first = data.find(find)
    if first < 0:
        raise HardError(
            f"row {entry.id!r} ({entry.claim}): the anchor is NOT PRESENT in {entry.path}. The row would "
            f"mutate nothing; take the anchor from the file again:\n  {entry.find[:200]!r}")
    count = 1
    at = data.find(find, first + 1)
    while at >= 0:
        count += 1
        at = data.find(find, at + 1)
    if count > 1:
        raise HardError(
            f"row {entry.id!r} ({entry.claim}): the anchor occurs {count} times in {entry.path}. An "
            f"ambiguous anchor mutates more than the row says; widen it:\n  {entry.find[:200]!r}")
    return first


def mutate(entry: Entry, before: bytes, word: str | None) -> bytes:
    """The file's bytes under the row's mutation; a hard error unless they really change."""
    if entry.kind in TRANSFORMS:
        try:
            after = TRANSFORMS[entry.kind](before)
            same = json.loads(after) == json.loads(before)
        except ValueError as exc:
            raise HardError(f"row {entry.id!r}: {entry.path} is not JSON ({exc})") from exc
        if not same:
            raise HardError(f"row {entry.id!r}: the rewrite of {entry.path} changed its value")
    else:
        if entry.planted and not word:
            raise HardError(f"row {entry.id!r}: no word to plant")
        at = check_anchor(entry, before)
        find, replace = _anchor(entry, word)
        after = before[:at] + replace + before[at + len(find):]
    if after == before:
        raise HardError(f"row {entry.id!r}: the mutation changed NO bytes of {entry.path}")
    return after


def plan_rows(root: Path, entries: list[Entry], env: dict) -> list[Plan]:
    """Every row made ready before any file is written: each anchor found exactly once, each
    mutation's bytes computed and different, each planted word resolved, or the row skipped."""
    plans = []
    for entry in entries:
        path = root / entry.path
        if not path.is_file():
            raise HardError(f"row {entry.id!r}: {entry.path} does not exist")
        before = path.read_bytes()
        word, skipped = resolve_plant(root, entry, env) if entry.planted else (None, "")
        if skipped:
            check_anchor(entry, before)
            plans.append(Plan(entry, before=before, skipped=skipped))
            continue
        after = mutate(entry, before, word)
        if word is not None and word.encode() in before:
            raise HardError(f"row {entry.id!r}: {entry.path} holds the planted word already")
        plans.append(Plan(entry, before=before, after=after, word=word))
    return plans


# --- running the named tests -------------------------------------------------------------------


@dataclass
class Outcome:
    """What one named test did in one run: its cases passed, failed, errored and skipped."""

    test: str
    passed: int = 0
    failed: int = 0
    errored: int = 0
    skipped: int = 0

    @property
    def red(self) -> bool:
        return self.failed + self.errored > 0

    @property
    def green(self) -> bool:
        return not self.red and self.passed > 0

    def as_dict(self) -> dict:
        return {"test": self.test, "passed": self.passed, "failed": self.failed,
                "errored": self.errored, "skipped": self.skipped, "red": self.red}


@dataclass
class Run:
    outcomes: list[Outcome]
    seconds: float


def _junit_cases(path: Path) -> list[tuple[str, str]]:
    """``(classname.name, status)`` for every case of a JUnit report: passed, failed, errored or skipped.
    The report is the one pytest just wrote into a temporary folder this battery made."""
    if not path.is_file():
        return []
    root = ElementTree.parse(path).getroot()
    cases = []
    for case in root.iter("testcase"):
        tags = {child.tag for child in case}
        status = ("failed" if "failure" in tags else "errored" if "error" in tags
                  else "skipped" if "skipped" in tags else "passed")
        name = case.get("name", "")
        classname = case.get("classname", "")
        cases.append((f"{classname}.{name}" if classname else name, status))
    return cases


class Runner:
    """Runs a row's named tests, each run with a fresh ``--basetemp`` under one folder outside the
    repository, and an empty bytecode cache of its own."""

    def __init__(self, root: Path, basetemp: Path):
        self.root, self.basetemp, self.count = root, basetemp, 0

    def _pytest_env(self, pycache: str) -> dict:
        env = dict(os.environ)
        env.pop("PYTEST_ADDOPTS", None)
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(self.root / "src"), str(self.root), env.get("PYTHONPATH", "")) if part)
        # The cached bytecode of a clean file stays valid for a mutated one of the same size written
        # in the same second (the check is size and whole-second mtime), so every run compiles the
        # bytes on disk: an empty cache of its own, and nothing written to it or beside the sources.
        env["PYTHONPYCACHEPREFIX"] = pycache
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def run(self, entry: Entry, phase: str, planted: str | None = None) -> Run:
        """The named tests of ``entry``, run once. A hard error's message never carries ``planted``,
        the word a planted row wrote into the file, though the test's own output does."""
        start = time.monotonic()
        try:
            outcomes = self._pytest(entry, phase) if entry.runner == "pytest" else self._vitest(entry)
        except HardError as exc:
            if planted and planted in str(exc):
                raise HardError(str(exc).replace(planted, "<the planted word>")) from None
            raise
        return Run(outcomes, round(time.monotonic() - start, 1))

    def _pytest(self, entry: Entry, phase: str) -> list[Outcome]:
        self.count += 1
        basetemp = self.basetemp / f"{self.count:03d}-{re.sub(r'[^A-Za-z0-9]+', '-', entry.id)}-{phase}"
        try:
            with tempfile.TemporaryDirectory(prefix="pyc-") as pycache, \
                    tempfile.TemporaryDirectory(prefix="junit-") as tmp:
                report = Path(tmp) / "junit.xml"
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", *entry.tests, "-p", "no:cacheprovider",
                     "--basetemp", str(basetemp), "--junit-xml", str(report), "-o", "junit_family=xunit2"],
                    cwd=self.root, env=self._pytest_env(pycache), capture_output=True, text=True)
                cases = _junit_cases(report)
        finally:
            shutil.rmtree(basetemp, ignore_errors=True)
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-30:])
        if proc.returncode not in (0, 1):
            raise HardError(
                f"row {entry.id!r} ({phase}): pytest exited {proc.returncode} and could not run "
                f"{list(entry.tests)} (2: a module failed to load or the run was interrupted; 3: an "
                f"internal error; 4: a usage error; 5: nothing collected). That is not a kill:\n{tail}")
        dotted = {test: _dotted(test) for test in entry.tests}
        outcomes = {test: Outcome(test) for test in entry.tests}
        for case, status in cases:
            owners = [test for test, name in dotted.items() if _belongs(case, name)]
            if len(owners) != 1:
                raise HardError(f"row {entry.id!r} ({phase}): the report's case {case!r} belongs to "
                                f"{len(owners)} of the named tests")
            outcome = outcomes[owners[0]]
            setattr(outcome, status, getattr(outcome, status) + 1)
        for outcome in outcomes.values():
            if outcome.passed + outcome.failed + outcome.errored + outcome.skipped == 0:
                raise HardError(f"row {entry.id!r} ({phase}): {outcome.test} ran no case at all")
        if (proc.returncode == 1) != any(outcome.red for outcome in outcomes.values()):
            raise HardError(f"row {entry.id!r} ({phase}): pytest exited {proc.returncode}, which its own "
                            f"report contradicts:\n{tail}")
        return list(outcomes.values())

    def _vitest(self, entry: Entry) -> list[Outcome]:
        outcomes = []
        for test in entry.tests:
            file_path, _, name = test.partition("::")
            with tempfile.TemporaryDirectory(prefix="vitest-") as tmp:
                out = Path(tmp) / "vitest.json"
                proc = subprocess.run(
                    ["npx", "vitest", "run", file_path, "-t", name, "--reporter=json", f"--outputFile={out}"],
                    cwd=self.root / "web", capture_output=True, text=True)
                if not out.is_file():
                    raise HardError(f"row {entry.id!r}: vitest wrote no report for {test!r} (exit "
                                    f"{proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
                report = json.loads(out.read_text())
            files = report.get("testResults", [])
            if len(files) != 1 or (files[0].get("status") == "failed" and not files[0].get("assertionResults")):
                message = files[0].get("message", "") if files else "no test file ran"
                raise HardError(f"row {entry.id!r}: {file_path} did not load under vitest, which is not a "
                                f"kill:\n{message[:2000]}")
            outcome = Outcome(test)
            for case in files[0].get("assertionResults", []):
                status = case.get("status")
                if status == "passed":
                    outcome.passed += 1
                elif status == "failed":
                    outcome.failed += 1
                else:
                    outcome.skipped += 1
            if outcome.passed + outcome.failed == 0:
                raise HardError(f"row {entry.id!r}: the vitest filter {name!r} matches no test of {file_path}")
            if proc.returncode not in (0, 1) or (proc.returncode == 1) != outcome.red:
                raise HardError(f"row {entry.id!r}: vitest exited {proc.returncode} for {test!r}, which its "
                                f"report contradicts:\n{proc.stderr[-2000:]}")
            outcomes.append(outcome)
        return outcomes


# --- one row ------------------------------------------------------------------------------------


@contextlib.contextmanager
def _signals_held():
    """Ctrl-C and SIGTERM wait until the block is done, so a write is never cut in half."""
    if not hasattr(signal, "pthread_sigmask"):
        yield
        return
    old = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)


def _write_checked(path: Path, data: bytes, what: str) -> str:
    """Write ``data`` to ``path`` and read it back; the sha256 of what is now on disk."""
    with _signals_held():
        path.write_bytes(data)
        back = path.read_bytes()
        digest = hashlib.sha256(back).hexdigest()
        if back != data or digest != hashlib.sha256(data).hexdigest():
            raise HardError(f"{what} FAILED for {path}: the bytes on disk are not the bytes written. "
                            "Inspect the file before anything else")
    return digest


def changed_span(before: bytes, after: bytes) -> tuple[int, int, int]:
    """Where two versions of a file differ: ``(offset, bytes removed, bytes put in)``, the one
    stretch between their longest common start and their longest common end."""
    limit = min(len(before), len(after))
    start = next((i for i in range(limit) if before[i] != after[i]), limit)
    end = 0
    while end < limit - start and before[len(before) - 1 - end] == after[len(after) - 1 - end]:
        end += 1
    return start, len(before) - start - end, len(after) - start - end


@dataclass
class Result:
    entry: Entry
    #: "killed" (every named test red), "partly killed", "survived" (none red) or "skipped"
    outcome: str
    reason: str = ""
    seconds: float = 0.0
    mutated: list[Outcome] = field(default_factory=list)
    clean_again: list[Outcome] = field(default_factory=list)
    bytes_before: int = 0
    bytes_after: int = 0
    #: the stretch the mutation changed: where it starts, the bytes taken out and those put in
    changed_at: int = -1
    removed_bytes: int = 0
    inserted_bytes: int = 0
    sha256_before: str = ""
    sha256_restored: str = ""

    @property
    def ok(self) -> bool:
        if self.outcome == "skipped":
            return True
        return self.outcome == ("killed" if self.entry.expect == MUST_FAIL else "survived")

    @property
    def red_tests(self) -> list[str]:
        return [outcome.test for outcome in self.mutated if outcome.red]

    @property
    def byte_change(self) -> str:
        return (f"{self.bytes_before} -> {self.bytes_after} bytes: {self.removed_bytes} at offset "
                f"{self.changed_at} replaced by {self.inserted_bytes}")

    def as_dict(self) -> dict:
        entry = self.entry
        return {
            "id": entry.id, "claim": entry.claim, "expect": entry.expect, "path": entry.path,
            "kind": entry.kind, "runner": entry.runner, "tests": list(entry.tests), "note": entry.note,
            "planted_rule": entry.plant_rule or None, "outcome": self.outcome, "ok": self.ok,
            "reason": self.reason, "seconds": self.seconds, "red_tests": self.red_tests,
            "mutated": [o.as_dict() for o in self.mutated],
            "clean_again": [o.as_dict() for o in self.clean_again],
            "bytes_before": self.bytes_before, "bytes_after": self.bytes_after,
            "changed_at": self.changed_at, "removed_bytes": self.removed_bytes,
            "inserted_bytes": self.inserted_bytes,
            "sha256_before": self.sha256_before, "sha256_restored": self.sha256_restored,
        }


def run_row(root: Path, plan: Plan, runner: Runner) -> Result:
    entry = plan.entry
    if plan.skipped:
        return Result(entry, "skipped", reason=plan.skipped)
    path = root / entry.path
    start = time.monotonic()
    before = path.read_bytes()
    if before != plan.before:
        raise HardError(f"row {entry.id!r}: {entry.path} changed after the run began")
    sha_before = hashlib.sha256(before).hexdigest()
    try:
        _write_checked(path, plan.after, f"row {entry.id!r}: the MUTATION")
        mutated = runner.run(entry, "mutated", plan.word)
    finally:
        restored = _write_checked(path, before, f"row {entry.id!r}: the RESTORE")
        if restored != sha_before:
            raise HardError(f"row {entry.id!r}: RESTORE FAILED for {entry.path}: sha256 {restored[:12]}, "
                            f"saved {sha_before[:12]}")
    again = runner.run(entry, "clean-again", plan.word)
    if not all(outcome.green for outcome in again.outcomes):
        raise HardError(f"row {entry.id!r}: the named tests are not green again on the restored "
                        f"{entry.path}, whose bytes are back: something else in the tree moved "
                        f"({[o.as_dict() for o in again.outcomes]})")
    red = sum(outcome.red for outcome in mutated.outcomes)
    outcome = "killed" if red == len(entry.tests) else "partly killed" if red else "survived"
    at, removed, inserted = changed_span(before, plan.after)
    return Result(
        entry, outcome, seconds=round(time.monotonic() - start, 1), mutated=mutated.outcomes,
        clean_again=again.outcomes, bytes_before=len(before), bytes_after=len(plan.after),
        changed_at=at, removed_bytes=removed, inserted_bytes=inserted, sha256_before=sha_before,
        sha256_restored=restored)


# --- the run ------------------------------------------------------------------------------------


IGNORED_LIMIT = ("What this check does not cover: `git status --porcelain` does not see git-ignored "
                 "paths, so it says nothing about data/, runs/, node_modules/ and the like. No row "
                 "touches one; a --table of your own could.")


def git_status(root: Path) -> str:
    proc = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HardError(f"--root {root} is not a usable git repository (`git status --porcelain` exited "
                        f"{proc.returncode}: {proc.stderr.strip() or 'no output'}); without git there is no "
                        "way to show that the tree came back")
    return proc.stdout


def check_clean(root: Path, allowed: set[str]) -> str:
    """The porcelain status the run starts from: clean, apart from the untracked paths allowed."""
    status = git_status(root)
    dirty = [line for line in status.splitlines()
             if line.strip() and not (line.startswith("?? ") and line[3:] in allowed)]
    if dirty:
        raise HardError("REFUSING TO RUN: the tree is not clean. This battery edits tracked files in place, "
                        "and a crash with uncommitted work present could take it with it. Commit or stash "
                        f"first, or name an untracked path with --allow-untracked. {IGNORED_LIMIT}\n"
                        + "\n".join(dirty))
    return status


def _outside(root: Path, path: Path, what: str) -> Path:
    path = path.expanduser().resolve()
    if path == root or root in path.parents:
        raise HardError(f"{what} {path} is inside the repository; it would leave the tree changed")
    return path


def _head(root: Path) -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def load_table(path: Path) -> list[Entry]:
    return [Entry(**row) for row in json.loads(path.read_text())]


def select(entries: list[Entry], only: list[str]) -> list[Entry]:
    """The rows ``only`` names, by id or by a piece of the claim, in table order."""
    wanted = [item.strip() for group in only for item in group.split(",") if item.strip()]
    chosen = [entry for entry in entries
              if any(item == entry.id or item.lower() in entry.claim.lower() for item in wanted)]
    if not chosen:
        raise HardError(f"no row matches --only {wanted}")
    return chosen


def print_table(entries: list[Entry], env: dict) -> None:
    for entry in entries:
        planted = ""
        if entry.planted:
            planted = " (plants a listed word" + ("" if env.get(GUARD_WORDS_ENV) else
                                                  f"; skipped while {GUARD_WORDS_ENV} is unset") + ")"
        print(f"{entry.id:>3}  {entry.expect:12s} {entry.claim}{planted}")
        print(f"     {entry.kind:12s} {entry.path}")
        for test in entry.tests:
            print(f"     {entry.runner:12s} {test}")
    print(f"\n{len(entries)} rows: {sum(e.expect == MUST_FAIL for e in entries)} must fail, "
          f"{sum(e.expect == MUST_SURVIVE for e in entries)} controls")


def _raise_interrupted(signum, frame):
    raise Interrupted()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(REPO_ROOT), help="the checkout to mutate (default: this one)")
    parser.add_argument("--table", default=None, help="rows as JSON instead of the built-in table")
    parser.add_argument("--only", action="append", default=[], help="row ids or claim pieces, comma-separated")
    parser.add_argument("--list", action="store_true", help="print the rows and exit")
    parser.add_argument("--results", default=None, help="write every row's result to this JSON file")
    parser.add_argument("--basetemp", default=None, help="the folder each pytest run's --basetemp goes under")
    parser.add_argument("--allow-untracked", action="append", default=[], metavar="PATH",
                        help="an untracked path (as git status prints it) the run may start with")
    args = parser.parse_args(argv)
    env = dict(os.environ)

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results: list[Result] = []
    error, code, made_basetemp, basetemp, root = "", 2, False, None, Path(args.root).resolve()
    results_path: Path | None = None
    # Ctrl-C and SIGTERM always stop the run and restore the file, even when the battery was started
    # with SIGINT ignored, as a shell starts a background job of a script
    previous_int = signal.signal(signal.SIGINT, signal.default_int_handler)
    previous_term = signal.signal(signal.SIGTERM, _raise_interrupted)
    try:
        entries = load_table(Path(args.table)) if args.table else list(CLAIMS)
        ids = [entry.id for entry in entries]
        if len(set(ids)) != len(ids):
            raise HardError(f"row ids repeat: {ids}")
        if args.only:
            entries = select(entries, args.only)
            print(f"PARTIAL RUN: --only selected {len(entries)} of {len(ids)} rows. It says nothing about "
                  "the rows left out.", flush=True)
        if args.list:
            print_table(entries, env)
            return 0
        if args.results:
            results_path = _outside(root, Path(args.results), "--results")
        if args.basetemp:
            basetemp = _outside(root, Path(args.basetemp), "--basetemp")
            basetemp.mkdir(parents=True, exist_ok=True)
        else:
            basetemp, made_basetemp = Path(tempfile.mkdtemp(prefix="sightline-battery-")), True
        start_status = check_clean(root, set(args.allow_untracked))
        plans = plan_rows(root, entries, env)
        runner = Runner(root, basetemp)

        print(f"checking that the named tests pass on the clean tree ({len(plans)} rows)", flush=True)
        checked: dict[tuple, Run] = {}
        for plan in plans:
            key = (plan.entry.runner, plan.entry.tests)
            if plan.skipped or key in checked:
                continue
            run = runner.run(plan.entry, "clean", plan.word)
            for outcome in run.outcomes:
                if outcome.red:
                    raise HardError(f"row {plan.entry.id!r}: {outcome.test} is already red on the clean tree "
                                    f"({outcome.as_dict()}); every result of this row would be meaningless")
                if not outcome.green:
                    raise HardError(f"row {plan.entry.id!r}: {outcome.test} passes no case on the clean tree "
                                    f"({outcome.as_dict()}): it only skips, so it can never go red")
            checked[key] = run
            print(f"      clean  {plan.entry.id:>3}  {run.seconds:6.1f}s  "
                  + ", ".join(f"{o.passed} passed" for o in run.outcomes), flush=True)

        for index, plan in enumerate(plans, start=1):
            entry = plan.entry
            print(f"[{index}/{len(plans)}] {entry.id}: {entry.claim}", flush=True)
            result = run_row(root, plan, runner)
            results.append(result)
            if result.outcome == "skipped":
                print(f"      -- skipped: {result.reason}", flush=True)
                continue
            mark = "ok" if result.ok else "!!"
            red = ", ".join(result.red_tests) or "none"
            print(f"      {mark} {result.outcome:13s} expected {entry.expect:12s} {result.seconds:6.1f}s  "
                  f"{result.byte_change}; red: {red}", flush=True)

        end_status = git_status(root)
        if end_status != start_status:
            raise HardError("the tree is not as it started after every file was restored. "
                            f"{IGNORED_LIMIT}\nbefore:\n{start_status}\nafter:\n{end_status}")
        code = report(results)
    except HardError as exc:
        error = str(exc)
        print(f"\nHARD ERROR: {exc}", file=sys.stderr, flush=True)
        code = 2
    except (KeyboardInterrupt, Interrupted):
        error = "interrupted"
        status = ""
        with contextlib.suppress(Exception):
            status = git_status(root)
        print("\nINTERRUPTED: the file being mutated was restored from its byte copy. git status now:\n"
              + (status or "(clean)"), file=sys.stderr, flush=True)
        code = 130
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        if made_basetemp and basetemp is not None:
            shutil.rmtree(basetemp, ignore_errors=True)
    if results_path is not None:
        write_results(results_path, root, started, results, code, error)
    return code


def write_results(path: Path, root: Path, started: str, results: list[Result], code: int, error: str) -> None:
    payload = {
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "head": _head(root),
        "python": platform.python_version(),
        "exit_code": code,
        "error": error,
        "rows": [result.as_dict() for result in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def report(results: list[Result]) -> int:
    ran = [r for r in results if r.outcome != "skipped"]
    must_fail = [r for r in ran if r.entry.expect == MUST_FAIL]
    controls = [r for r in ran if r.entry.expect == MUST_SURVIVE]
    survivors = [r for r in must_fail if not r.ok]
    reddened = [r for r in controls if not r.ok]
    skipped = [r for r in results if r.outcome == "skipped"]

    print("\n" + "=" * 78)
    print(f"{len(results)} rows: {len(ran)} ran, {len(skipped)} skipped")
    print(f"  must-fail rows killed by every named test: {len(must_fail) - len(survivors)}/{len(must_fail)}")
    print(f"  controls that stayed green               : {len(controls) - len(reddened)}/{len(controls)}")
    print("\n  id   expect        outcome         seconds  tests that went red")
    for r in results:
        red = ", ".join(t.rsplit("::", 1)[-1] for t in r.red_tests) or "-"
        print(f"  {r.entry.id:>3}  {r.entry.expect:12s}  {r.outcome:14s} {r.seconds:8.1f}  {red}")
    if survivors:
        print("\nSURVIVORS -- these claims have no test behind them:")
        for r in survivors:
            green = [o.test for o in r.mutated if not o.red]
            print(f"  - {r.entry.id}: {r.entry.claim}\n      mutation: {r.entry.path} ({r.byte_change})\n"
                  f"      named tests that stayed GREEN: {green}\n      the mutation: {r.entry.note}")
    if reddened:
        print("\nCONTROLS THAT WENT RED -- a change that keeps the meaning broke a test, so the test reads "
              "how the file is written, not what it says:")
        for r in reddened:
            print(f"  - {r.entry.id}: {r.entry.claim}\n      red: {r.red_tests}\n      {r.entry.note}")
    if skipped:
        print("\nSKIPPED -- not evidence about these claims either way:")
        for r in skipped:
            print(f"  - {r.entry.id}: {r.entry.claim}\n      {r.reason}")
    failures = len(survivors) + len(reddened)
    if failures:
        print(f"\nFAILED: {failures} row(s) did not behave as the table says.")
        return 1
    print("\nPASSED: every must-fail row that ran was killed by every test it names, and every control "
          "that ran stayed green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
