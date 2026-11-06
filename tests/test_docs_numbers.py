"""Every number README.md and the model card print is read from the reports when the tests run.

The two documents are prose, so nothing regenerates them. What holds a sentence to a measurement
is reading the measurement from the JSON here and requiring the sentence that carries it.

``PINS``
    One row per printed figure: the document, the phrase it sits in with ``{}`` where the figure
    goes, the file it is read from (a report under ``reports/``, or the page's
    ``web/public/data/results.json``), a JSON pointer into that file (RFC 6901: ``~1`` stands for
    a ``/`` inside a key) and the format it is printed in. The value is never written here: the
    test renders it and requires the whole phrase, with the figure standing as a number of its own
    (``3.05`` is not found inside ``13.05`` or ``3.055``).
``ROWS``
    The same for a phrase that carries several figures, such as a table row; each row is checked
    as one pin per figure, the other figures rendered in place, so a row is pinned whole.
``DERIVED``
    The few figures a document computes from report values (a ratio of two errors, the best and the
    worst seed, the closest blocked evaluation frame in time), each formula written out below.
``CLAIMS``
    The sentences whose words depend on the numbers ("about three times", "in every seed",
    "always steering straight wins"), each with the condition on the reports that makes it true.
``ALLOWED``
    The numbers that are not measurements, each with the reason it may be printed: a year naming a
    drive, a version, a command-line setting, a file name, a reference, a name with digits in it
    (a drive, a data type, a chip, a link), a constant of the code (checked against the code), a
    count the code fixes (frames a model reads, its convolutions, the graph's outputs; checked),
    and the times commands took, which no report records (the training times are checked against
    the run folders where they are present). Each entry is anchored to the phrase it allows.

Four checks run over all of them: every number token and number word in either document must sit
inside a pinned figure or an allowed phrase; every figure printed after "at least" must be its value
rounded down, never up; moving any pinned value by one unit of its printed precision must break its
pin; and the reports rewritten with an indent of four must leave every pin holding, so nothing here
reads a report's layout instead of its values.

A number token is read whole, as a reader sees it: digits with their thousands commas, decimals and
exponent (``1,073.2``, ``1.8E-04``, ``.5``), with any letters glued to them (``3x``, ``57ms``,
``20th``), and a word with digits in it (``drive2018``, ``M1``) is a token too, so a name is allowed
by naming it, never by skipping every digit next to a letter. Number words are the cardinals, the
large counts (``hundred``, ``a dozen``), the ordinals from ``third`` on and the multiples (``twice``,
``single``, ``half``); ``first``, ``second``, ``once`` and ``both`` are left out, as the documents use
them for order, a unit of time, a moment and a pair, not a count.

The last tests hold the prose to a shape: lines and code blocks within the wrap, sentences of fifty words at
most, the frame spacing of the whole drive never put to a clip, a rung defined before it is used, and every
excerpt of the dataset that the repository ships named in the licence and the README.
"""
from __future__ import annotations

import copy
import fnmatch
import functools
import inspect
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCUMENTS = {"README": "README.md", "CARD": "web/public/model-card.md"}

SUMMARY = "reports/summary.json"
LEAKAGE = "reports/leakage.json"
LEGACY = "reports/legacy.json"
EXPORT = "reports/export.json"
RESULTS = "web/public/data/results.json"
#: run reports read for what only they record: parameter counts and the training device
SHIPPED_RUN = "reports/runs/drive2018_sightlinenet_blocked_s2.json"
ONE_FRAME_RUN = "reports/runs/drive2018_sightlinenet-1f_blocked_s0.json"
PILOTNET_RUN = "reports/runs/drive2018_pilotnet_blocked_s0.json"
#: the measured timing of the drive the runs trained on, and the spread of the time from the history
#: frame to the current one (the shipped run has one history frame, so one entry)
TIMING = "/timing/drive2018"
LAG_GAP = f"{TIMING}/lag_gap_s/0"

MINUS = chr(0x2212)  # the minus sign the documents print
ENDASH = chr(0x2013)  # the dash the documents print in a range
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve")


# --- reading the reports ---------------------------------------------------------------------

class Reader:
    """The JSON files under one root, each parsed once; ``get`` resolves an RFC 6901 pointer."""

    def __init__(self, root: Path):
        self.root = root
        self._files: dict[str, object] = {}

    def file(self, rel: str):
        if rel not in self._files:
            self._files[rel] = json.loads((self.root / rel).read_text(encoding="utf-8"))
        return self._files[rel]

    def get(self, rel: str, pointer: str):
        node = self.file(rel)
        for step in _steps(pointer):
            node = node[int(step)] if isinstance(node, list) else node[step]
        return node


def _steps(pointer: str) -> list[str]:
    assert pointer.startswith("/"), f"{pointer!r} is not a JSON pointer"
    return [raw.replace("~1", "/").replace("~0", "~") for raw in pointer[1:].split("/")]


class Moved(Reader):
    """The reports of ``base`` with the value at one pointer of one file replaced by ``value``."""

    def __init__(self, base: Reader, rel: str, pointer: str, value):
        super().__init__(base.root)
        tree = copy.deepcopy(base.file(rel))
        *path, last = _steps(pointer)
        node = tree
        for step in path:
            node = node[int(step)] if isinstance(node, list) else node[step]
        node[int(last) if isinstance(node, list) else last] = value
        self._files = {**base._files, rel: tree}
        self.base = base

    def file(self, rel: str):
        return self._files[rel] if rel in self._files else self.base.file(rel)


# --- how a figure is printed -------------------------------------------------------------------

def _minus(text: str) -> str:
    """A negative figure is printed with a true minus sign."""
    return MINUS + text[1:] if text.startswith("-") else text


def _whole(value) -> int:
    assert float(value) == int(value), f"{value!r} is not a whole number"
    return int(value)


def _signed_percent(x: float) -> str:
    text = f"{100 * x:.0f}"
    if text in ("0", "-0"):
        return "0"
    return MINUS + text[1:] if text.startswith("-") else "+" + text


def _g_unit(x: float) -> float:
    """One step in the last significant digit ``f"{x:g}"`` prints."""
    digits = re.sub(r"[^0-9]", "", f"{abs(x):g}").lstrip("0").rstrip("0") or "0"
    return 10.0 ** (math.floor(math.log10(abs(x))) - (len(digits) - 1))


def _floored(x: float, decimals: int) -> float:
    """``x`` rounded down to ``decimals`` places; a value a hair under a step, as floating point
    leaves ``24.3`` times ten, is taken as the step."""
    step = 10 ** decimals
    return math.floor(round(x * step, 6)) / step


@dataclass(frozen=True)
class Format:
    """How a report value becomes the printed figure: ``number`` gives the quantity printed,
    ``show`` prints it, ``unit`` is one step of its printed precision (None for text) and ``scale``
    turns the quantity into the printed number's own unit (a thousand for milliseconds)."""

    number: Callable
    show: Callable
    unit: Callable | None
    scale: float = 1.0


def _word(n: int) -> str:
    return WORDS[n]


FORMATS: dict[str, Format] = {
    "int": Format(_whole, lambda x: _minus(str(x)), lambda x: 1),
    "int,": Format(_whole, lambda x: _minus(f"{x:,}"), lambda x: 1),
    "len": Format(len, str, lambda x: 1),
    "word": Format(_whole, _word, lambda x: 1),
    "lenword": Format(len, _word, lambda x: 1),
    "lenWord": Format(len, lambda x: _word(x).capitalize(), lambda x: 1),
    "1f": Format(float, lambda x: _minus(f"{x:.1f}"), lambda x: 0.1),
    "2f": Format(float, lambda x: _minus(f"{x:.2f}"), lambda x: 0.01),
    "1f,": Format(float, lambda x: _minus(f"{x:,.1f}"), lambda x: 0.1),
    #: a lower bound: the value rounded down to one decimal, so "at least" never claims more
    "floor1": Format(float, lambda x: _minus(f"{_floored(x, 1):.1f}"), lambda x: 0.1),
    "pct0": Format(float, lambda x: _minus(f"{100 * x:.0f}"), lambda x: 0.01, 100.0),
    "pct1": Format(float, lambda x: _minus(f"{100 * x:.1f}"), lambda x: 0.001, 100.0),
    "spct0": Format(float, _signed_percent, lambda x: 0.01, 100.0),
    "ms0": Format(float, lambda x: _minus(f"{1000 * x:.0f}"), lambda x: 0.001, 1000.0),
    "ms1": Format(float, lambda x: _minus(f"{1000 * x:.1f}"), lambda x: 0.0001, 1000.0),
    "min0": Format(float, lambda x: _minus(f"{x / 60:.0f}"), lambda x: 60.0, 1 / 60),
    "min1": Format(float, lambda x: _minus(f"{x / 60:.1f}"), lambda x: 6.0, 1 / 60),
    "e1": Format(float, lambda x: _minus(f"{x:.1e}"),
                 lambda x: 10.0 ** (math.floor(math.log10(abs(x))) - 1)),
    "g": Format(float, lambda x: _minus(f"{x:g}"), _g_unit),
    "text": Format(str, str, None),
    "commit": Format(lambda v: str(v)[:7], str, None),
    "json": Format(json.dumps, str, None),
}


# --- the figures each document prints ----------------------------------------------------------

#: (document, phrase with ``{}`` where the figure goes, file, JSON pointer, format)
PINS: list[tuple[str, str, str, str, str]] = [
    # README: what it is
    ("README", "on a {}-minute drive", RESULTS, "/drive_profile/drives/drive2018/duration_s", "min0"),
    ("README", "sees the current frame and the frame {} before it", EXPORT, "/models/sightline/frame_lags/0",
     "word"),
    ("README", "It tests them {} ways", SUMMARY, "/ladder/sightlinenet", "lenword"),
    ("README", "deviations are over {} training seeds", SUMMARY, "/groups/sightlinenet~1blocked/seeds", "lenword"),
    ("README", "For each of {} seeds, a line from the error", SUMMARY, "/groups/sightlinenet~1blocked/seeds",
     "lenword"),
    # the defect
    ("README", "Its test score promises an error {} times smaller", "derived", "promise", "2f"),
    # how much better than straight
    ("README", "The {} tests are the rungs of a ladder", SUMMARY, "/ladder/sightlinenet", "lenword"),
    ("README", "PilotNet does not, later in the drive or on another day; later in the drive its seeds disagree "
     "widely, from {}% to", "derived", "pilotnet_forward_best", "spct0"),
    ("README", "% to {}%. The shuffled rung is not an honest test", "derived", "pilotnet_forward_worst", "spct0"),
    # the lookup table
    ("README", "{} of those frames, from nearly straight to the sharpest turn", LEAKAGE, "/pairs", "lenWord"),
    ("README", "In all {}, the nearest-looking training frame under the shuffled split", LEAKAGE, "/pairs",
     "lenword"),
    # the baselines
    ("README", "the blocked-split SightlineNet of seed {} (chosen on validation error)", SUMMARY, "/shipped/seed",
     "int"),
    # the standard-recipe PilotNet
    ("README", "the widely copied TensorFlow 1 way on {}, scored from its weights", LEGACY, "/trained_on", "text"),
    ("README", "The recipe, as the report records it: *{}*.", LEGACY, "/recipe", "text"),
    # how it works
    ("README", "The model takes the current frame t and frame t−{},", EXPORT, "/models/sightline/frame_lags/0",
     "int"),
    ("README", "The model learned the motion between frames t−{} and t on the 2018 drive", EXPORT,
     "/models/sightline/frame_lags/0", "int"),
    ("README", "each demo clip carries the {} frames before its first scored frame", EXPORT, "/clips/c2018/lead_in",
     "word"),
    ("README", "The head applies dropout {} after its first layer", SUMMARY,
     "/train_config/models/sightlinenet/dropout", "g"),
    # the protocols
    ("README", "Every evaluation frame is then at least {} frames from a training frame, twice the firewall plus "
     "one,", LEAKAGE, "/gaps/blocked/test/min_frames", "int"),
    ("README", "and at least {} s of driving from any training frame, measured from the timestamps", "derived",
     "closest_blocked_s", "floor1"),
    ("README", "at least {} frames after the last training frame", LEAKAGE, "/gaps/forward/test/min_frames",
     "int,"),
    ("README", "which is why always steering straight is off by only {}° there", SUMMARY,
     "/groups/sightlinenet~1forward/test_zero_mae_deg", "2f"),
    ("README", "The {} runs are PilotNet and SightlineNet", SUMMARY, "/runs", "len"),
    ("README", "PilotNet and SightlineNet under the {} splits and the one-frame SightlineNet", RESULTS,
     "/split_rules", "lenword"),
    ("README", "under the blocked split, each for {} seeds", SUMMARY, "/groups/sightlinenet~1blocked/seeds",
     "lenword"),
    # every command
    ("README", "train the shipped run as the matrix trained it: SightlineNet, blocked split, seed {}, the full",
     SUMMARY, "/shipped/seed", "int"),
    ("README", "sightline leakage --drive drive2018 --seed {}", LEAKAGE, "/seed", "int"),
    ("README", "sightline demo --drive drive2018 --run {} \\ --start", SUMMARY, "/shipped/run", "text"),
    # the model card
    ("CARD", "SightlineNet trained on the blocked split of drive2018 (run `{}`)", SUMMARY, "/shipped/run", "text"),
    ("CARD", "`models/sightline-shuffled.onnx` (run `{}`)", SUMMARY, "/twin/run", "text"),
    ("CARD", "| parameters | {} |", SHIPPED_RUN, "/n_params", "int,"),
    ("CARD", "frame t first and frame t−{} second", EXPORT, "/models/sightline/frame_lags/0", "int"),
    ("CARD", "the {} zeros whose neighbours both steer at least 15° the same way are repaired by "
     "interpolating the neighbours", SUMMARY, "/drives/drive2018/repaired_labels", "int"),
    ("CARD", "**{} tests**:", SUMMARY, "/ladder/sightlinenet", "lenWord"),
    ("CARD", "*later in the drive* (train on the first {}% of the drive, test on its end)", RESULTS,
     "/split_rules/forward/train_frac", "pct0"),
    ("CARD", "(drive2017, a different day, every frame from frame {} on)", LEGACY, "/seen_first_frame", "int"),
    ("CARD", "Means ± standard deviations over {} seeds unless the line names the shipped run.", SUMMARY,
     "/groups/sightlinenet~1blocked/seeds", "lenword"),
    ("CARD", "The model reads motion between frame t−{} and frame t", EXPORT, "/models/sightline/frame_lags/0",
     "int"),
    ("CARD", "One dataset, one area: {} daytime drives", SUMMARY, "/drives", "lenword"),
    ("CARD", "Training ran on an Apple GPU through PyTorch's `{}` device", SHIPPED_RUN, "/train_device", "text"),
    ("CARD", "every evaluation ran on the `{}`.", SUMMARY, "/device", "text"),
]

#: a phrase with several figures, each (file, pointer, format) or ("derived", name, format), in order
ROWS: list[tuple[str, str, list[tuple[str, str, str]]]] = [
    # README: the headline
    ("README", "the test frames that both splits hold out: {}, {} and {} frames for the {} seeds.", [
        (SUMMARY, "/paired_leakage/sightlinenet/per_seed/0/n", "int"),
        (SUMMARY, "/paired_leakage/sightlinenet/per_seed/1/n", "int"),
        (SUMMARY, "/paired_leakage/sightlinenet/per_seed/2/n", "int"),
        (SUMMARY, "/paired_leakage/sightlinenet/per_seed", "lenword")]),
    *[("README", f"| {label} | {{}}° ± {{}}° | {{}}° ± {{}}° | **{{}} ± {{}}** | {{}}, {{}}, {{}} |", [
        (SUMMARY, f"/paired_leakage/{model}/shuffled_mae_deg/mean", "2f"),
        (SUMMARY, f"/paired_leakage/{model}/shuffled_mae_deg/sd", "2f"),
        (SUMMARY, f"/paired_leakage/{model}/blocked_mae_deg/mean", "2f"),
        (SUMMARY, f"/paired_leakage/{model}/blocked_mae_deg/sd", "2f"),
        (SUMMARY, f"/paired_leakage/{model}/ratio/mean", "2f"),
        (SUMMARY, f"/paired_leakage/{model}/ratio/sd", "2f"),
        *[(SUMMARY, f"/paired_leakage/{model}/per_seed/{i}/ratio", "2f") for i in range(3)]])
      for label, model in (("SightlineNet", "sightlinenet"), ("PilotNet", "pilotnet"))],
    # the defect
    ("README", "The SightlineNet trained on shuffled frames scores {}° ± {}° on its own test frames and {}° ± {}° "
     "on another day's drive.", [
        (SUMMARY, "/groups/sightlinenet~1leaky/test_mae_deg/mean", "2f"),
        (SUMMARY, "/groups/sightlinenet~1leaky/test_mae_deg/sd", "2f"),
        (SUMMARY, "/groups/sightlinenet~1leaky/cross_mae_deg/mean", "2f"),
        (SUMMARY, "/groups/sightlinenet~1leaky/cross_mae_deg/sd", "2f")]),
    ("README", "The blocked split's test score keeps its promise: {}° ± {}° on its test frames, {}° ± {}° on "
     "another day.", [
        (SUMMARY, "/groups/sightlinenet~1blocked/test_mae_deg/mean", "2f"),
        (SUMMARY, "/groups/sightlinenet~1blocked/test_mae_deg/sd", "2f"),
        (SUMMARY, "/groups/sightlinenet~1blocked/cross_mae_deg/mean", "2f"),
        (SUMMARY, "/groups/sightlinenet~1blocked/cross_mae_deg/sd", "2f")]),
    ("README", "it is worse on another day than trained on the blocked split ({}° ± {}° against {}° ± {}°), and "
     "neither beats always steering straight there ({}°).", [
        (SUMMARY, "/groups/pilotnet~1leaky/cross_mae_deg/mean", "2f"),
        (SUMMARY, "/groups/pilotnet~1leaky/cross_mae_deg/sd", "2f"),
        (SUMMARY, "/groups/pilotnet~1blocked/cross_mae_deg/mean", "2f"),
        (SUMMARY, "/groups/pilotnet~1blocked/cross_mae_deg/sd", "2f"),
        (SUMMARY, "/groups/pilotnet~1blocked/cross_zero_mae_deg", "2f")]),
    # how much better than straight
    ("README", "on the {} rungs it is off by {}°, {}°, {}° and {}°,", [
        (SUMMARY, "/ladder/sightlinenet", "lenword"),
        *[(SUMMARY, f"/ladder/sightlinenet/{i}/zero_mae_deg", "2f") for i in range(4)]]),
    ("README", "| {} | {} shuffled test frames per seed | {}% ± {}% | {}% ± {}% | not trained |", [
        (SUMMARY, "/ladder/sightlinenet/0/rung", "text"),
        (RESULTS, "/split_sizes/leaky/test", "int,"),
        (SUMMARY, "/groups/sightlinenet~1leaky/test_skill/mean", "spct0"),
        (SUMMARY, "/groups/sightlinenet~1leaky/test_skill/sd", "pct0"),
        (SUMMARY, "/groups/pilotnet~1leaky/test_skill/mean", "spct0"),
        (SUMMARY, "/groups/pilotnet~1leaky/test_skill/sd", "pct0")]),
    ("README", "| {} | {} test frames, at least {} frames from training | {}% ± {}% | {}% ± {}% | {}% ± {}% |", [
        (SUMMARY, "/ladder/sightlinenet/1/rung", "text"),
        (RESULTS, "/split_sizes/blocked/test", "int,"),
        (LEAKAGE, "/gaps/blocked/test/min_frames", "int"),
        (SUMMARY, "/groups/sightlinenet~1blocked/test_skill/mean", "spct0"),
        (SUMMARY, "/groups/sightlinenet~1blocked/test_skill/sd", "pct0"),
        (SUMMARY, "/groups/pilotnet~1blocked/test_skill/mean", "spct0"),
        (SUMMARY, "/groups/pilotnet~1blocked/test_skill/sd", "pct0"),
        (SUMMARY, "/groups/sightlinenet-1f~1blocked/test_skill/mean", "spct0"),
        (SUMMARY, "/groups/sightlinenet-1f~1blocked/test_skill/sd", "pct0")]),
    ("README", "| {} | the last {} frames of the drive | {}% ± {}% | {}% ± {}% | not trained |", [
        (SUMMARY, "/ladder/sightlinenet/2/rung", "text"),
        (RESULTS, "/split_sizes/forward/test", "int,"),
        (SUMMARY, "/groups/sightlinenet~1forward/test_skill/mean", "spct0"),
        (SUMMARY, "/groups/sightlinenet~1forward/test_skill/sd", "pct0"),
        (SUMMARY, "/groups/pilotnet~1forward/test_skill/mean", "spct0"),
        (SUMMARY, "/groups/pilotnet~1forward/test_skill/sd", "pct0")]),
    ("README", "| {} | {} frames of drive2017, from frame {} on | {}% ± {}% | {}% ± {}% | {}% ± {}% |", [
        (SUMMARY, "/ladder/sightlinenet/3/rung", "text"),
        (RESULTS, "/baselines/cross_drive/n", "int,"),
        (LEGACY, "/seen_first_frame", "int"),
        (SUMMARY, "/groups/sightlinenet~1blocked/cross_skill/mean", "spct0"),
        (SUMMARY, "/groups/sightlinenet~1blocked/cross_skill/sd", "pct0"),
        (SUMMARY, "/groups/pilotnet~1blocked/cross_skill/mean", "spct0"),
        (SUMMARY, "/groups/pilotnet~1blocked/cross_skill/sd", "pct0"),
        (SUMMARY, "/groups/sightlinenet-1f~1blocked/cross_skill/mean", "spct0"),
        (SUMMARY, "/groups/sightlinenet-1f~1blocked/cross_skill/sd", "pct0")]),
    *[("README", "| {} | {}° | {}° ± {}° | {}° ± {}° | not trained |", [
        (SUMMARY, f"/ladder/sightlinenet/{i}/rung", "text"),
        (SUMMARY, f"/ladder/sightlinenet/{i}/zero_mae_deg", "2f"),
        (SUMMARY, f"/ladder/sightlinenet/{i}/mae_deg/mean", "2f"),
        (SUMMARY, f"/ladder/sightlinenet/{i}/mae_deg/sd", "2f"),
        (SUMMARY, f"/ladder/pilotnet/{i}/mae_deg/mean", "2f"),
        (SUMMARY, f"/ladder/pilotnet/{i}/mae_deg/sd", "2f")]) for i in (0, 2)],
    *[("README", "| {} | {}° | {}° ± {}° | {}° ± {}° | {}° ± {}° |", [
        (SUMMARY, f"/ladder/sightlinenet/{i}/rung", "text"),
        (SUMMARY, f"/ladder/sightlinenet/{i}/zero_mae_deg", "2f"),
        (SUMMARY, f"/ladder/sightlinenet/{i}/mae_deg/mean", "2f"),
        (SUMMARY, f"/ladder/sightlinenet/{i}/mae_deg/sd", "2f"),
        (SUMMARY, f"/ladder/pilotnet/{i}/mae_deg/mean", "2f"),
        (SUMMARY, f"/ladder/pilotnet/{i}/mae_deg/sd", "2f"),
        (SUMMARY, f"/ladder/sightlinenet-1f/{j}/mae_deg/mean", "2f"),
        (SUMMARY, f"/ladder/sightlinenet-1f/{j}/mae_deg/sd", "2f")]) for i, j in ((1, 0), (3, 1))],
    # two frames
    ("README", "on the current frame alone, {} parameters against {}, is nearly as good on the blocked test frames "
     "({}% ± {}% against {}% ± {}%). But on another day it is only level with always steering straight "
     "({}% ± {}%), where two frames reach {}% ± {}%.", [
        (ONE_FRAME_RUN, "/n_params", "int,"),
        (SHIPPED_RUN, "/n_params", "int,"),
        (SUMMARY, "/groups/sightlinenet-1f~1blocked/test_skill/mean", "spct0"),
        (SUMMARY, "/groups/sightlinenet-1f~1blocked/test_skill/sd", "pct0"),
        (SUMMARY, "/groups/sightlinenet~1blocked/test_skill/mean", "spct0"),
        (SUMMARY, "/groups/sightlinenet~1blocked/test_skill/sd", "pct0"),
        (SUMMARY, "/groups/sightlinenet-1f~1blocked/cross_skill/mean", "spct0"),
        (SUMMARY, "/groups/sightlinenet-1f~1blocked/cross_skill/sd", "pct0"),
        (SUMMARY, "/groups/sightlinenet~1blocked/cross_skill/mean", "spct0"),
        (SUMMARY, "/groups/sightlinenet~1blocked/cross_skill/sd", "pct0")]),
    # the lookup table
    ("README", "On the {} paired frames of the first seed, it scores {}° when it searches the shuffled split's "
     "training frames and {}° when it searches the blocked split's. Always steering straight scores {}° on "
     "those frames.", [
        (LEAKAGE, "/paired/n", "int"),
        (LEAKAGE, "/memoriser/leaky/mae_deg", "2f"),
        (LEAKAGE, "/memoriser/blocked/mae_deg", "2f"),
        (SUMMARY, "/paired_leakage/sightlinenet/per_seed/0/zero_mae_deg", "2f")]),
    ("README", "the most similar training frame is a median of {} frame away; under the blocked split, {} frames "
     "away.", [
        (LEAKAGE, "/memoriser/leaky/median_gap_frames", "int"),
        (LEAKAGE, "/memoriser/blocked/median_gap_frames", "int,")]),
    *[("README", "| {} ({}°) | {} | {} ms | {}° | {} | {} s | {}° |", [
        (LEAKAGE, f"/pairs/{i}/frame", "int,"),
        (LEAKAGE, f"/pairs/{i}/angle_deg", "2f"),
        (LEAKAGE, f"/pairs/{i}/leaky/nn_frame", "int,"),
        (LEAKAGE, f"/pairs/{i}/leaky/gap_s", "ms0"),
        (LEAKAGE, f"/pairs/{i}/leaky/nn_angle_deg", "2f"),
        (LEAKAGE, f"/pairs/{i}/blocked/nn_frame", "int,"),
        (LEAKAGE, f"/pairs/{i}/blocked/gap_s", "1f,"),
        (LEAKAGE, f"/pairs/{i}/blocked/nn_angle_deg", "2f")]) for i in range(6)],
    # the baselines
    ("README", "| predictor | blocked test frames ({}) | another day ({} frames) |", [
        (RESULTS, "/baselines/test/n", "int,"),
        (RESULTS, "/baselines/cross_drive/n", "int,")]),
    ("README", "| SightlineNet, the shipped run | {}° | {}° |", [
        (RESULTS, "/baselines/test/model/mae_deg", "2f"),
        (RESULTS, "/baselines/cross_drive/model/mae_deg", "2f")]),
    *[("README", f"| {label} | {{}}° | {{}}° |", [
        (RESULTS, f"/baselines/test/baselines/{key}/mae_deg", "2f"),
        (RESULTS, f"/baselines/cross_drive/baselines/{key}/mae_deg", "2f")])
      for label, key in (("always steering straight", "zero"), ("always the training mean", "mean"),
                         ("oracle: the human's angle one frame earlier", "zoh"),
                         ("oracle: a straight line through the human's two previous angles", "extrap"))],
    # the regimes
    ("README", "(SightlineNet seed {}, PilotNet seed {}, the one-frame SightlineNet seed {}).", [
        (RESULTS, "/regimes/models/sightlinenet/seed", "int"),
        (RESULTS, "/regimes/models/pilotnet/seed", "int"),
        (RESULTS, "/regimes/models/sightlinenet-1f/seed", "int")]),
    *[("README", template, [
        *bins,
        (RESULTS, f"/regimes/zero/{regime}/n", "int,"),
        *[(RESULTS, f"/regimes/models/{model}/regimes/{regime}/mae_deg", "2f")
          for model in ("sightlinenet", "pilotnet", "sightlinenet-1f")],
        (RESULTS, f"/regimes/zero/{regime}/mae_deg", "2f")])
      for template, regime, bins in (
          ("| straight, under {}° either way | {} | {}° | {}° | {}° | {}° |", "straight",
           [(RESULTS, "/cutoffs/regime_bins_deg/0", "g")]),
          ("| gentle, {}° to {}° | {} | {}° | {}° | {}° | {}° |", "gentle",
           [(RESULTS, "/cutoffs/regime_bins_deg/0", "g"), (RESULTS, "/cutoffs/regime_bins_deg/1", "g")]),
          ("| sharp, {}° or more | {} | {}° | {}° | {}° | {}° |", "sharp",
           [(RESULTS, "/cutoffs/regime_bins_deg/1", "g")]))],
    ("README", "| the regimes weighted equally | | {}° | {}° | {}° | {}° |", [
        *[(RESULTS, f"/regimes/models/{model}/regimes/balanced_mae_deg", "2f")
          for model in ("sightlinenet", "pilotnet", "sightlinenet-1f")],
        (RESULTS, "/regimes/zero/balanced_mae_deg", "2f")]),
    # the standard-recipe PilotNet
    ("README", "| its own training drive, drive2017 from frame {} on | {} | {}° | {}° |", [
        (LEGACY, "/seen_first_frame", "int"),
        (LEGACY, "/eval/drive2017_all_seen/model/n", "int,"),
        (LEGACY, "/eval/drive2017_all_seen/model/mae_deg", "2f"),
        (LEGACY, "/eval/drive2017_all_seen/zero_mae_deg", "2f")]),
    *[("README", f"| drive2018, {label} | {{}} | {{}}° | {{}}° |", [
        (LEGACY, f"/eval/{key}/model/n", "int,"),
        (LEGACY, f"/eval/{key}/model/mae_deg", "2f"),
        (LEGACY, f"/eval/{key}/zero_mae_deg", "2f")])
      for label, key in (("the blocked test frames", "drive2018_blocked_test"),
                         ("the later-in-the-drive test frames", "drive2018_forward_test"))],
    ("README", "PilotNet trained by Sightline's recipe on the blocked split scores {}° ± {}°.", [
        (SUMMARY, "/groups/pilotnet~1blocked/test_mae_deg/mean", "2f"),
        (SUMMARY, "/groups/pilotnet~1blocked/test_mae_deg/sd", "2f")]),
    ("README", "its largest prediction is {}°, while {} labels in drive2018 turn the wheel past 180°.", [
        (LEGACY, "/atan_cap/max_abs_pred_deg", "1f"),
        (LEGACY, "/atan_cap/labels_beyond_180/drive2018/n", "int")]),
    ("README", "Once every prediction is made, {} of them are computed again by an independent NumPy "
     "implementation of the TensorFlow graph; the largest difference is {}°.", [
        (LEGACY, "/port_parity/frames", "int"),
        (LEGACY, "/port_parity/max_abs_diff_deg", "e1")]),
    # from PyTorch to the browser
    *[("README", f"| `{name}` | `{{}}` | {{}} bytes | {{}} | {{}}° (gate {{}}°) | {{}} (gate {{}}) |", [
        (EXPORT, f"/models/{key}/run", "text"),
        (EXPORT, f"/models/{key}/bytes", "int,"),
        (EXPORT, f"/models/{key}/parity/frames", "int"),
        (EXPORT, f"/models/{key}/parity/max_angle_diff_deg", "e1"),
        (EXPORT, f"/models/{key}/parity/angle_tolerance_deg", "g"),
        (EXPORT, f"/models/{key}/parity/max_saliency_diff", "e1"),
        (EXPORT, f"/models/{key}/parity/saliency_tolerance", "g")])
      for name, key in (("sightline.onnx", "sightline"), ("sightline-shuffled.onnx", "shuffled"))],
    # how it works: the spread is the whole 2018 drive's, so its sentence is pinned through its full stop and
    # not as a prefix that more words could follow ("on the 2018 drive's demo clip")
    *[(doc, "Frames t−{} and t are a median {} ms apart ({}–{} ms for the middle 90% of frame pairs) on the 2018 "
       "drive, whose frames carry timestamps" + end, [
        (EXPORT, "/models/sightline/frame_lags/0", "int"),
        (RESULTS, f"{LAG_GAP}/median", "ms0"),
        (RESULTS, f"{LAG_GAP}/p05", "ms0"),
        (RESULTS, f"{LAG_GAP}/p95", "ms0")])
      for doc, end in (("README", "."), ("CARD", ": the spacing varies from pair to pair."))],
    *[(doc, "there frame t−{} is simply the frame {} back", [
        (EXPORT, "/models/sightline/frame_lags/0", "int"),
        (EXPORT, "/models/sightline/frame_lags/0", "word")]) for doc in DOCUMENTS],
    ("README", "It has {} parameters; PilotNet has {}.", [
        (SHIPPED_RUN, "/n_params", "int,"),
        (PILOTNET_RUN, "/n_params", "int,")]),
    ("README", "The first {} frames of a sequence have no history: evaluation starts at frame {},", [
        (LEGACY, "/seen_first_frame", "word"),
        (LEGACY, "/seen_first_frame", "int")]),
    ("README", "AdamW at a learning rate of {} on a {}-epoch cosine schedule, with weight decay {} and batches of "
     "{}.", [
        (SUMMARY, "/train_config/lr", "g"),
        (SUMMARY, "/train_config/epochs", "int"),
        (SUMMARY, "/train_config/weight_decay", "g"),
        (SUMMARY, "/train_config/batch_size", "int")]),
    ("README", "PilotNet gets the lighter {} augmentation and dropout {} after each of its hidden fully connected "
     "layers.", [
        (SUMMARY, "/train_config/models/pilotnet/augment", "text"),
        (SUMMARY, "/train_config/models/pilotnet/dropout", "g")]),
    # the protocols
    ("README", "Every run trains on drive2018: {} frames, {} minutes, a timestamp on every frame.", [
        (SUMMARY, "/drives/drive2018/frames", "int,"),
        (RESULTS, "/drive_profile/drives/drive2018/duration_s", "min1")]),
    ("README", "The {} zeros whose neighbours both steer at least 15° the same way are repaired by "
     "interpolating the neighbours ({} in drive2017);", [
        (SUMMARY, "/drives/drive2018/repaired_labels", "int"),
        (SUMMARY, "/drives/drive2017/repaired_labels", "int")]),
    ("README", "then deal {}% to training, {}% to validation and {}% to test: {}, {} and {} frames. In the first "
     "seed's shuffle, {}% of the test frames sit within one frame of a training frame.", [
        (RESULTS, "/split_rules/leaky/train_frac", "pct0"),
        (RESULTS, "/split_rules/leaky/val_frac", "pct0"),
        (RESULTS, "/split_rules/leaky/test_frac", "pct0"),
        (RESULTS, "/split_sizes/leaky/train", "int,"),
        (RESULTS, "/split_sizes/leaky/val", "int,"),
        (RESULTS, "/split_sizes/leaky/test", "int,"),
        (LEAKAGE, "/gaps/leaky/test/frac_within_1", "pct1")]),
    ("README", "cut the drive into blocks of {} frames, dealt out by the repeating pattern `{}` (T train, "
     "V validation, E test). Then drop {} frames, the firewall, on each side", [
        (RESULTS, "/split_rules/blocked/block", "int,"),
        (RESULTS, "/split_rules/blocked/pattern", "text"),
        (RESULTS, "/split_rules/blocked/firewall", "int")]),
    ("README", "so training, validation and test frames all come from along the whole drive: {} training, {} "
     "validation and {} test frames, {} dropped.", [
        (RESULTS, "/split_sizes/blocked/train", "int,"),
        (RESULTS, "/split_sizes/blocked/val", "int,"),
        (RESULTS, "/split_sizes/blocked/test", "int,"),
        (RESULTS, "/split_sizes/blocked/dropped", "int,")]),
    ("README", "train on the first {}% of the drive, validate on the next {}% and test on the rest, with the same "
     "{}-frame firewall at both boundaries: {}, {} and {} frames, {} dropped.", [
        (RESULTS, "/split_rules/forward/train_frac", "pct0"),
        (RESULTS, "/split_rules/forward/val_frac", "pct0"),
        (RESULTS, "/split_rules/forward/firewall", "int"),
        (RESULTS, "/split_sizes/forward/train", "int,"),
        (RESULTS, "/split_sizes/forward/val", "int,"),
        (RESULTS, "/split_sizes/forward/test", "int,"),
        (RESULTS, "/split_sizes/forward/dropped", "int,")]),
    ("README", "{} of its {} frames, every frame from frame {} on, since the first {} have no history.", [
        (RESULTS, "/baselines/cross_drive/n", "int,"),
        (SUMMARY, "/drives/drive2017/frames", "int,"),
        (LEGACY, "/seen_first_frame", "int"),
        (LEGACY, "/seen_first_frame", "word")]),
    ("README", "are compared on identical frames: {}, {} and {} frames for the {} seeds.", [
        *[(SUMMARY, f"/paired_leakage/sightlinenet/per_seed/{i}/n", "int") for i in range(3)],
        (SUMMARY, "/paired_leakage/sightlinenet/per_seed", "lenword")]),
    # get the data
    ("README", "recorded for it in `src/sightline/data/verify.py` ({} and {} labelled frames)", [
        (SUMMARY, "/drives/drive2018/frames", "int,"),
        (SUMMARY, "/drives/drive2017/frames", "int,")]),
    # every command: the training run as its report records its config and split
    ("README", "--protocol blocked --seed {} \\ --epochs {} --patience {} --firewall {} --block {}", [
        (SUMMARY, "/shipped/seed", "int"),
        (SUMMARY, "/train_config/epochs", "int"),
        (SUMMARY, "/train_config/patience", "int"),
        (SHIPPED_RUN, "/splits/firewall", "int"),
        (SHIPPED_RUN, "/splits/block", "int")]),
    ("README", "python scripts/train_matrix.py --parallel 3 --epochs {} --patience {}", [
        (SUMMARY, "/train_config/epochs", "int"),
        (SUMMARY, "/train_config/patience", "int")]),
    ("README", "--start {} --frames {} --out demo.mp4", [
        (EXPORT, "/clips/c2018/first_frame", "int"),
        (EXPORT, "/clips/c2018/n", "int")]),
    # limitations
    ("README", "On average PilotNet is worse than always steering straight on both: {}% ± {}% later in the drive "
     "(its seeds from {}% to {}%) and {}% ± {}% on another day. The one-frame SightlineNet is level with always "
     "steering straight on another day ({}% ± {}%);", [
        (SUMMARY, "/groups/pilotnet~1forward/test_skill/mean", "spct0"),
        (SUMMARY, "/groups/pilotnet~1forward/test_skill/sd", "pct0"),
        ("derived", "pilotnet_forward_best", "spct0"),
        ("derived", "pilotnet_forward_worst", "spct0"),
        (SUMMARY, "/groups/pilotnet~1blocked/cross_skill/mean", "spct0"),
        (SUMMARY, "/groups/pilotnet~1blocked/cross_skill/sd", "pct0"),
        (SUMMARY, "/groups/sightlinenet-1f~1blocked/cross_skill/mean", "spct0"),
        (SUMMARY, "/groups/sightlinenet-1f~1blocked/cross_skill/sd", "pct0")]),
    ("README", "On frames steered {}° or more, the shipped model is off by {}°.", [
        (RESULTS, "/cutoffs/regime_bins_deg/1", "g"),
        (RESULTS, "/regimes/models/sightlinenet/regimes/sharp/mae_deg", "2f")]),
    ("README", "The headline ratio rests on {}, {} and {} frames of drive2018 and on {} seeds, and SightlineNet's "
     "ratio varies from seed to seed ({} ± {}).", [
        *[(SUMMARY, f"/paired_leakage/sightlinenet/per_seed/{i}/n", "int") for i in range(3)],
        (SUMMARY, "/paired_leakage/sightlinenet/per_seed", "lenword"),
        (SUMMARY, "/paired_leakage/sightlinenet/ratio/mean", "2f"),
        (SUMMARY, "/paired_leakage/sightlinenet/ratio/sd", "2f")]),
    ("README", "trained on {}% of the 2018 clip's frames, and {}% of the clip's frames sit within one frame of a "
     "frame it trained on. So on this clip it looks better than the shipped model ({}° against {}°). On the "
     "2017 clip, which neither model trained on, the twin is off by {}° and the shipped model by {}°.", [
        (EXPORT, "/clips/c2018/twin_trained_frac", "pct0"),
        (EXPORT, "/clips/c2018/twin_neighbour_frac", "pct0"),
        (EXPORT, "/clips/c2018/clip_mae_deg/shuffled", "2f"),
        (EXPORT, "/clips/c2018/clip_mae_deg/sightline", "2f"),
        (EXPORT, "/clips/c2017/clip_mae_deg/shuffled", "2f"),
        (EXPORT, "/clips/c2017/clip_mae_deg/sightline", "2f")]),
    # the model card: details
    ("CARD", "| format | ONNX, opset {}, float32, {} bytes; its metadata names the frame lags (`{}`) and the run |", [
        (EXPORT, "/models/sightline/opset", "int"),
        (EXPORT, "/models/sightline/bytes", "int,"),
        (EXPORT, "/models/sightline/frame_lags", "json")]),
    ("CARD", "onnxruntime reproduced PyTorch on {} frame stacks, every clip frame and self-test input the page uses: "
     "the largest angle difference was {}° (gate {}°) and the largest saliency difference {} (gate {}).", [
        (EXPORT, "/models/sightline/parity/frames", "int"),
        (EXPORT, "/models/sightline/parity/max_angle_diff_deg", "e1"),
        (EXPORT, "/models/sightline/parity/angle_tolerance_deg", "g"),
        (EXPORT, "/models/sightline/parity/max_saliency_diff", "e1"),
        (EXPORT, "/models/sightline/parity/saliency_tolerance", "g")]),
    # training data
    ("CARD", "MIT License): {} dashcam frames over {} minutes, a timestamp on every frame. Consecutive frames are "
     "{} to {} ms apart, a median {} ms.", [
        (SUMMARY, "/drives/drive2018/frames", "int,"),
        (RESULTS, "/drive_profile/drives/drive2018/duration_s", "min1"),
        (RESULTS, f"{TIMING}/frame_interval_s/min", "ms0"),
        (RESULTS, f"{TIMING}/frame_interval_s/max", "ms0"),
        (RESULTS, f"{TIMING}/frame_interval_s/median", "ms1")]),
    ("CARD", "The drive is cut into blocks of {} frames dealt out by the repeating pattern `{}` (T train, "
     "V validation, E test), and {} frames are dropped on each side of every boundary between a training block "
     "and an evaluation block. Every evaluation frame is then at least {} frames, and at least {} s of driving, "
     "from any training frame: {} training, {} validation and {} test frames.", [
        (RESULTS, "/split_rules/blocked/block", "int,"),
        (RESULTS, "/split_rules/blocked/pattern", "text"),
        (RESULTS, "/split_rules/blocked/firewall", "int"),
        (LEAKAGE, "/gaps/blocked/test/min_frames", "int"),
        ("derived", "closest_blocked_s", "floor1"),
        (RESULTS, "/split_sizes/blocked/train", "int,"),
        (RESULTS, "/split_sizes/blocked/val", "int,"),
        (RESULTS, "/split_sizes/blocked/test", "int,")]),
    ("CARD", "**Recipe**: {} epochs of AdamW on a cosine schedule from a learning rate of {}, weight decay {}, "
     "batches of {} and a Huber loss on the angle. The head applies dropout {} after its first layer. "
     "Augmentation is {}:", [
        (SUMMARY, "/train_config/epochs", "int"),
        (SUMMARY, "/train_config/lr", "g"),
        (SUMMARY, "/train_config/weight_decay", "g"),
        (SUMMARY, "/train_config/batch_size", "int"),
        (SUMMARY, "/train_config/models/sightlinenet/dropout", "g"),
        (SUMMARY, "/train_config/models/sightlinenet/augment", "text")]),
    ("CARD", "**Seed choice**: {} seeds were trained; the seed with the lowest validation error ships (seed {}, "
     "{}°).", [
        (SUMMARY, "/groups/sightlinenet~1blocked/seeds", "lenword"),
        (SUMMARY, "/shipped/seed", "int"),
        ("derived", "shipped_val", "2f")]),
    # evaluation
    ("CARD", "*shuffled* (frames shuffled, then dealt {}%, {}% and {}%)", [
        (RESULTS, "/split_rules/leaky/train_frac", "pct0"),
        (RESULTS, "/split_rules/leaky/val_frac", "pct0"),
        (RESULTS, "/split_rules/leaky/test_frac", "pct0")]),
    ("CARD", "have in common, {}, {} and {} frames for the {} seeds,", [
        *[(SUMMARY, f"/paired_leakage/sightlinenet/per_seed/{i}/n", "int") for i in range(3)],
        (SUMMARY, "/paired_leakage/sightlinenet/per_seed", "lenword")]),
    # results
    ("CARD", "the model trained on the blocked split is off by {} ± {} times as much as the model trained on "
     "shuffled frames (PilotNet: {} ± {}).", [
        (SUMMARY, "/paired_leakage/sightlinenet/ratio/mean", "2f"),
        (SUMMARY, "/paired_leakage/sightlinenet/ratio/sd", "2f"),
        (SUMMARY, "/paired_leakage/pilotnet/ratio/mean", "2f"),
        (SUMMARY, "/paired_leakage/pilotnet/ratio/sd", "2f")]),
    ("CARD", "**Blocked test frames**: {}° ± {}°, where always steering straight scores {}°: {}% ± {}% better than "
     "always straight.", [
        (SUMMARY, "/groups/sightlinenet~1blocked/test_mae_deg/mean", "2f"),
        (SUMMARY, "/groups/sightlinenet~1blocked/test_mae_deg/sd", "2f"),
        (SUMMARY, "/groups/sightlinenet~1blocked/test_zero_mae_deg", "2f"),
        (SUMMARY, "/groups/sightlinenet~1blocked/test_skill/mean", "pct0"),
        (SUMMARY, "/groups/sightlinenet~1blocked/test_skill/sd", "pct0")]),
    ("CARD", "**Another day**: {}° ± {}°, where always steering straight scores {}°: {}% ± {}% better.", [
        (SUMMARY, "/groups/sightlinenet~1blocked/cross_mae_deg/mean", "2f"),
        (SUMMARY, "/groups/sightlinenet~1blocked/cross_mae_deg/sd", "2f"),
        (SUMMARY, "/groups/sightlinenet~1blocked/cross_zero_mae_deg", "2f"),
        (SUMMARY, "/groups/sightlinenet~1blocked/cross_skill/mean", "pct0"),
        (SUMMARY, "/groups/sightlinenet~1blocked/cross_skill/sd", "pct0")]),
    ("CARD", "**The shipped run alone**: {}° on the blocked test frames and {}° on another day.", [
        (RESULTS, "/baselines/test/model/mae_deg", "2f"),
        (RESULTS, "/baselines/cross_drive/model/mae_deg", "2f")]),
    ("CARD", "**By steering regime**, shipped run, blocked test frames: {}° on nearly straight frames, where always "
     "steering straight scores {}° and wins; {}° on gentle turns ({}°); {}° on sharp ones ({}°).", [
        (RESULTS, "/regimes/models/sightlinenet/regimes/straight/mae_deg", "2f"),
        (RESULTS, "/regimes/zero/straight/mae_deg", "2f"),
        (RESULTS, "/regimes/models/sightlinenet/regimes/gentle/mae_deg", "2f"),
        (RESULTS, "/regimes/zero/gentle/mae_deg", "2f"),
        (RESULTS, "/regimes/models/sightlinenet/regimes/sharp/mae_deg", "2f"),
        (RESULTS, "/regimes/zero/sharp/mae_deg", "2f")]),
    ("CARD", "SightlineNet scores {}° ± {}° on its own shuffled test frames and {}° ± {}° on another day.", [
        (SUMMARY, "/groups/sightlinenet~1leaky/test_mae_deg/mean", "2f"),
        (SUMMARY, "/groups/sightlinenet~1leaky/test_mae_deg/sd", "2f"),
        (SUMMARY, "/groups/sightlinenet~1leaky/cross_mae_deg/mean", "2f"),
        (SUMMARY, "/groups/sightlinenet~1leaky/cross_mae_deg/sd", "2f")]),
    ("CARD", "it is worse on another day than trained on the blocked split ({}° ± {}° against {}° ± {}°).", [
        (SUMMARY, "/groups/pilotnet~1leaky/cross_mae_deg/mean", "2f"),
        (SUMMARY, "/groups/pilotnet~1leaky/cross_mae_deg/sd", "2f"),
        (SUMMARY, "/groups/pilotnet~1blocked/cross_mae_deg/mean", "2f"),
        (SUMMARY, "/groups/pilotnet~1blocked/cross_mae_deg/sd", "2f")]),
    # the multi-frame caveat
    ("CARD", "The first {} frames of any sequence have no history: evaluation starts at frame {}, and the page's "
     "clips carry the {} frames before their first scored frame.", [
        (LEGACY, "/seen_first_frame", "word"),
        (LEGACY, "/seen_first_frame", "int"),
        (EXPORT, "/clips/c2018/lead_in", "word")]),
    # limitations
    ("CARD", "On frames steered {}° or more, the shipped run is still off by {}°.", [
        (RESULTS, "/cutoffs/regime_bins_deg/1", "g"),
        (RESULTS, "/regimes/models/sightlinenet/regimes/sharp/mae_deg", "2f")]),
    ("CARD", "only level with always steering straight on another day ({}% ± {}%, against {}% ± {}% with two "
     "frames)", [
        (SUMMARY, "/groups/sightlinenet-1f~1blocked/cross_skill/mean", "spct0"),
        (SUMMARY, "/groups/sightlinenet-1f~1blocked/cross_skill/sd", "pct0"),
        (SUMMARY, "/groups/sightlinenet~1blocked/cross_skill/mean", "spct0"),
        (SUMMARY, "/groups/sightlinenet~1blocked/cross_skill/sd", "pct0")]),
    ("CARD", "The paired set is small ({}, {} and {} frames, {} seeds).", [
        *[(SUMMARY, f"/paired_leakage/sightlinenet/per_seed/{i}/n", "int") for i in range(3)],
        (SUMMARY, "/paired_leakage/sightlinenet/per_seed", "lenword")]),
    ("CARD", "the twin trained on {}% of the frames and {}% sit within one frame of a frame it trained on, so it looks "
     "better there than the shipped model ({}° against {}° on this clip).", [
        (EXPORT, "/clips/c2018/twin_trained_frac", "pct0"),
        (EXPORT, "/clips/c2018/twin_neighbour_frac", "pct0"),
        (EXPORT, "/clips/c2018/clip_mae_deg/shuffled", "2f"),
        (EXPORT, "/clips/c2018/clip_mae_deg/sightline", "2f")]),
    # versions: the ones the evaluation recorded
    ("CARD", "Evaluation: Python {}, PyTorch {}, NumPy {}.", [
        (RESULTS, "/versions/python", "text"),
        (RESULTS, "/versions/torch", "text"),
        (RESULTS, "/versions/numpy", "text")]),
    ("CARD", "Export: onnx {} and onnxruntime {}.", [
        (EXPORT, "/versions/onnx", "text"),
        (EXPORT, "/versions/onnxruntime", "text")]),
    ("CARD", "Code: trained at commit `{}`, evaluated at commit `{}`.", [
        (SUMMARY, "/code_versions/train/0", "commit"),
        (SUMMARY, "/code_versions/eval/0", "commit")]),
]


# --- the figures the documents compute ---------------------------------------------------------

def _promise(r: Reader) -> float:
    """How many times its own test score the shuffled-trained SightlineNet's another-day error is."""
    group = r.get(SUMMARY, "/groups/sightlinenet~1leaky")
    return group["cross_mae_deg"]["mean"] / group["test_mae_deg"]["mean"]


def _closest_blocked_s(r: Reader) -> float:
    """The closest in time any blocked evaluation frame comes to a training frame: the smaller of the
    validation and the test figure the timestamps give. The documents print it as a lower bound, so
    its format rounds it down."""
    closest = r.get(RESULTS, f"{TIMING}/closest_eval_s/blocked")
    return min(closest["val"]["gap_s"], closest["test"]["gap_s"])


def _pilotnet_forward(r: Reader) -> list[float]:
    return r.get(SUMMARY, "/groups/pilotnet~1forward/test_skill/values")


def _shipped_val(r: Reader) -> float:
    """The shipped seed's validation error: the figure the choice was made on."""
    group = r.get(SUMMARY, "/groups/sightlinenet~1blocked")
    return group["val_mae_deg"]["values"][group["seeds"].index(r.get(SUMMARY, "/shipped/seed"))]


DERIVED: dict[str, Callable[[Reader], float]] = {
    "promise": _promise,
    "closest_blocked_s": _closest_blocked_s,
    "pilotnet_forward_best": lambda r: max(_pilotnet_forward(r)),
    "pilotnet_forward_worst": lambda r: min(_pilotnet_forward(r)),
    "shipped_val": _shipped_val,
}


# --- the sentences whose words rest on the numbers ---------------------------------------------

def _per_seed(r: Reader, group: str, stat: str) -> list[float]:
    return r.get(SUMMARY, f"/groups/{group}/{stat}/values")


def _run_report(r: Reader, model: str, protocol: str, seed: int) -> dict:
    return r.file(f"reports/runs/drive2018_{model}_{protocol}_s{seed}.json")


def _about_three(r: Reader) -> bool:
    """The words "about three times": both networks' ratios are three to the nearest half."""
    ratios = [r.get(SUMMARY, f"/paired_leakage/{model}/ratio/mean") for model in ("sightlinenet", "pilotnet")]
    return all(round(2 * ratio) / 2 == 3 for ratio in ratios)


def _most_shuffled_test_frames_next_door(r: Reader) -> bool:
    return r.get(LEAKAGE, "/gaps/leaky/test/frac_within_1") > 0.5


def _shuffled_sightlinenet_ahead_every_seed(r: Reader) -> bool:
    shuffled = _per_seed(r, "sightlinenet~1leaky", "cross_mae_deg")
    blocked = _per_seed(r, "sightlinenet~1blocked", "cross_mae_deg")
    same_seeds = r.get(SUMMARY, "/groups/sightlinenet~1leaky/seeds") == r.get(SUMMARY, "/groups/sightlinenet~1blocked/seeds")
    return same_seeds and all(s < b for s, b in zip(shuffled, blocked))


def _pilotnet_shuffled_behind(r: Reader) -> bool:
    """Worse, in every seed: PilotNet trained on shuffled frames is further off on another day."""
    shuffled = _per_seed(r, "pilotnet~1leaky", "cross_mae_deg")
    blocked = _per_seed(r, "pilotnet~1blocked", "cross_mae_deg")
    return all(s > b for s, b in zip(shuffled, blocked))


def _pilotnet_shuffled_behind_and_neither_beats_straight(r: Reader) -> bool:
    shuffled = _per_seed(r, "pilotnet~1leaky", "cross_mae_deg")
    blocked = _per_seed(r, "pilotnet~1blocked", "cross_mae_deg")
    straight = r.get(SUMMARY, "/groups/pilotnet~1blocked/cross_zero_mae_deg")
    return _pilotnet_shuffled_behind(r) and min(shuffled + blocked) > straight


def _much_driving_near_straight(r: Reader) -> bool:
    """Much of the driving: at least two fifths of the blocked test frames, spread along the drive, are
    steered less than the straight band either way."""
    zero = r.get(RESULTS, "/regimes/zero")
    counts = [zero[regime]["n"] for regime in ("straight", "gentle", "sharp")]
    return counts[0] / sum(counts) >= 0.4


def _sightlinenet_ahead_on_every_honest_rung(r: Reader) -> bool:
    return all(skill > 0 for stat_group in (("sightlinenet~1blocked", "test_skill"),
                                            ("sightlinenet~1forward", "test_skill"),
                                            ("sightlinenet~1blocked", "cross_skill"))
               for skill in _per_seed(r, *stat_group))


def _pilotnet_behind_later_and_another_day(r: Reader) -> bool:
    return r.get(SUMMARY, "/groups/pilotnet~1forward/test_skill/mean") < 0 \
        and r.get(SUMMARY, "/groups/pilotnet~1blocked/cross_skill/mean") < 0


def _shuffled_zero_is_the_seed_mean(r: Reader) -> bool:
    """The shuffled rung's always-straight error is the mean of its seeds' own, which differ; every other
    rung's is the one value all its seeds' reports share."""
    seeds = r.get(SUMMARY, "/groups/sightlinenet~1blocked/seeds")
    rungs = {entry["rung"]: entry["zero_mae_deg"] for entry in r.get(SUMMARY, "/ladder/sightlinenet")}

    def zeros(protocol: str, where: str) -> list[float]:
        return [_run_report(r, "sightlinenet", protocol, seed)[where]["baselines"]["zero"]["mae_deg"]
                for seed in seeds]

    shuffled = zeros("leaky", "test")
    others = {"blocked": zeros("blocked", "test"), "later in the drive": zeros("forward", "test"),
              "another day": zeros("blocked", "cross_drive")}
    return len(set(shuffled)) > 1 and math.isclose(rungs["shuffled"], sum(shuffled) / len(shuffled), rel_tol=1e-12) \
        and all(len(set(values)) == 1 and rungs[rung] == values[0] for rung, values in others.items())


def _one_frame_level_on_another_day(r: Reader) -> bool:
    """Level: the mean is within one standard deviation of zero."""
    skill = r.get(SUMMARY, "/groups/sightlinenet-1f~1blocked/cross_skill")
    return abs(skill["mean"]) < skill["sd"]


def _one_frame_nearly_as_good_on_blocked(r: Reader) -> bool:
    one = r.get(SUMMARY, "/groups/sightlinenet-1f~1blocked/test_skill/mean")
    two = r.get(SUMMARY, "/groups/sightlinenet~1blocked/test_skill/mean")
    return 0 < one < two and two - one < 0.1


def _one_frame_blocked_only(r: Reader) -> bool:
    """The one-frame SightlineNet was trained on the blocked split alone, so it has no later-in-the-drive
    score: its only group and its ladder's rungs say so."""
    groups = [g for g in r.get(SUMMARY, "/groups") if g.startswith("sightlinenet-1f/")]
    rungs = [entry["rung"] for entry in r.get(SUMMARY, "/ladder/sightlinenet-1f")]
    return groups == ["sightlinenet-1f/blocked"] and rungs == ["blocked", "another day"]


def _single_frame_models_no_edge(r: Reader) -> bool:
    """Do not reliably beat: PilotNet and the one-frame SightlineNet both read one frame, and later in the drive
    and on another day no mean edge of theirs over always steering straight is larger than the spread of its
    seeds (one standard deviation); PilotNet's means are below it, and the one-frame model has no later score.
    A single seed can still be ahead of always steering straight, which is why the sentence says "reliably"."""
    from sightline.models import MODEL_SPECS
    single = all(MODEL_SPECS[name].frame_lags == () for name in ("pilotnet", "sightlinenet-1f"))
    skills = [r.get(SUMMARY, f"/groups/{group}/{stat}") for group, stat in (
        ("pilotnet~1forward", "test_skill"), ("pilotnet~1blocked", "cross_skill"),
        ("sightlinenet-1f~1blocked", "cross_skill"))]
    return single and all(skill["mean"] < skill["sd"] for skill in skills) \
        and _pilotnet_behind_later_and_another_day(r) and _one_frame_blocked_only(r)


def _straight_wins_on_straight_road(r: Reader) -> bool:
    regimes = r.get(RESULTS, "/regimes")
    return all(regimes["zero"]["straight"]["mae_deg"] < m["regimes"]["straight"]["mae_deg"]
               for m in regimes["models"].values())


def _models_ahead_on_curves(r: Reader) -> bool:
    """Earn their keep on curves: on gentle and sharp frames every model beats always steering straight."""
    regimes = r.get(RESULTS, "/regimes")
    return all(m["regimes"][regime]["mae_deg"] < regimes["zero"][regime]["mae_deg"]
               for m in regimes["models"].values() for regime in ("gentle", "sharp"))


def _sharp_turns_hardest(r: Reader) -> bool:
    """Sharp turns stay hard: every model is further off on sharp frames than on straight or gentle ones."""
    return all(m["regimes"]["sharp"]["mae_deg"] > max(m["regimes"][regime]["mae_deg"] for regime in ("straight", "gentle"))
               for m in r.get(RESULTS, "/regimes/models").values())


def _blocked_promise_kept(r: Reader) -> bool:
    group = r.get(SUMMARY, "/groups/sightlinenet~1blocked")
    return abs(group["cross_mae_deg"]["mean"] / group["test_mae_deg"]["mean"] - 1) < 0.1


def _legacy_good_at_home_worse_than_straight_on_2018(r: Reader) -> bool:
    scores = r.get(LEGACY, "/eval")
    home = scores["drive2017_all_seen"]
    return home["model"]["mae_deg"] < home["zero_mae_deg"] and all(
        scores[key]["model"]["mae_deg"] > scores[key]["zero_mae_deg"]
        for key in ("drive2018_blocked_test", "drive2018_forward_test"))


def _legacy_under_the_cap(r: Reader) -> bool:
    from sightline.legacy import CAP_DEG
    cap = r.get(LEGACY, "/atan_cap")
    return cap["max_abs_pred_deg"] < CAP_DEG and cap["labels_beyond_180"]["drive2018"]["n"] > 0


def _pilotnet_keeps_the_arctangent(r: Reader) -> bool:
    """PilotNet here keeps the arctangent output of the TensorFlow 1 recipe: however large its last layer's
    answer, it stays inside the cap, while SightlineNet's linear head does not."""
    import torch
    from sightline.data.preprocess import INPUT_HEIGHT, INPUT_WIDTH
    from sightline.legacy import CAP_DEG
    from sightline.models import PilotNet, SightlineNet
    frame = torch.zeros((1, INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=torch.uint8)
    pilotnet, sightlinenet = PilotNet().eval(), SightlineNet().eval()
    with torch.no_grad():
        pilotnet.head.bias.fill_(1e3)
        sightlinenet.regressor[-1].bias.fill_(1e2)
        capped, linear = abs(float(pilotnet(frame))), abs(float(sightlinenet(frame)))
    return capped < CAP_DEG < linear and "atan" in r.get(LEGACY, "/recipe")


def _sightlinenet_dropout_after_first_layer(r: Reader) -> bool:
    """SightlineNet's one dropout sits in its head, after the first fully connected layer and before the
    second."""
    from torch import nn
    from sightline.models import SightlineNet
    layers = list(SightlineNet().regressor)
    drops = [i for i, layer in enumerate(layers) if isinstance(layer, nn.Dropout)]
    linears = [i for i, layer in enumerate(layers) if isinstance(layer, nn.Linear)]
    return len(drops) == 1 and linears[0] < drops[0] < linears[1]


def _pilotnet_dropout_after_each_hidden_layer(r: Reader) -> bool:
    """PilotNet's dropout runs after each hidden fully connected layer: the order of the calls in one pass."""
    import torch
    from sightline.data.preprocess import INPUT_HEIGHT, INPUT_WIDTH
    from sightline.models import PilotNet
    model = PilotNet().eval()
    order: list[str] = []
    for i, layer in enumerate(model.fc):
        layer.register_forward_hook(lambda *_, i=i: order.append(f"fc{i}"))
    model.drop.register_forward_hook(lambda *_: order.append("drop"))
    model.head.register_forward_hook(lambda *_: order.append("head"))
    with torch.no_grad():
        model(torch.zeros((1, INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=torch.uint8))
    return order == [step for i in range(len(model.fc)) for step in (f"fc{i}", "drop")] + ["head"]


def _pairs(r: Reader) -> list[dict]:
    return r.get(LEAKAGE, "/pairs")


def _shuffled_neighbour_next_door(r: Reader) -> bool:
    return all(p["leaky"]["gap_frames"] == 1 for p in _pairs(r))


def _shuffled_neighbour_same_angle(r: Reader) -> bool:
    """Nearly the same angle: within the regimes' straight band of each other."""
    band = r.get(RESULTS, "/cutoffs/regime_bins_deg/0")
    return all(abs(p["leaky"]["nn_angle_deg"] - p["angle_deg"]) < band for p in _pairs(r))


def _blocked_neighbour_minutes_away(r: Reader) -> bool:
    """Over a minute away: every one of the six blocked-split neighbours is more than sixty seconds off."""
    return all(p["blocked"]["gap_s"] > 60 for p in _pairs(r))


def _blocked_neighbour_often_elsewhere(r: Reader) -> bool:
    """Often steering somewhere else: for most of the six, and for most of the paired frames, the blocked
    split's most similar training frame steers more than 5° away (a tenth or fewer under the shuffled split)."""
    band = r.get(RESULTS, "/cutoffs/regime_bins_deg/0")
    away = sum(abs(p["blocked"]["nn_angle_deg"] - p["angle_deg"]) > band for p in _pairs(r))
    memoriser = r.get(LEAKAGE, "/memoriser")
    return band == 5 and 2 * away > len(_pairs(r)) \
        and memoriser["blocked"]["within_5_deg"] < 0.5 and memoriser["leaky"]["within_5_deg"] > 0.9


def _later_rung_calmer(r: Reader) -> bool:
    """Calmer: the drive's last stretch has a smaller mean absolute angle (always steering straight's error
    there) than the whole drive, weighted by frames over the drive profile's buckets."""
    profile = r.get(RESULTS, "/drive_profile/drives/drive2018")
    starts, means, frames = profile["start_frame"], profile["mean_abs_deg"], profile["frames"]
    sizes = [end - start for start, end in zip(starts, starts[1:] + [frames])]
    whole = sum(mean * size for mean, size in zip(means, sizes)) / frames
    return r.get(SUMMARY, "/groups/sightlinenet~1forward/test_zero_mae_deg") < whole


def _firewall_arithmetic(r: Reader) -> bool:
    firewall = r.get(LEAKAGE, "/splits/firewall")
    gaps = r.get(LEAKAGE, "/gaps/blocked")
    return gaps["test"]["min_frames"] == gaps["val"]["min_frames"] == 2 * firewall + 1


def _closest_in_time_respects_the_firewall(r: Reader) -> bool:
    """The pairs the timing names closest in time, for the blocked split's validation and test frames, are as
    many frames apart as the firewall allows at least: frames and seconds measure the same separation."""
    closest = r.get(RESULTS, f"{TIMING}/closest_eval_s/blocked")
    least = r.get(LEAKAGE, "/gaps/blocked/test/min_frames")
    return set(closest) == {"val", "test"} and all(abs(c["eval_frame"] - c["train_frame"]) >= least
                                                  for c in closest.values())


def _blocked_spread_along_the_drive(r: Reader) -> bool:
    """Along the whole drive: the blocked split's training, validation and test frames each reach within one
    turn of the pattern (its blocks times the block) of both ends of the drive, and the split the code makes
    is the one the reports count."""
    from sightline.data.splits import make_split
    rules = r.get(RESULTS, "/split_rules/blocked")
    frames = r.get(SUMMARY, "/drives/drive2018/frames")
    split = make_split("blocked", frames, firewall=rules["firewall"], block=rules["block"])
    turn = len(rules["pattern"]) * rules["block"]
    sizes = r.get(RESULTS, "/split_sizes/blocked")
    return all(part.min() < turn and part.max() >= frames - turn for part in (split.train, split.val, split.test)) \
        and [len(split.train), len(split.val), len(split.test)] == [sizes["train"], sizes["val"], sizes["test"]]


def _forward_test_after_validation(r: Reader) -> bool:
    """The later-in-the-drive test frames come after all its validation frames, which come after all its
    training frames; its test frames' distance from training is from the last training frame."""
    from sightline.data.splits import make_split
    frames = r.get(SUMMARY, "/drives/drive2018/frames")
    split = make_split("forward", frames, firewall=r.get(RESULTS, "/split_rules/forward/firewall"))
    return split.train.max() < split.val.min() and split.val.max() < split.test.min() \
        and split.test.min() - split.train.max() == r.get(LEAKAGE, "/gaps/forward/test/min_frames")


def _everything_on_the_cpu(r: Reader) -> bool:
    return r.get(SUMMARY, "/device") == r.get(LEGACY, "/device") == r.get(EXPORT, "/device") == "cpu"


def _trained_on_mps(r: Reader) -> bool:
    return {json.loads(path.read_text())["train_device"]
            for path in (r.root / "reports" / "runs").glob("*.json")} == {"mps"}


def _trained_on_the_gpu_scored_on_the_cpu(r: Reader) -> bool:
    return _trained_on_mps(r) and _everything_on_the_cpu(r)


def _the_matrix(r: Reader) -> bool:
    expected = {f"{m}/{p}" for m in ("pilotnet", "sightlinenet") for p in ("leaky", "blocked", "forward")}
    return set(r.get(SUMMARY, "/groups")) == expected | {"sightlinenet-1f/blocked"} and all(
        len(g["seeds"]) == len(r.get(SUMMARY, "/groups/sightlinenet~1blocked/seeds"))
        for g in r.get(SUMMARY, "/groups").values())


def _ladder_rungs_line_up(r: Reader) -> bool:
    ladder = r.get(SUMMARY, "/ladder")
    rungs = [e["rung"] for e in ladder["sightlinenet"]]
    return [e["rung"] for e in ladder["pilotnet"]] == rungs \
        and [e["rung"] for e in ladder["sightlinenet-1f"]] == [rungs[1], rungs[3]]


def _shipped_chosen_on_validation(r: Reader) -> bool:
    shipped = r.get(SUMMARY, "/shipped")
    group = r.get(SUMMARY, "/groups/sightlinenet~1blocked")
    values = group["val_mae_deg"]["values"]
    return shipped["selected_by"] == "val_mae_deg" and shipped["run"] == r.get(RESULTS, "/baselines/run") \
        and group["seeds"][values.index(min(values))] == shipped["seed"] \
        and r.get(SHIPPED_RUN, "/run") == shipped["run"] == r.get(EXPORT, "/models/sightline/run")


def _twin_on_the_clips(r: Reader) -> bool:
    clips = r.get(EXPORT, "/clips")
    return clips["c2018"]["clip_mae_deg"]["shuffled"] < clips["c2018"]["clip_mae_deg"]["sightline"] \
        and clips["c2017"]["twin_trained_frac"] == 0 and clips["c2017"]["drive"] == "drive2017" \
        and r.get(SUMMARY, "/twin/run") == r.get(EXPORT, "/models/shuffled/run")


def _clips_harder_than_average(r: Reader) -> bool:
    """Harder than an average stretch: each clip is chosen by the spread of the human's labels alone, and on
    it both the shipped model and always steering straight are further off than on all the frames of its
    kind (the blocked test frames for the 2018 clip, the other drive for the 2017 one)."""
    clips = r.get(EXPORT, "/clips")
    baselines = r.get(RESULTS, "/baselines")
    kinds = {"c2018": baselines["test"], "c2017": baselines["cross_drive"]}
    return set(clips) == set(kinds) and all(
        "standard deviation of its steering labels" in clips[clip]["rule"]
        and clips[clip]["clip_mae_deg"]["zero"] > kind["baselines"]["zero"]["mae_deg"]
        and clips[clip]["clip_mae_deg"]["sightline"] > kind["model"]["mae_deg"]
        for clip, kind in kinds.items())


def _every_clip_has_the_lead_in(r: Reader) -> bool:
    """Each clip carries as many frames before its first scored frame as the model looks back."""
    clips = r.get(EXPORT, "/clips")
    return len(clips) > 1 and all(clip["lead_in"] == max(r.get(EXPORT, "/models/sightline/frame_lags"))
                                  for clip in clips.values())


def _timed_lag_is_the_shipped_lag(r: Reader) -> bool:
    """The spread printed for frames t−2 and t is the one measured for the shipped model's frame lag."""
    return [gap["lag"] for gap in r.get(RESULTS, f"{TIMING}/lag_gap_s")] == r.get(EXPORT, "/models/sightline/frame_lags")


def _drive2017_untimed(r: Reader) -> bool:
    return r.get(RESULTS, "/timing/drive2017") is None and r.get(SUMMARY, "/drives/drive2017/median_dt_s") is None


def _verify_knows_the_reported_drives(r: Reader) -> bool:
    """The label files `sightline data verify` accepts are the ones every report was measured on."""
    from sightline.data.verify import KNOWN_DRIVES
    drives = r.get(SUMMARY, "/drives")
    return {name: (drive["frames"], drive["label_sha256"]) for name, drive in drives.items()} \
        == {name: (known["frames"], known["label_sha256"]) for name, known in KNOWN_DRIVES.items()}


def _parity_on_every_page_input(r: Reader) -> bool:
    clips = r.get(EXPORT, "/clips")
    stacks = sum(clip["n"] for clip in clips.values()) + len(r.get(EXPORT, "/selftest/frames"))
    return all(model["parity"]["frames"] == stacks for model in r.get(EXPORT, "/models").values())


def _one_commit_each(r: Reader) -> bool:
    versions = r.get(SUMMARY, "/code_versions")
    return len(versions["train"]) == len(versions["eval"]) == 1


def _one_set_of_evaluation_versions(r: Reader) -> bool:
    """The versions the card names are the ones every run's evaluation recorded."""
    versions = r.get(RESULTS, "/versions")
    return versions == r.get(SUMMARY, "/versions") and all(
        json.loads(path.read_text())["versions"] == versions for path in (r.root / "reports" / "runs").glob("*.json"))


def _second_frame_carries_most(r: Reader) -> bool:
    """Most of what the model carries to another day comes with the second frame: the one-frame model keeps
    less than half of the two-frame model's edge over always steering straight there."""
    one = r.get(SUMMARY, "/groups/sightlinenet-1f~1blocked/cross_skill/mean")
    two = r.get(SUMMARY, "/groups/sightlinenet~1blocked/cross_skill/mean")
    return two > 0 and two - one > two / 2


def _first_seed_is_the_leakage_seed(r: Reader) -> bool:
    first = r.get(SUMMARY, "/paired_leakage/sightlinenet/per_seed/0")
    return first["seed"] == r.get(LEAKAGE, "/seed") == min(r.get(SUMMARY, "/groups/sightlinenet~1blocked/seeds")) \
        and first["n"] == r.get(LEAKAGE, "/paired/n")


#: (document, phrase that must stand in it, the condition on the reports that makes it true)
CLAIMS: list[tuple[str, str, Callable[[Reader], bool]]] = [
    ("README", "make a steering model look about three times better than it is", _about_three),
    ("README", "makes the same kind of model look about three times better than it is on road it has not seen "
     "nearby", _about_three),
    ("README", "is off by about three times as much", _about_three),
    ("README", "most test frames have a training frame right next to them", _most_shuffled_test_frames_next_door),
    ("README", "on another day it is more accurate than the SightlineNet trained on the blocked split, in every "
     "seed", _shuffled_sightlinenet_ahead_every_seed),
    ("README", "PilotNet differs: trained on shuffled frames, it is worse on another day than trained on the "
     "blocked split", _pilotnet_shuffled_behind_and_neither_beats_straight),
    ("README", "and neither beats always steering straight there", _pilotnet_shuffled_behind_and_neither_beats_straight),
    ("README", "The blocked split's test score keeps its promise", _blocked_promise_kept),
    ("README", "because much of the driving is close to straight ahead", _much_driving_near_straight),
    ("README", "SightlineNet beats always steering straight on every honest rung, in every seed.",
     _sightlinenet_ahead_on_every_honest_rung),
    ("README", "PilotNet does not, later in the drive or on another day", _pilotnet_behind_later_and_another_day),
    ("README", "the shuffled rung's always-straight figure is the mean over the seeds; on every other rung all "
     "seeds are scored on the same frames", _shuffled_zero_is_the_seed_mean),
    ("README", "is nearly as good on the blocked test frames", _one_frame_nearly_as_good_on_blocked),
    ("README", "But on another day it is only level with always steering straight", _one_frame_level_on_another_day),
    ("README", "On straight road, always steering straight wins", _straight_wins_on_straight_road),
    ("README", "The models earn their keep on curves", _models_ahead_on_curves),
    ("README", "and sharp turns stay hard", _sharp_turns_hardest),
    ("README", "Sharp turns stay hard.", _sharp_turns_hardest),
    ("README", "Good on its own training drive, worse than always steering straight on the 2018 drive.",
     _legacy_good_at_home_worse_than_straight_on_2018),
    ("README", "The arctangent head is a ceiling as well", _legacy_under_the_cap),
    ("README", "not the arctangent output of the widely copied TensorFlow 1 implementation of PilotNet (which this "
     "repository's PilotNet keeps)", _pilotnet_keeps_the_arctangent),
    ("README", "after its first layer, and the checkpoint", _sightlinenet_dropout_after_first_layer),
    ("README", "after each of its hidden fully connected layers", _pilotnet_dropout_after_each_hidden_layer),
    ("README", "the nearest-looking training frame under the shuffled split is the frame next door",
     _shuffled_neighbour_next_door),
    ("README", "steering at nearly the same angle", _shuffled_neighbour_same_angle),
    ("README", "Under the blocked split it is over a minute away", _blocked_neighbour_minutes_away),
    ("README", "and often steering somewhere else", _blocked_neighbour_often_elsewhere),
    ("README", "the drive's last stretch is calmer than the drive as a whole", _later_rung_calmer),
    ("README", "Its test frames come after the whole validation stretch", _forward_test_after_validation),
    ("README", "twice the firewall plus one", _firewall_arithmetic),
    ("README", "of driving from any training frame, measured from the timestamps", _closest_in_time_respects_the_firewall),
    ("README", "so training, validation and test frames all come from along the whole drive",
     _blocked_spread_along_the_drive),
    ("README", "Training ran on the GPU and every evaluation on the CPU.", _trained_on_the_gpu_scored_on_the_cpu),
    ("README", "The 21 runs are PilotNet and SightlineNet under the three splits and the one-frame SightlineNet "
     "under the blocked split", _the_matrix),
    ("README", "| rung | always straight | SightlineNet | PilotNet | SightlineNet, one frame |", _ladder_rungs_line_up),
    ("README", "the blocked-split SightlineNet of seed 2 (chosen on validation error)", _shipped_chosen_on_validation),
    ("README", "So on this clip it looks better than the shipped model", _twin_on_the_clips),
    ("README", "On the 2017 clip, which neither model trained on", _twin_on_the_clips),
    ("README", "chosen from the human's labels alone, so it is harder than an average stretch",
     _clips_harder_than_average),
    ("README", "each demo clip carries the two frames before its first scored frame", _every_clip_has_the_lead_in),
    ("README", "of frame pairs) on the 2018 drive, whose frames carry timestamps", _timed_lag_is_the_shipped_lag),
    ("README", "drive2017 has no timestamps", _drive2017_untimed),
    ("README", "every clip frame and self-test input the page uses", _parity_on_every_page_input),
    ("README", "(PyTorch's MPS backend)", _trained_on_mps),
    ("README", "paired frames of the first seed", _first_seed_is_the_leakage_seed),
    ("README", "In the first seed's shuffle", _first_seed_is_the_leakage_seed),
    ("README", "recorded for it in `src/sightline/data/verify.py`", _verify_knows_the_reported_drives),
    ("README", "Single-frame models do not reliably beat always steering straight later in the drive or on another "
     "day.", _single_frame_models_no_edge),
    ("README", "On average PilotNet is worse than always steering straight on both",
     _pilotnet_behind_later_and_another_day),
    ("README", "The one-frame SightlineNet is level with always steering straight on another day",
     _one_frame_level_on_another_day),
    ("README", "it was never trained on the later-in-the-drive split, so it has no score there",
     _one_frame_blocked_only),
    ("CARD", "look about three times better than it is on road it has not seen nearby", _about_three),
    ("CARD", "overstates accuracy by about three times here", _about_three),
    ("CARD", "For SightlineNet, the defect is the score the shuffled split reports, not the model it trains",
     _shuffled_sightlinenet_ahead_every_seed),
    ("CARD", "is more accurate than SightlineNet trained on the blocked split, in every seed",
     _shuffled_sightlinenet_ahead_every_seed),
    ("CARD", "PilotNet is different: trained on shuffled frames, it is worse on another day than trained on the "
     "blocked split", _pilotnet_shuffled_behind),
    ("CARD", "where always steering straight scores", _straight_wins_on_straight_road),
    ("CARD", "only level with always steering straight on another day", _one_frame_level_on_another_day),
    ("CARD", "so most of what this model carries to another day depends on the second frame",
     _second_frame_carries_most),
    ("CARD", "the seed with the lowest validation error ships", _shipped_chosen_on_validation),
    ("CARD", "after its first layer. Augmentation is", _sightlinenet_dropout_after_first_layer),
    ("CARD", "every clip frame and self-test input the page uses", _parity_on_every_page_input),
    ("CARD", "so it looks better there than the shipped model", _twin_on_the_clips),
    ("CARD", "The clips are chosen for having the most steering, so their errors describe the clip.",
     _clips_harder_than_average),
    ("CARD", "the page's clips carry the two frames before their first scored frame", _every_clip_has_the_lead_in),
    ("CARD", "of frame pairs) on the 2018 drive, whose frames carry timestamps", _timed_lag_is_the_shipped_lag),
    ("CARD", "drive2017 has no timestamps", _drive2017_untimed),
    ("CARD", "of driving, from any training frame", _closest_in_time_respects_the_firewall),
    ("CARD", "Training ran on an Apple GPU", _trained_on_mps),
    ("CARD", "Evaluation: Python", _one_set_of_evaluation_versions),
    ("CARD", "Code: trained at commit", _one_commit_each),
]


# --- the numbers that are not measurements -----------------------------------------------------

def _camera(m: re.Match) -> bool:
    from sightline.export.site_data import CAMERA_SIZE
    return (int(m["w"]), int(m["h"])) == tuple(CAMERA_SIZE)


def _crop(m: re.Match) -> bool:
    from sightline.data.preprocess import CROP_BOTTOM
    return int(m["rows"]) == CROP_BOTTOM


def _input(m: re.Match) -> bool:
    from sightline.data.preprocess import INPUT_HEIGHT, INPUT_WIDTH
    from sightline.models import SHIPPED_LAGS
    channels = m.groupdict().get("c")
    return (int(m["h"]), int(m["w"])) == (INPUT_HEIGHT, INPUT_WIDTH) and (
        channels is None or int(channels) == 3 * (1 + len(SHIPPED_LAGS)))


def _one_angle(m: re.Match) -> bool:
    from sightline.models import SightlineNet
    return SightlineNet().regressor[-1].out_features == int(m["n"])


def _huber(m: re.Match) -> bool:
    from sightline.train import HUBER_SCALE_DEG
    return float(m["deg"]) == HUBER_SCALE_DEG


def _cap(m: re.Match) -> bool:
    from sightline.legacy import CAP_DEG
    return float(m["deg"]) == CAP_DEG


def _repair_threshold(m: re.Match) -> bool:
    from sightline.data.labels import repair_dropouts
    return float(m["deg"]) == inspect.signature(repair_dropouts).parameters["min_neighbor_deg"].default


def _middle_share(m: re.Match) -> bool:
    from sightline.export.site_data import SPREAD_PERCENTILES
    return int(m["pct"]) == SPREAD_PERCENTILES["p95"] - SPREAD_PERCENTILES["p05"]


def _number_words(m: re.Match) -> list[str]:
    return [word.lower() for word in NUMBER_WORD.findall(m[0])]


def _frames_read(m: re.Match) -> bool:
    """"two" frames is what SightlineNet reads, the current frame and its history frame (SHIPPED_LAGS); "one"
    or a "single" frame is what PilotNet and the one-frame SightlineNet read."""
    from sightline.models import MODEL_SPECS
    reads = {name: 1 + len(spec.frame_lags) for name, spec in MODEL_SPECS.items()}
    counted = {"one": 1, "single": 1, "two": 2}
    return reads == {"pilotnet": 1, "sightlinenet": 2, "sightlinenet-1f": 1} \
        and all(word in counted for word in _number_words(m))


def _convolutions(m: re.Match) -> bool:
    from torch import nn
    from sightline.models import SightlineNet
    convolutions = [layer for layer in SightlineNet().features.modules() if isinstance(layer, nn.Conv2d)]
    return _number_words(m) == ["five"] and len(convolutions) == 5


@functools.lru_cache(maxsize=1)
def _shipped_graph():
    """The graph of the ONNX file the page runs, read once."""
    import onnx
    return onnx.load(str(REPO / "web" / "public" / "models" / "sightline.onnx")).graph


def _graph_outputs(m: re.Match) -> bool:
    """One graph, two answers: the shipped ONNX file holds a single graph with the angle and the saliency
    map as its two outputs."""
    outputs = [output.name for output in _shipped_graph().output]
    return _number_words(m) == ["one", "two"] and outputs == ["steering_deg", "saliency"]


def _data_types(m: re.Match) -> bool:
    """The data type names the documents give the graph: its input bytes and its float outputs."""
    from onnx import TensorProto
    graph = _shipped_graph()
    types = {"uint8": [i.type.tensor_type.elem_type for i in graph.input] == [TensorProto.UINT8],
             "float32": {o.type.tensor_type.elem_type for o in graph.output} == {TensorProto.FLOAT}}
    return types[m[0]]


def _drive_name(m: re.Match) -> bool:
    return m[0] in READER.get(SUMMARY, "/drives")


def _oracles(m: re.Match) -> bool:
    """The oracles as the evaluation defines them: the human's angle one frame earlier, and the straight line
    through the human's two previous angles, carried on to the frame scored."""
    import numpy as np
    from sightline.evaluate import baseline_predictions
    angles = np.array([10.0, 20.0, 40.0, 80.0])
    predicted = baseline_predictions(angles, np.array([3]), train_mean_deg=0.0)
    return predicted["zoh"].tolist() == [40.0] and predicted["extrap"].tolist() == [60.0]


def _within_one_frame(m: re.Match) -> bool:
    """"within one frame" names the measure the reports keep: the share of frames next to a training frame."""
    return "frac_within_1" in READER.get(LEAKAGE, "/gaps/leaky/test") \
        and "twin_neighbour_frac" in READER.get(EXPORT, "/clips/c2018")


def _skill_formula(m: re.Match) -> bool:
    """The share of always-straight's error a model removes is one minus the ratio of the two errors, seed by
    seed, as the summary records it."""
    group = READER.get(SUMMARY, "/groups/sightlinenet~1blocked")
    zero = group["test_zero_mae_deg"]
    return all(math.isclose(skill, 1 - mae / zero, rel_tol=1e-9)
               for skill, mae in zip(group["test_skill"]["values"], group["test_mae_deg"]["values"]))


def _parallel_three(m: re.Match) -> bool:
    """Three runs at a time is how the matrix was run, as its command in the README says."""
    return _number_words(m) == ["three"] and "--parallel 3 " in document("README")


def _two_drives(m: re.Match) -> bool:
    return sorted(READER.get(SUMMARY, "/drives")) == ["drive2017", "drive2018"]


#: the hosts the documents link to; a digit in a link's address names an account, a colour or a file
LINK_HOSTS = {"github.com", "img.shields.io", "safdar-hussain1.github.io", "drive.google.com", "arxiv.org",
              "localhost:4173"}


def _link(m: re.Match) -> bool:
    from urllib.parse import urlparse
    return urlparse(m[0]).netloc in LINK_HOSTS


def _python(m: re.Match) -> bool:
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    return f'requires-python = ">={m["v"]}"' in pyproject


def _notebook(m: re.Match) -> bool:
    return (REPO / "notebooks" / m[0]).is_file()


#: the README's sentence on training times; ``test_the_training_times_match_the_run_histories`` reads the
#: figures in it again from the run folders wherever they are present
TRAINING_TIMES = (r"each run took (?P<least>\d+) to (?P<most>\d+) minutes by its own history; the shipped run "
                  r"took (?P<shipped>\d+) minutes")

#: (documents, pattern, why the numbers in its matches may be printed, a check on a match or None)
def _verdict(m: re.Match) -> bool:
    """The verdict's shape is the one scripts/verify_page.sh accepts: every check passed, on one thread."""
    script = (REPO / "scripts" / "verify_page.sh").read_text(encoding="utf-8")
    return "threads=1" in script


ALLOWED: list[tuple[tuple[str, ...], str, str, Callable[[re.Match], bool] | None]] = [
    (("README", "CARD"), r"(?P<w>\d+) × (?P<h>\d+) (?:pixels|frame)",
     "the camera's frame size, the site builder's CAMERA_SIZE", _camera),
    (("README", "CARD"), r"bottom (?P<rows>\d+) rows", "the crop, the preprocessing's CROP_BOTTOM", _crop),
    (("README", "CARD"), r"resized to (?P<w>\d+) × (?P<h>\d+)", "the model's input size, the preprocessing's "
     "INPUT_WIDTH and INPUT_HEIGHT", _input),
    (("README",), r"(?P<h>\d+) × (?P<w>\d+) × (?P<c>\d+)", "the model's input array: the input size and three "
     "channels for each frame the shipped model reads (SHIPPED_LAGS)", _input),
    (("CARD",), r"\(N, (?P<h>\d+), (?P<w>\d+)(?:, (?P<c>\d+))?\)", "the graph's input and saliency shapes, from "
     "the same constants", _input),
    (("CARD",), r"\(N, (?P<n>\d+)\)", "the angle output's shape: one angle per frame stack", _one_angle),
    (("CARD",), r"\[0, 1\]", "the saliency map's range: each map is scaled from its minimum to its maximum", None),
    (("README",), r"units of (?P<deg>\d+)°", "the Huber loss's scale, the training code's HUBER_SCALE_DEG", _huber),
    (("README",), r"past ±?(?P<deg>\d+)°", "the arctangent head's limit, the standard-recipe code's CAP_DEG", _cap),
    (("README", "CARD"), r"neighbours both steer at least (?P<deg>\d+)° the same way",
     "the dropout repair's threshold, the default of repair_dropouts' min_neighbor_deg", _repair_threshold),
    (("README", "CARD"), r"for the middle (?P<pct>\d+)% of frame pairs", "the share of frame pairs between the "
     "5th and 95th percentiles the timing records (SPREAD_PERCENTILES)", _middle_share),
    (("README",), r"A predictor that always answers 0°|costs more than answering 0°",
     "what always steering straight answers, by definition", None),
    (("CARD",), r"always steering straight \(0°\)", "what always steering straight answers, by definition", None),
    # counts the code fixes, each checked against it
    (("README", "CARD"), r"SightlineNet, one frame|one-frame SightlineNet|Single-frame models|a single frame shows "
     r"only scenery|Two frames, not one|where two frames reach|The difference between two frames|\*\*Two "
     r"frames\.\*\*|two RGB frames|The same network on one frame|with two frames",
     "how many frames a model reads: SightlineNet two, PilotNet and the one-frame SightlineNet one (MODEL_SPECS)",
     _frames_read),
    (("README", "CARD"), r"\bfive convolutions", "SightlineNet's convolutions, counted in the model", _convolutions),
    (("README",), r"\*\*One graph, two answers\.\*\*", "the exported graph's outputs: the angle and the saliency "
     "map, read from the ONNX file", _graph_outputs),
    (("README",), r"the human's angle one frame earlier|the human's two previous angles", "the oracles' "
     "definitions in the evaluation code, checked on a short run of angles", _oracles),
    (("README", "CARD"), r"within one frame", "the leakage measure's definition: a frame next to a training "
     "frame (frac_within_1, twin_neighbour_frac)", _within_one_frame),
    (("README",), r"one minus the ratio of the two errors", "the skill formula, checked seed by seed against "
     "the summary", _skill_formula),
    (("README",), r"twice the firewall plus one", "the blocked split's least separation in frames, checked "
     "against the leakage report", lambda m: _firewall_arithmetic(READER)),
    (("README", "CARD"), r"about three times", "the paired ratio of both networks, three to the nearest half",
     lambda m: _about_three(READER)),
    (("README",), r"three at a time|up to three runs sharing it", "how the matrix was run, its --parallel 3",
     _parallel_three),
    # words that are not counts, and counts of the setting rather than of a result
    (("README", "CARD"), r"logs a zero for a frame", "the logged value itself, a reading of 0°, not a count", None),
    (("README", "CARD"), r"One dataset, one area|from one dataset", "the data's provenance: both drives come "
     "from one dataset page, which names the same places for both", _two_drives),
    (("README", "CARD"), r"one WebAssembly thread|WebAssembly, one thread", "the page's onnxruntime-web "
     "setting: the page's own tests hold it to a single thread", None),
    # names with digits in them
    (("README", "CARD"), r"\bdrive201[78]\b", "the two drives' names, as the code and the reports name them",
     _drive_name),
    (("README", "CARD"), r"\b(?:uint8|float32)\b", "data type names: the graph's input bytes and float outputs, "
     "read from the ONNX file", _data_types),
    (("README",), r"Apple M1 Pro", "the training machine's chip, a product name", None),
    (("README",), r"\bpython3\b", "the Python 3 interpreter's command name", None),
    (("README",), r"\b[01]\.jpg\b", "the frames' file names, as the dataset folders hold them", None),
    (("README", "CARD"), r"https?://[^\s<>()\[\]`]*\d[^\s<>()\[\]`]*", "a link's address, where a digit names "
     "an account, a colour or a file; only the hosts in LINK_HOSTS", _link),
    (("README",), r"Python (?P<v>\d+\.\d+)(?= or newer|\+)", "the Python versions pyproject.toml accepts", _python),
    (("README",), r"--parallel 3\b", "how many runs the matrix runner trains at once: how the matrix was run, "
     "not a result", None),
    (("README",), r"localhost:4173", "the port Vite's preview server listens on by default", None),
    (("README",), r"\?selftest=1\b", "the switch that runs the page's self-test", None),
    (("README",), r"\b0[12]-[a-z-]+\.ipynb", "the notebooks' file names", _notebook),
    (("README",), TRAINING_TIMES, "training times, summed from the epoch times in each run's history.json; the run "
     "folders are not part of the repository, so they are read again only where they are present", None),
    (("README",), r"scoring the shipped run again took (?P<eval>\d+) s, and `eval --all` scores the runs one after "
     r"another; `legacy` took (?P<legacy>\d+\.\d) s", "how long two commands took when they were run on the "
     "machine named above; a duration depends on the machine and no report records it", None),
    (("README",), r'"07/01/2018 Dataset"|"Dataset 1"', "the dataset page's own names for the two drives", None),
    (("README", "CARD"), r"\b(?:2017|2018)(?= (?:drive|clip|demo clip)\b)", "a year naming a drive or its clip",
     None),
    (("README", "CARD"), r"Copyright \(c\) 2018 Sully Chen", "the dataset's copyright notice, from its LICENSE",
     None),
    (("README",), r"arXiv:\d{4}\.\d{5}|arxiv\.org/abs/\d{4}\.\d{5}", "references' arXiv identifiers", None),
    (("README",), r"TensorFlow 1\b", "the framework's major version, part of its name", None),
    (("README",), r"SHA-256", "the hash function's name", None),
    (("README",), r"(?<![^ ])[1-4]\. \*\*(?=Two frames|The network|One graph, two answers|In the browser)",
     "the numbered steps of How it works", None),
    (("README",), r"`PASS (?P<k>\d)/(?P=k) threads=1`", "the page verifier's verdict on a pass: the self-test's checks, all "
     "passed, on one thread (scripts/verify_page.sh matches its exact shape)", _verdict),
    (("README",), r"IPv6", "the name of the internet protocol's version", None),
    (("README",), r"`--host 127\.0\.0\.1`|`127\.0\.0\.1` may refuse", "the loopback address, a command-line "
     "setting of Vite's preview server", None),
]

#: "at least" before these is a rule's own threshold, a constant of the code an ALLOWED entry checks, not a
#: measured bound that could be rounded the wrong way
DEFINED_BOUNDS = (r"neighbours both steer at least (\d+)° the same way",)


# --- matching -----------------------------------------------------------------------------------

#: a number token, whole: it starts where no letter, digit, underscore or point is glued before it, holds
#: a digit before its first character that is not part of a word, and runs on through letters, digits and
#: underscores, a thousands comma or a decimal point that a digit follows, and an exponent's sign
#: (``1.8E-04``); a decimal may start at its point (``.5``). ``0.jpg`` reads as ``0``, a file name the
#: allowed list names.
NUMBER = re.compile(r"""
    (?<![\w.])
    (?=\.?\w*\d)
    \.?\w+
    (?:(?:[.,]\d|(?<=\d[eE])[+-]\d)\w*)*
""", re.VERBOSE)

_CARDINALS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
              "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety").split()
_COUNTS = ("hundred thousand million billion dozen quarter third fourth fifth sixth seventh eighth ninth tenth "
           "eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth "
           "twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth ninetieth hundredth thousandth "
           "millionth").split()
#: a number word: a cardinal (``threefold`` too), a large count or an ordinal from ``third`` on (plurals
#: too), or a multiple
NUMBER_WORD = re.compile(r"\b(?:(?:" + "|".join(_CARDINALS) + r")(?:fold)?|(?:" + "|".join(_COUNTS)
                         + r")s?|twice|thrice|half|halves|single|double|triple)\b", re.IGNORECASE)


def tokens(text: str) -> list[re.Match]:
    """Every number token and number word in ``text``, in order."""
    return sorted([*NUMBER.finditer(text), *NUMBER_WORD.finditer(text)], key=lambda m: m.start())


def _collapse(text: str) -> str:
    """Runs of whitespace as one space, so a phrase the document wraps over two lines still matches."""
    return re.sub(r"\s+", " ", text)


def document(key: str) -> str:
    path = REPO / DOCUMENTS[key]
    assert path.is_file(), f"{DOCUMENTS[key]} is missing, so every check here would search nothing"
    return _collapse(path.read_text(encoding="utf-8"))


def phrase_pattern(phrase: str, figure: str) -> re.Pattern:
    """The phrase with ``figure`` in place of its one ``{}``; the figure is a number of its own."""
    before, sep, after = phrase.partition("{}")
    assert sep and "{}" not in after, f"{phrase!r} needs exactly one {{}}"
    return re.compile(re.escape(_collapse(before)) + r"(?<![\d.,])(" + re.escape(figure) + r")(?!\d)"
                      + re.escape(_collapse(after)))


@dataclass(frozen=True)
class Check:
    """One printed figure: where it stands, where its value comes from and how it is printed."""

    doc: str
    phrase: str
    source: str
    pointer: str
    fmt: str

    def number(self, r: Reader):
        if self.source == "derived":
            return DERIVED[self.pointer](r)
        return FORMATS[self.fmt].number(r.get(self.source, self.pointer))

    def figure(self, r: Reader) -> str:
        return FORMATS[self.fmt].show(self.number(r))

    def pattern(self, r: Reader, figure: str | None = None) -> re.Pattern:
        return phrase_pattern(self.phrase, self.figure(r) if figure is None else figure)

    @property
    def label(self) -> str:
        return f"{self.doc}|{self.pointer}|{self.phrase[:40]}"


def _fill(template: str, figures: list[str], keep: int) -> str:
    """``template`` with every ``{}`` but the ``keep``-th replaced by its figure."""
    parts = template.split("{}")
    assert len(parts) == len(figures) + 1, f"{template!r} has {len(parts) - 1} places for {len(figures)} figures"
    out = parts[0]
    for i, figure in enumerate(figures):
        out += ("{}" if i == keep else figure) + parts[i + 1]
    return out


def checks(r: Reader) -> list[Check]:
    """Every printed figure, ``ROWS`` expanded into one check per figure."""
    out = [Check(*pin) for pin in PINS]
    for doc, template, values in ROWS:
        cells = [Check(doc, "{}", source, pointer, fmt) for source, pointer, fmt in values]
        figures = [cell.figure(r) for cell in cells]
        for i, cell in enumerate(cells):
            out.append(Check(doc, _fill(template, figures, i), cell.source, cell.pointer, cell.fmt))
    return out


def failures(r: Reader) -> list[str]:
    """Every check that does not hold with the reports ``r`` reads: pins, claims, bounds and the scan."""
    texts = {key: document(key) for key in DOCUMENTS}
    problems = [f"{c.label}: the document lacks {_fill(c.phrase, [c.figure(r)], -1)!r}"
                for c in checks(r) if not c.pattern(r).search(texts[c.doc])]
    for doc, phrase, condition in CLAIMS:
        if _collapse(phrase) not in texts[doc]:
            problems.append(f"{doc}: the claim {phrase!r} is not in the document")
        elif not condition(r):
            problems.append(f"{doc}: the reports no longer support {phrase!r}")
    problems += [f"{doc}: {token!r} at {at} is neither pinned nor allowed" for doc, token, at in uncovered(r, texts)]
    problems += [f"{doc}: 'at least {figure}' at {at} {why}" for doc, figure, at, why in loose_bounds(r, texts)]
    return problems


def uncovered(r: Reader, texts: dict[str, str]) -> list[tuple[str, str, int]]:
    """Every number token and number word outside a pinned figure and outside an allowed match."""
    spans: dict[str, list[tuple[int, int]]] = {key: [] for key in texts}
    for c in checks(r):
        if c.doc in texts:
            spans[c.doc] += [m.span(1) for m in c.pattern(r).finditer(texts[c.doc])]
    for docs, pattern, _reason, check in ALLOWED:
        for doc in docs:
            if doc in texts:
                # an allowed phrase covers its numbers only while its check on the code holds
                spans[doc] += [m.span() for m in re.finditer(pattern, texts[doc]) if check is None or check(m)]
    return [(doc, m[0], m.start()) for doc, text in texts.items() for m in tokens(text)
            if not any(start <= m.start() and m.end() <= end for start, end in spans[doc])]


AT_LEAST = re.compile(r"\bat least (\d[\d,]*(?:\.\d+)?)")


def bound_holds(figure: str, exact: float) -> bool:
    """Whether ``figure``, printed after "at least", is no more than ``exact`` rounded down to the figure's
    own precision: "24.2" holds for 24.286, "24.3" does not."""
    decimals = len(figure.partition(".")[2])
    return float(figure.replace(",", "")) <= _floored(exact, decimals)


def loose_bounds(r: Reader, texts: dict[str, str], pinned: list[Check] | None = None,
                 ) -> list[tuple[str, str, int, str]]:
    """Every figure the documents print after "at least" that is not a pinned value rounded down (or a
    rule's own threshold): a lower bound must never claim more than was measured. ``pinned`` is the
    checks to read, every check by default."""
    where: dict[tuple[str, tuple[int, int]], list[Check]] = {}
    for c in checks(r) if pinned is None else pinned:
        if c.doc in texts:
            for found in c.pattern(r).finditer(texts[c.doc]):
                where.setdefault((c.doc, found.span(1)), []).append(c)
    out = []
    for doc, text in texts.items():
        defined = [m.span(1) for pattern in DEFINED_BOUNDS for m in re.finditer(pattern, text)]
        for m in AT_LEAST.finditer(text):
            if m.span(1) in defined:
                continue
            bounding = where.get((doc, m.span(1)), [])
            if not bounding:
                out.append((doc, m[1], m.start(), "is not a pinned figure"))
            for c in bounding:
                exact = FORMATS[c.fmt].scale * c.number(r)
                if not bound_holds(m[1], exact):
                    out.append((doc, m[1], m.start(), f"claims more than {c.source} {c.pointer} ({exact!r})"))
    return out


# --- the tests ----------------------------------------------------------------------------------

READER = Reader(REPO)
CHECKS = checks(READER)


@pytest.mark.parametrize("check", CHECKS, ids=[c.label for c in CHECKS])
def test_every_printed_figure_is_the_reports_figure_in_its_phrase(check):
    text = document(check.doc)
    assert check.pattern(READER).search(text), (
        f"{DOCUMENTS[check.doc]} lacks {_fill(check.phrase, [check.figure(READER)], -1)!r}: {check.source} "
        f"{check.pointer} printed as {check.fmt}. Either the document is stale or the report moved; "
        f"read the JSON, never an earlier draft")


@pytest.mark.parametrize("doc, phrase, condition", CLAIMS, ids=[f"{d}|{p[:40]}" for d, p, _ in CLAIMS])
def test_every_claim_in_words_holds_in_the_reports(doc, phrase, condition):
    assert _collapse(phrase) in document(doc), f"{DOCUMENTS[doc]} no longer says {phrase!r}; update CLAIMS"
    assert condition(READER), f"the reports no longer support {phrase!r} ({condition.__name__})"


def test_every_number_in_the_documents_is_pinned_or_allowed():
    texts = {key: document(key) for key in DOCUMENTS}
    left = uncovered(READER, texts)
    assert left == [], "numbers no pin and no allowed entry accounts for: " + ", ".join(
        f"{doc} {token!r} ({texts[doc][max(0, at - 40):at + 20]!r})" for doc, token, at in left)
    found = {key: len(tokens(text)) for key, text in texts.items()}
    assert found["README"] >= 350 and found["CARD"] >= 100, f"the scan found almost nothing: {found}"


def test_every_at_least_is_its_value_rounded_down():
    """"at least 24.2 s" for a closest gap of 24.286 s holds; "at least 24.3 s" would claim 14 ms that were never
    measured. Every figure after "at least" is pinned, and no more than its value rounded down."""
    texts = {key: document(key) for key in DOCUMENTS}
    assert loose_bounds(READER, texts) == []
    bounds = {key: len(AT_LEAST.findall(text)) for key, text in texts.items()}
    assert bounds["README"] >= 5 and bounds["CARD"] >= 3, f"the documents print fewer lower bounds than expected: {bounds}"


@pytest.mark.parametrize("figure, exact, holds", [
    ("24.2", 24.286, True), ("24.3", 24.286, False), ("24.6", 24.613, True), ("501", 501, True),
    ("502", 501, False), ("10,075", 10075, True), ("24.3", 24.3, True), ("24.3", 24.299999999, True),
    ("24.2", 24.2, True), ("24", 24.9, True), ("25", 24.9, False),
])
def test_a_bound_holds_only_when_rounded_down(figure, exact, holds):
    assert bound_holds(figure, exact) is holds


def test_a_rounded_up_or_unpinned_bound_is_caught():
    """The closest blocked gap printed to the nearest tenth (24.3 s for 24.286 s) instead of rounded down is
    caught even when its pin agrees with the text, and so is an "at least" figure no pin accounts for."""
    readme = document("README")
    rounded_up = readme.replace("at least 24.2 s of driving", "at least 24.3 s of driving")
    assert rounded_up != readme
    nearest = [Check("README", "and at least {} s of driving from any training frame, measured from the "
                     "timestamps", "derived", "closest_blocked_s", "1f")]
    caught = [(figure, why) for _, figure, _, why in loose_bounds(READER, {"README": rounded_up}, nearest)
              if figure == "24.3"]
    assert len(caught) == 1 and caught[0][1].startswith("claims more than"), caught
    unpinned = readme.replace("at least 24.2 s of driving", "at least 24.1 s of driving")
    assert ("24.1", "is not a pinned figure") in [(figure, why) for _, figure, _, why
                                                   in loose_bounds(READER, {"README": unpinned})]


def test_the_training_times_match_the_run_histories():
    """The README's training times are summed from each run's history.json, which no clone has: wherever the
    run folders are present, they are read again here."""
    histories = sorted((REPO / "runs").glob("drive2018_*/history.json")) if (REPO / "runs").is_dir() else []
    if not histories:
        pytest.skip("the run folders are not part of the repository, so the training times cannot be read here")
    minutes = {path.parent.name: sum(epoch["seconds"] for epoch in json.loads(path.read_text())["history"]) / 60
               for path in histories}
    assert sorted(minutes) == sorted(READER.get(SUMMARY, "/runs")), "the run folders are not the reported runs"
    printed = re.search(TRAINING_TIMES, document("README"))
    assert printed, "the README no longer states the training times"
    shipped = minutes[READER.get(SUMMARY, "/shipped/run")]
    assert (int(printed["least"]), int(printed["most"]), int(printed["shipped"])) == (
        round(min(minutes.values())), round(max(minutes.values())), round(shipped)), minutes


@pytest.mark.parametrize("docs, pattern, reason, check", ALLOWED, ids=[a[1][:30] for a in ALLOWED])
def test_every_allowed_number_has_a_reason_is_used_and_matches_the_code(docs, pattern, reason, check):
    assert len(reason.split()) >= 3, f"{pattern!r} needs a reason"
    for doc in docs:
        matches = list(re.finditer(pattern, document(doc)))
        assert matches, f"{pattern!r} allows nothing in {DOCUMENTS[doc]}; remove it there"
        assert all(tokens(m[0]) for m in matches), f"{pattern!r} matches text without a number"
        if check is not None:
            assert all(check(m) for m in matches), f"{DOCUMENTS[doc]}: {pattern!r} disagrees with the code"


def moved_values(value, fmt: str) -> list:
    """The report value moved one unit of its printed precision either way: a number by the format's
    unit, a list by one item (its length is what is printed), a text by one character at its start."""
    if fmt in ("len", "lenword", "lenWord", "json"):
        if isinstance(value, dict):
            return [{**value, "x": None}, dict(list(value.items())[:-1])]
        return [[*value, value[-1]], value[:-1]]
    if FORMATS[fmt].unit is None:
        return ["x" + value, value[1:]]
    unit = FORMATS[fmt].unit(FORMATS[fmt].number(value))
    return [value + unit, value - unit]


@pytest.mark.parametrize("check", CHECKS, ids=[c.label for c in CHECKS])
def test_a_one_unit_move_of_the_report_value_breaks_its_pin(check):
    """A pin that a moved report would still satisfy protects nothing. The value is moved in the report
    itself, one unit of its printed precision up and down (a derived figure is moved where it is
    computed), and the phrase the check requires must then be gone from the document."""
    text = document(check.doc)
    printed = check.figure(READER)
    if check.source == "derived":
        fmt = FORMATS[check.fmt]
        x = check.number(READER)
        figures = [fmt.show(x + fmt.unit(x)), fmt.show(x - fmt.unit(x))]
    else:
        value = READER.get(check.source, check.pointer)
        figures = [check.figure(Moved(READER, check.source, check.pointer, moved))
                   for moved in moved_values(value, check.fmt)]
    for figure in figures:
        assert figure != printed, f"{check.label}: a one-unit move does not change {printed!r}"
        assert not check.pattern(READER, figure).search(text), (
            f"{check.label}: {figure!r} also stands in this phrase, so a report moving to it would leave the pin "
            f"green; anchor the phrase better")


def test_a_moved_report_value_reddens_every_place_that_prints_it():
    """End to end: the headline ratio moved by one unit in a copy of the summary, and every check is run
    again on it; each place that prints the ratio must fail, in both documents."""
    pointer = "/paired_leakage/sightlinenet/ratio/mean"
    problems = failures(Moved(READER, SUMMARY, pointer, READER.get(SUMMARY, pointer) + 0.01))
    printed = [c for c in CHECKS if c.pointer == pointer]
    named = [p for p in problems if f"|{pointer}|" in p]
    assert len(printed) >= 3 and {c.doc for c in printed} == set(DOCUMENTS)
    assert len(named) == len(printed), problems


def test_reports_written_with_another_indent_leave_every_pin_holding(tmp_path):
    """The checks read values, not text: every report and the page's results re-indented with four spaces
    must pass exactly as the committed files do."""
    sources = {SUMMARY, LEAKAGE, LEGACY, EXPORT, RESULTS, *(p.relative_to(REPO).as_posix()
                                                         for p in (REPO / "reports" / "runs").glob("*.json"))}
    changed = 0
    for rel in sorted(sources):
        original = (REPO / rel).read_text(encoding="utf-8")
        rewritten = json.dumps(json.loads(original), indent=4)
        changed += rewritten != original
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(rewritten, encoding="utf-8")
    assert changed == len(sources), "a rewritten report kept its layout, so this test proves nothing"
    assert failures(Reader(tmp_path)) == []
    assert failures(READER) == []


def test_figures_come_only_from_the_reports_and_the_page_results():
    used = {c.source for c in CHECKS if c.source != "derived"}
    assert all((rel.startswith("reports/") and rel.endswith(".json")) or rel == RESULTS for rel in used), used
    assert {SUMMARY, LEAKAGE, LEGACY, EXPORT, RESULTS} <= used
    assert all(name in DERIVED for name in {c.pointer for c in CHECKS if c.source == "derived"})


def test_the_checks_are_not_vacuous():
    per_doc = {key: sum(c.doc == key for c in CHECKS) for key in DOCUMENTS}
    assert len(CHECKS) >= 300 and per_doc["README"] >= 220 and per_doc["CARD"] >= 70, per_doc
    assert len(CLAIMS) >= 30 and len(ALLOWED) >= 20


@pytest.mark.parametrize("text, found", [
    ("3.05 ± 0.61, 1,073.2 s and 1.8e-04°", ["3.05", "0.61", "1,073.2", "1.8e-04"]),
    ("frame t−2 and (N, 66, 200, 6)", ["2", "66", "200", "6"]),
    ("8,469, 57 ms and 893, 889 and 943 frames.", ["8,469", "57", "893", "889", "943"]),
    # digits with letters glued to them, a decimal that starts at its point and a capital exponent are each
    # one token, read whole
    ("3x faster, 57ms apart at 60fps over 12k frames", ["3x", "57ms", "60fps", "12k"]),
    ("the 3rd seed and the 20th epoch", ["3rd", "20th"]),
    ("off by .5 s, or 1.8E-04° and 2.5e+3", [".5", "1.8E-04", "2.5e+3"]),
    # a word with digits in it is a token, so a name is allowed only by naming it
    ("drive2018, sightlinenet-1f, s2, M1, uint8 and sha256", ["drive2018", "1f", "s2", "M1", "uint8", "sha256"]),
    ("0.jpg, data.txt, .venv and e.g. a README.md", ["0"]),
])
def test_the_scan_reads_numbers_as_a_reader_does(text, found):
    assert NUMBER.findall(text) == found


@pytest.mark.parametrize("text, found", [
    ("nine seeds, twelve runs and a dozen clips", ["nine", "twelve", "dozen"]),
    ("Twice the firewall plus one, a single drive, threefold, three-fold", ["Twice", "one", "single", "threefold",
                                                                           "three"]),
    ("hundreds of thousands, a third of the frames, the twentieth epoch, half the runs",
     ["hundreds", "thousands", "third", "twentieth", "half"]),
    ("someone, alone, none, often, weight, phone, tone, zones, done", []),
    ("the first frame, a second later, once, both seeds", []),
])
def test_the_scan_reads_number_words(text, found):
    assert NUMBER_WORD.findall(text) == found


#: phrases each planted in turn into a copy of the README's text, every one a figure or a count that no pin
#: or allowed phrase accounts for; the scan must name each
PLANTED = ["3x faster", "57ms apart", "at 60fps", "12k frames", "its 3rd seed", "the 20th epoch", "off by .5 s",
           "off by 1.8E-04°", "nine seeds", "a dozen runs", "off by 0° on average", "trained on drive2019",
           "seed s9", "an Apple M2 chip", "twice as good", "a single seed", "half the frames", "one more run",
           "https://example.com/v=95"]


@pytest.mark.parametrize("plant", PLANTED)
def test_a_planted_figure_is_caught(plant):
    readme = document("README")
    at = readme.index("This is research software.")
    planted = readme[:at] + f"Planted: {plant}. " + readme[at:]
    left = [token for doc, token, _ in uncovered(READER, {"README": planted})]
    assert left, f"{plant!r} went through the scan"
    assert all(token in plant for token in left), f"the plant shifted another figure: {left}"


def _read(key: str) -> str:
    return (REPO / DOCUMENTS[key]).read_text(encoding="utf-8")


def prose_runs(text: str) -> list[list[str]]:
    """The lines of a document a reader reads as running text (not code, not a table row, not a heading, not an
    image or a badge line), in runs of lines that follow one another: a blank line, a line of any of those kinds
    and a code block each end a run."""
    runs: list[list[str]] = [[]]
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
            runs.append([])
        elif fenced:
            continue
        elif line.strip() and not line.startswith(("|", "#", "![", "[![")):
            runs[-1].append(line)
        else:
            runs.append([])
    return [run for run in runs if run]


def _prose(key: str) -> list[str]:
    """The lines of a document a reader reads as running text."""
    return [line for run in prose_runs(_read(key)) for line in run]


def _code(key: str) -> list[str]:
    """The lines inside the code blocks of a document."""
    lines, fenced = [], False
    for line in _read(key).splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif fenced:
            lines.append(line)
    return lines


#: the width the documents' running text and code blocks are wrapped to
WRAP = 100
#: the most words a sentence of the running text may hold
MAX_SENTENCE_WORDS = 50
#: what opens a list item: a bullet, or a number and its stop
LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
#: abbreviations that end in a stop and are followed by a space without ending a sentence ("e.g. a floor");
#: "etc." is not among them, as it can end one. The documents hold none of these but "et al.", which is
#: followed by a comma
ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "vs.", "cf.")
#: where a sentence ends: at the space after a full stop, a question mark or an exclamation mark, whatever the
#: next sentence begins with ("... as above. drive2017 has no timestamps"). A decimal (3.05) has no space after
#: its stop, and an abbreviation is left whole
SENTENCE_END = re.compile(r"(?<=[.!?])" + "".join(rf"(?<!\b{re.escape(a)})" for a in ABBREVIATIONS) + r"\s+")


def units(text: str) -> list[str]:
    """The running text as the pieces a sentence never runs across: each paragraph and each list item, its
    marker left off and its wrapped lines joined. A blank line, a code block, a table, a heading or the next
    list item ends one, so a lead-in that ends in a colon before a code block, or a list item with no stop, is
    not read on into the text after it."""
    out: list[str] = []
    for run in prose_runs(text):
        item: list[str] = []
        for line in run:
            if item and LIST_MARKER.match(line):
                out.append(" ".join(item))
                item = []
            item.append(LIST_MARKER.sub("", line, count=1))
        out.append(" ".join(item))
    return [_collapse(piece) for piece in out]


def sentences(pieces: list[str]) -> list[str]:
    """The sentences of ``pieces``, each piece cut at its sentence ends (``SENTENCE_END``)."""
    return [sentence for piece in pieces for sentence in SENTENCE_END.split(piece) if sentence]


def word_count(sentence: str) -> int:
    return len(sentence.split())


@pytest.mark.parametrize("key", DOCUMENTS)
def test_the_running_text_keeps_to_the_wrap(key):
    """A line of running text stays within the wrap unless a link's address alone carries it past."""
    long = [line for line in _prose(key) if len(line) > WRAP
            and len(re.sub(r"\]\([^)]*\)|<https?://[^>]*>", "", line)) > WRAP]
    assert long == [], long


def test_the_code_blocks_keep_to_the_wrap():
    """The lines of a code block, the repository tree's among them, stay within the wrap as well."""
    code = {key: _code(key) for key in DOCUMENTS}
    assert len(code["README"]) >= 50, "the scan found almost no code"
    assert {key: [line for line in lines if len(line) > WRAP] for key, lines in code.items()} == {
        key: [] for key in DOCUMENTS}


@pytest.mark.parametrize("key", DOCUMENTS)
def test_no_sentence_runs_on(key):
    """Running text is broken into sentences a reader can hold: none longer than fifty words."""
    longest = max(sentences(units(_read(key))), key=word_count)
    assert word_count(longest) <= MAX_SENTENCE_WORDS, longest


@pytest.mark.parametrize("text, count", [
    # a sentence may begin with a lower-case name, a digit, a link or a code span: the cut is at the stop
    ("It is as above. drive2017 has no timestamps, so there frame t−2 is the frame two back.", 2),
    ("The split is dealt. 90.9% of the test frames sit next to a training frame.", 2),
    ("It cannot run on file://. <http://localhost:4173/?selftest=1> runs the self-test.", 2),
    ("Install it. `dev` brings the tests. (The extras are optional.)", 3),
    ("Is it level? yes! It is. fine.", 4),
    # a decimal, a version, a thousands comma, a file name and a link have no space after their stop
    ("It is off by 3.05° against 10.09° on 1,073.2 s in Python 3.10 with data.txt, 0.jpg and "
     "arxiv.org/abs/1604.07316.", 1),
    # an abbreviation is not an end
    ("A bound, e.g. a floor, holds, i.e. it claims no more, as Bojarski et al. show, cf. the table, x vs. y.", 1),
])
def test_a_sentence_ends_at_a_stop_and_not_at_a_decimal_or_an_abbreviation(text, count):
    assert len(SENTENCE_END.split(text)) == count


def test_a_sentence_never_runs_across_a_blank_line_a_code_block_or_a_list_item():
    markdown = "\n".join([
        "# A heading",
        "Unzip each into its own folder, so that it holds the frames directly:",
        "",
        "```text",
        "data/",
        "```",
        "",
        "Then check them against the label files:",
        "```bash",
        "sightline data verify",
        "```",
        "It compares each file with its hash.",
        "",
        "- [Results](#results)",
        "- [How it works](#how-it-works)",
        "1. **Two frames.** Each frame is cut",
        "   to its bottom rows. The model takes two.",
        "| a | b |",
        "Last words with no stop",
    ])
    assert units(markdown) == [
        "Unzip each into its own folder, so that it holds the frames directly:",
        "Then check them against the label files:",
        "It compares each file with its hash.",
        "[Results](#results)",
        "[How it works](#how-it-works)",
        "**Two frames.** Each frame is cut to its bottom rows. The model takes two.",
        "Last words with no stop",
    ]


@pytest.mark.parametrize("key", DOCUMENTS)
@pytest.mark.parametrize("opening", ["A", "drive2017", "90.9%", "`dev`"])
@pytest.mark.parametrize("length, runs_on", [(MAX_SENTENCE_WORDS, False), (MAX_SENTENCE_WORDS + 1, True)])
def test_a_planted_sentence_over_the_limit_is_caught_whatever_it_begins_with(key, opening, length, runs_on):
    """A sentence of fifty words passes and one of fifty-one does not, however it begins (a capital, a lower-case
    name, a digit or a code span) and after whatever sentence: it is counted on its own."""
    planted = " ".join([opening, *["word"] * (length - 1)]) + "."
    assert word_count(planted) == length
    longest = max(sentences([*units(_read(key)), f"The one before it ends here. {planted}"]), key=word_count)
    assert (word_count(longest) > MAX_SENTENCE_WORDS) is runs_on, longest


#: words that name a piece of footage rather than the drive
FOOTAGE = re.compile(r"\b(?:clips?|demos?|excerpts?|footage)\b", re.IGNORECASE)


def lag_band_problems(r: Reader, written: dict[str, list[str]]) -> list[str]:
    """What is wrong with how the documents print the spread of the time between frames t−2 and t: the median
    and the band the middle 90% of frame pairs fall in. It is measured over the whole 2018 drive, so every
    sentence that carries it must say it is of that drive and name no clip, demo or excerpt. ``written`` is the
    sentences of each document."""
    gap = r.get(RESULTS, LAG_GAP)
    show = FORMATS["ms0"].show
    carries = re.compile("|".join([re.escape(f"median {show(gap['median'])} ms"),
                                   re.escape(f"{show(gap['p05'])}{ENDASH}{show(gap['p95'])} ms"),
                                   r"middle \d+% of frame pairs"]))
    problems = []
    for doc, found in written.items():
        carrying = [sentence for sentence in found if carries.search(sentence)]
        if not carrying:
            problems.append(f"{doc}: no sentence carries the spread, so nothing here holds it to the drive")
        for sentence in carrying:
            if "on the 2018 drive" not in sentence:
                problems.append(f"{doc}: the spread is not said to be the 2018 drive's: {sentence!r}")
            if FOOTAGE.search(sentence):
                problems.append(f"{doc}: the spread of the whole drive is put to a clip: {sentence!r}")
    return problems


def test_the_drive_wide_lag_band_is_never_attached_to_a_clip():
    written = {key: sentences(units(_read(key))) for key in DOCUMENTS}
    assert lag_band_problems(READER, written) == []


def _swap_words(text: str, old: str, new: str) -> str:
    """``text`` with the one place that says ``old``, wherever the document wraps it, saying ``new``."""
    pattern = re.compile(r"\s+".join(re.escape(word) for word in old.split()))
    assert len(pattern.findall(text)) == 1, f"{old!r} is not said exactly once"
    return pattern.sub(lambda _: new, text)


#: changes to the sentence that carries the spread, each putting it to a clip: (words it says, words that
#: replace them). "on the 2018 drive's demo clip" is the one an extended phrase alone must refuse
LAG_BAND_PLANTS = {
    "the drive's demo clip": ("on the 2018 drive, whose", "on the 2018 drive's demo clip, whose"),
    "the clip alone": ("on the 2018 drive, whose", "on the 2018 clip, whose"),
    "a clip named first": (f"Frames t{MINUS}2 and t are", f"In the demo clip, Frames t{MINUS}2 and t are"),
}


@pytest.mark.parametrize("key", DOCUMENTS)
@pytest.mark.parametrize("plant", [*LAG_BAND_PLANTS, "a second sentence"])
def test_a_lag_band_put_to_a_clip_is_caught(key, plant):
    raw = _read(key)
    if plant == "a second sentence":
        gap = READER.get(RESULTS, LAG_GAP)
        show = FORMATS["ms0"].show
        planted = raw + f"\n\nThe demo clip has the same spread, {show(gap['p05'])}{ENDASH}{show(gap['p95'])} ms.\n"
    else:
        planted = _swap_words(raw, *LAG_BAND_PLANTS[plant])
    assert planted != raw
    assert lag_band_problems(READER, {key: sentences(units(planted))}), f"{plant!r} went through the rule"


@pytest.mark.parametrize("key", DOCUMENTS)
@pytest.mark.parametrize("plant", ["the drive's demo clip", "the clip alone"])
def test_the_pins_of_the_lag_band_refuse_a_sentence_that_goes_on_about_a_clip(key, plant):
    """The pins hold the whole sentence, so a clip put after "the 2018 drive" breaks them as well as the rule."""
    pins = [c for c in CHECKS if c.doc == key and " ms apart (" in c.phrase]
    assert len(pins) == 4
    assert all(c.pattern(READER).search(document(key)) for c in pins)
    planted = _collapse(_swap_words(_read(key), *LAG_BAND_PLANTS[plant]))
    assert [c.pointer for c in pins if c.pattern(READER).search(planted)] == []


def test_a_rung_is_defined_before_it_is_used():
    """The results speak of rungs; the first time the README uses the word, it says what a rung is."""
    text = document("README")
    first = re.search(r"\brungs?\b", text, re.IGNORECASE)
    defined = re.search(r"The \w+ tests are the (rungs) of a ladder", text)
    assert first and defined and defined.start(1) == first.start(), first and text[first.start() - 80:first.end() + 20]
    assert not re.search(r"\brungs?\b", document("CARD"), re.IGNORECASE), "the card uses rung without defining it"


def _excerpt_names(text: str, folders: set[str]) -> set[str]:
    """The paths under web/public/data/ that ``text`` names, relative to it and with a star for the part that
    varies (``clips/*/sprite-*.jpg``): a folder of ``folders``, then a file name with an extension."""
    names = re.findall(r"(?<![\w*/.-])((?:[\w*-]+/)+[\w*-]+\.[a-z0-9]+)", text)
    return {name for name in names if name.split("/")[0] in folders}


def test_every_excerpt_of_the_dataset_the_repository_ships_is_named_in_the_licence_and_the_readme():
    """Everything under web/public/data/ but the JSON is an excerpt of the dataset's drives (a clip, the sheets of
    its frames, a thumbnail, the self-test frames), and so are the steering labels in two JSON files (the clips'
    ``labels_deg``, the leakage pairs' ``angle_deg``). The licence note and the README name them by path: each name
    matches a shipped file, and each shipped file is matched by a name."""
    data = REPO / "web" / "public" / "data"
    #: the two JSON files that carry the dataset's steering labels (a clip's ``labels_deg``, a pair's
    #: ``angle_deg``); every other JSON file holds this project's own measurements
    label_excerpts = {"clips/c2017/clip.json": "labels_deg", "clips/c2018/clip.json": "labels_deg",
                      "leakage/pairs.json": "angle_deg"}
    for name, key in label_excerpts.items():
        assert key in (data / name).read_text(encoding="utf-8"), f"{name} no longer carries {key}"
    shipped = sorted([path.relative_to(data).as_posix() for path in data.rglob("*")
                      if path.is_file() and not path.name.startswith(".") and path.suffix != ".json"]
                     + list(label_excerpts))
    folders = {name.split("/")[0] for name in shipped}
    assert len(shipped) >= 30 and len(folders) >= 3, shipped
    for label, text in (("LICENSE", (REPO / "LICENSE").read_text(encoding="utf-8")), ("README", _read("README"))):
        names = _excerpt_names(text, folders)
        assert names, f"{label} names no file under web/public/data/"
        unnamed = [path for path in shipped if not any(fnmatch.fnmatchcase(path, name) for name in names)]
        assert unnamed == [], f"{label} does not name {unnamed}"
        unshipped = [name for name in names if not any(fnmatch.fnmatchcase(path, name) for path in shipped)]
        assert unshipped == [], f"{label} names files that are not shipped: {unshipped}"


def test_a_figure_is_found_only_as_a_number_of_its_own():
    assert phrase_pattern("off by {}°", "3.05").search("off by 3.05°")
    assert not phrase_pattern("{}°", "3.05").search("off by 13.05°")
    assert not phrase_pattern("{}°", "3.05").search("off by 0.3.05°")
    assert not phrase_pattern("by {}", "3.0").search("off by 3.05")
    assert phrase_pattern("is {}.", "2.74").search("delivers is 2.74. Next")
