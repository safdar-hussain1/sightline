import hashlib
import inspect
import json

import numpy as np
import pytest
from conftest import REPO
from sightline.data.splits import (BLOCK, BLOCK_PATTERN, FIREWALL, PROTOCOLS, TRAIN_FRAC, VAL_FRAC,
                                   blocked_split, chronological_split, frames_sha256, make_split,
                                   paired_frames, random_split)

N = 63825  # drive2018 length; pure index arithmetic, no data needed

@pytest.mark.parametrize("protocol", ["blocked", "forward"])
def test_honest_protocols_keep_every_eval_frame_501_frames_from_training(protocol):
    s = make_split(protocol, N, firewall=250)
    assert s.min_train_distance("val") >= 501
    assert s.min_train_distance("test") >= 501

def test_the_shuffled_protocol_reproduces_the_failure_mode():
    s = make_split("leaky", N, seed=0)
    assert s.min_train_distance("test") == 1
    train = np.sort(s.train)
    pos = np.searchsorted(train, s.test)
    left = np.abs(s.test - train[np.clip(pos - 1, 0, train.size - 1)])
    right = np.abs(train[np.clip(pos, 0, train.size - 1)] - s.test)
    assert np.median(np.minimum(left, right)) == 1

def test_blocked_split_sizes_on_drive2018_length():
    s = blocked_split(N)
    assert (s.train.size, s.val.size, s.test.size) == (39825, 6000, 6000)

def test_blocked_split_interleaves_eval_blocks_across_the_whole_drive():
    s = blocked_split(N)
    # test frames come from at least 4 separate stretches spread over the drive
    starts = np.r_[s.test[0], s.test[1:][np.diff(s.test) > 1]]
    assert len(starts) >= 4 and starts.min() < N / 4 and starts.max() > 3 * N / 4

def test_blocked_split_is_deterministic_and_disjoint():
    a, b = blocked_split(N), blocked_split(N)
    for name in ("train", "val", "test"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert not set(a.train) & set(a.test) and not set(a.val) & set(a.test)

@pytest.mark.parametrize("kwargs, msg", [
    ({"block": 400, "firewall": 250}, "block"),
    ({"pattern": "TTT"}, "pattern"),
    ({"pattern": "TXVE"}, "pattern"),
])
def test_blocked_split_rejects_bad_parameters(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        blocked_split(N, **kwargs)

def test_paired_frames_are_in_both_test_sets_and_nowhere_in_either_training_set():
    b, l = make_split("blocked", N), make_split("leaky", N, seed=0)
    p = paired_frames(b, l)
    assert p.size > 500
    assert set(p) <= set(b.test) and set(p) <= set(l.test)
    assert not set(p) & set(b.train) and not set(p) & set(l.train)
    assert np.all(np.diff(p) > 0)

def test_make_split_rejects_unknown_protocols():
    with pytest.raises(ValueError, match="protocol"):
        make_split("honest", N)
    assert PROTOCOLS == ("leaky", "blocked", "forward")

def test_forward_is_the_chronological_split():
    f, c = make_split("forward", N), chronological_split(N)
    for name in ("train", "val", "test"):
        np.testing.assert_array_equal(getattr(f, name), getattr(c, name))

def test_make_split_passes_block_and_firewall_through():
    m, b = make_split("blocked", N, block=700, firewall=100), blocked_split(N, block=700, firewall=100)
    for name in ("train", "val", "test"):
        np.testing.assert_array_equal(getattr(m, name), getattr(b, name))
    assert m.train.size != blocked_split(N).train.size
    f, c = make_split("forward", N, firewall=100), chronological_split(N, firewall=100)
    for name in ("train", "val", "test"):
        np.testing.assert_array_equal(getattr(f, name), getattr(c, name))

@pytest.mark.parametrize("protocol", ["blocked", "forward"])
@pytest.mark.parametrize("firewall", [0, 20, 250])
def test_the_closest_eval_frame_is_exactly_2_firewall_plus_1_from_training(protocol, firewall):
    s = make_split(protocol, N, firewall=firewall)
    assert min(s.min_train_distance("val"), s.min_train_distance("test")) == 2 * firewall + 1

@pytest.mark.parametrize("protocol", ["blocked", "forward"])
def test_a_negative_firewall_is_rejected(protocol):
    with pytest.raises(ValueError, match="firewall must be non-negative"):
        make_split(protocol, N, firewall=-1)

@pytest.mark.parametrize("n", [1, 2000, 3001, 7500, 7501])
def test_blocked_split_says_when_the_drive_is_too_short(n):
    # 1500-frame blocks TTVTTET: no validation frame before 3000 (and a 1-frame
    # validation block at 3000 sits inside the firewall); no test frame before 7500
    with pytest.raises(ValueError, match=f"n={n} too small for the blocked split"):
        blocked_split(n)


# --- the rules every split is made with ---------------------------------------------------

def _defaults(function) -> dict:
    return {name: p.default for name, p in inspect.signature(function).parameters.items()
            if p.default is not p.empty}


def test_every_split_function_defaults_to_the_published_rules():
    assert _defaults(blocked_split) == {"block": BLOCK, "firewall": FIREWALL, "pattern": BLOCK_PATTERN}
    assert _defaults(chronological_split) == {"train_frac": TRAIN_FRAC, "val_frac": VAL_FRAC, "firewall": FIREWALL}
    assert _defaults(random_split) == {"train_frac": TRAIN_FRAC, "val_frac": VAL_FRAC, "seed": 0}
    assert _defaults(make_split) == {"seed": 0, "firewall": FIREWALL, "block": BLOCK}


def test_the_commands_and_analyses_default_to_the_published_rules():
    from sightline.cli import build_parser
    from sightline.leakage import leakage_report
    from sightline.legacy import evaluate_legacy

    for function in (leakage_report, evaluate_legacy):
        found = _defaults(function)
        assert (found["firewall"], found["block"]) == (FIREWALL, BLOCK), function.__name__
    parser = build_parser()
    for argv in (["train", "--drive", "drive2018"], ["leakage", "--drive", "drive2018"], ["legacy"]):
        args = parser.parse_args(argv)
        assert (args.firewall, args.block) == (FIREWALL, BLOCK), argv


#: sha256 of the sha256 digests of the train, validation and test frames, in that order, of the
#: split each protocol makes of the 63,825 frames of drive2018 with the published rules, by
#: protocol and seed (blocked and forward splits do not depend on the seed). Each is the digest
#: of the index sets, in order, that the trained runs of that protocol and seed saved.
TRAINED_SPLITS = {
    ("blocked", 0): "2066358478a3fc9c3cbe4fec84af3ba5a525e5e750a34831c3d1fe40b59f69d1",
    ("forward", 0): "0ce3da5f692215b453815f0c1a4498867bf7d234b5373c10147c06149b708814",
    ("leaky", 0): "d1abf670319fbbe160986aed82ce1679b0cb49678c1616d1ec028abee1a88f0f",
    ("leaky", 1): "d662cf04ae7e878613130c2aaccf5c76b3e8544aea59a04181e0124dfdbe3117",
    ("leaky", 2): "4f3276a4559cbbde78739e2e86ef649e1c0a41c7d8dffcf76ab698bc6687ae8e",
}


@pytest.mark.parametrize("protocol, seed", sorted(TRAINED_SPLITS))
def test_the_default_rules_make_exactly_the_splits_the_runs_were_trained_on(protocol, seed):
    made = make_split(protocol, N, seed=seed)
    digest = hashlib.sha256("".join(frames_sha256(getattr(made, name))
                                    for name in ("train", "val", "test")).encode()).hexdigest()
    assert digest == TRAINED_SPLITS[protocol, seed]


def _committed_runs() -> tuple[dict, list[dict]]:
    reports = REPO / "reports"
    summary = json.loads((reports / "summary.json").read_text())
    return summary, [json.loads((reports / "runs" / f"{name}.json").read_text()) for name in summary["runs"]]


def test_the_default_rules_make_the_split_sizes_of_every_committed_run_report():
    summary, runs = _committed_runs()
    assert runs and {r["protocol"] for r in runs} == set(PROTOCOLS)
    frames = summary["drives"]["drive2018"]["frames"]
    assert frames == N
    for report in runs:
        recorded = report["splits"]
        # made with the splits module's own settings, not the ones the report records
        made = make_split(report["protocol"], frames, seed=report["seed"])
        assert (made.train.size, made.val.size, made.test.size) == \
            (recorded["train_n"], recorded["val_n"], recorded["test_n"]), report["run"]
        assert (recorded["firewall"], recorded["block"]) == (FIREWALL, BLOCK), report["run"]


def test_the_default_rules_make_the_paired_frames_the_committed_reports_name():
    summary, runs = _committed_runs()
    paired = [r for r in runs if "paired" in r]
    assert {r["protocol"] for r in paired} == {"leaky", "blocked"}
    frames = summary["drives"]["drive2018"]["frames"]
    leakage = json.loads((REPO / "reports" / "leakage.json").read_text())
    named = [(r["run"], r["seed"], r["paired"]) for r in paired]
    named.append(("leakage.json", leakage["seed"], leakage["paired"]))
    for name, seed, recorded in named:
        both = paired_frames(make_split("blocked", frames), make_split("leaky", frames, seed=seed))
        assert (int(both.size), frames_sha256(both)) == (recorded["n"], recorded["frames_sha256"]), name
