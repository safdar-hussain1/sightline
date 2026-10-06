import numpy as np
import pytest
from sightline.data.splits import (PROTOCOLS, blocked_split, chronological_split,
                                   make_split, paired_frames, random_split)

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
    np.testing.assert_array_equal(f.test, c.test)
