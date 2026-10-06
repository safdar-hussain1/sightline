import numpy as np
import pytest

from sightline.evaluate import (baseline_predictions, contiguous_runs, pearson_r,
                                regime_breakdown, regression_metrics, score_split, sign_agreement)


def test_zero_baseline_mae_is_mean_absolute_angle():
    a = np.array([0.0, 5.0, -10.0, 20.0, 3.0])
    b = baseline_predictions(a, np.arange(2, 5), train_mean_deg=1.0)
    assert regression_metrics(b["zero"], a[2:])["mae_deg"] == pytest.approx(np.abs(a[2:]).mean())


def test_zoh_and_extrapolation_read_only_the_past():
    a = np.array([1.0, 2.0, 4.0, 8.0])
    b = baseline_predictions(a, np.array([2, 3]), 0.0)
    assert b["zoh"].tolist() == [2.0, 4.0] and b["extrap"].tolist() == [3.0, 6.0]


def test_sign_agreement_ignores_near_straight_frames():
    t = np.array([1.0, -1.0, 10.0, -10.0, 30.0])
    p = np.array([-5.0, 5.0, 2.0, 3.0, 1.0])
    assert sign_agreement(p, t) == pytest.approx(2 / 3)


def test_pearson_r_is_scale_free():
    t = np.linspace(-20, 20, 50)
    assert pearson_r(3 * t + 1, t) == pytest.approx(1.0)


def test_contiguous_runs_split_at_gaps():
    runs = contiguous_runs(np.array([3, 4, 5, 9, 10, 20]))
    assert [r.tolist() for r in runs] == [[3, 4, 5], [9, 10], [20]]


def test_skill_is_relative_to_zero_on_the_same_frames():
    a = np.r_[np.zeros(2), np.linspace(-30, 30, 40)]
    idx = np.arange(2, 42)
    s = score_split(pred=a[idx] * 0.5, target=a[idx], angles_all=a, indices=idx, train_mean=0.0, dt_s=None)
    assert s["skill_vs_zero"] == pytest.approx(0.5)


def test_balanced_mae_weights_regimes_equally():
    t = np.array([1.0, 1.0, 1.0, 10.0, 40.0])
    p = np.array([1.0, 1.0, 1.0, 12.0, 30.0])
    r = regime_breakdown(p, t)
    assert r["balanced_mae_deg"] == pytest.approx((0 + 2 + 10) / 3)


# --- beyond the core metrics ----------------------------------------------------------

def _angles(n=60):
    return 40.0 * np.sin(np.arange(n) / 5.0)


def test_smoothness_counts_only_steps_inside_contiguous_runs():
    a = _angles()
    idx = np.array([2, 3, 4, 10, 11, 16])  # runs [2,3,4], [10,11], [16]
    pred = np.array([0.0, 1.0, 3.0, 100.0, 104.0, -50.0])
    s = score_split(pred=pred, target=a[idx], angles_all=a, indices=idx, train_mean=0.0, dt_s=0.05)
    # steps 1, 2 and 4 -- never the jumps between runs
    assert s["smoothness"]["model"]["mean_abs_step_deg"] == pytest.approx(7 / 3)
    assert s["smoothness"]["model"]["mean_abs_rate_deg_per_s"] == pytest.approx(7 / 3 / 0.05)
    human = np.abs(np.r_[np.diff(a[[2, 3, 4]]), np.diff(a[[10, 11]])]).mean()
    assert s["smoothness"]["human"]["mean_abs_step_deg"] == pytest.approx(human)
    assert s["smoothness"]["n_steps"] == 3


def test_smoothness_has_no_rate_without_a_frame_interval():
    a = _angles()
    idx = np.arange(5, 25)
    s = score_split(pred=a[idx], target=a[idx], angles_all=a, indices=idx, train_mean=0.0, dt_s=None)
    assert "mean_abs_rate_deg_per_s" not in s["smoothness"]["model"]


def test_smoothness_is_none_when_no_two_frames_are_consecutive():
    a = _angles()
    idx = np.array([2, 5, 9, 16, 33, 40])
    s = score_split(pred=a[idx], target=a[idx], angles_all=a, indices=idx, train_mean=0.0, dt_s=None)
    assert s["smoothness"] is None


def test_zero_and_mean_score_every_frame_while_history_baselines_skip_the_first_two():
    a = _angles()
    idx = np.arange(0, 50)
    s = score_split(pred=a[idx], target=a[idx], angles_all=a, indices=idx, train_mean=3.0, dt_s=None)
    assert s["n"] == 50
    assert s["baselines"]["zero"]["n"] == s["baselines"]["mean"]["n"] == 50
    assert s["baselines"]["zoh"]["n"] == s["baselines"]["extrap"]["n"] == 48
    assert s["baselines"]["mean"]["mae_deg"] == pytest.approx(np.abs(a[idx] - 3.0).mean())


def test_split_score_carries_the_zero_baseline_per_regime():
    a = _angles()
    idx = np.arange(2, 50)
    s = score_split(pred=a[idx] * 0.9, target=a[idx], angles_all=a, indices=idx, train_mean=0.0, dt_s=None)
    assert s["zero_regimes"] == regime_breakdown(np.zeros(idx.size), a[idx])


def test_split_scores_need_increasing_indices():
    a = _angles()
    idx = np.array([5, 4, 6])
    with pytest.raises(ValueError, match="increasing"):
        score_split(pred=a[idx], target=a[idx], angles_all=a, indices=idx, train_mean=0.0, dt_s=None)


def test_a_constant_prediction_has_no_correlation():
    assert np.isnan(pearson_r(np.ones(5), np.arange(5.0)))
    a = _angles()
    idx = np.arange(2, 50)
    s = score_split(pred=np.full(idx.size, 1.0), target=a[idx], angles_all=a, indices=idx,
                    train_mean=0.0, dt_s=None)
    assert s["pearson_r"] is None  # JSON has no NaN


def test_sign_agreement_is_undefined_without_turning_frames():
    assert np.isnan(sign_agreement(np.ones(3), np.array([1.0, -2.0, 4.9])))


def test_a_zero_prediction_never_agrees_in_sign():
    assert sign_agreement(np.zeros(2), np.array([10.0, -10.0])) == 0.0


def test_split_scores_hold_only_plain_python_numbers():
    a = _angles()
    idx = np.arange(2, 50)
    s = score_split(pred=(a[idx] * 0.9).astype(np.float32), target=a[idx].astype(np.float32),
                    angles_all=a.astype(np.float32), indices=idx, train_mean=np.float32(1.5),
                    dt_s=np.float64(0.057))

    def walk(value):
        if isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        else:
            assert type(value) in (int, float, str, bool, type(None)), (type(value), value)

    walk(s)
