"""Evaluation harness: metrics, baselines, and regime breakdowns.

Steering-angle regression has a trap at each end. Most driving is straight,
so a model that always predicts zero already scores a deceptively decent
MAE; and consecutive frames are so correlated that copying the previous
frame's *label* (zero-order hold — an oracle that reads the answer sheet
one row up) is nearly unbeatable per-frame. Every Sightline result is
therefore published next to both baselines, and broken down by steering
regime so straight-road frames cannot hide curve behavior.

``score_split`` assembles one split's full score from predictions that were
already made; every value in it is a plain Python ``int``/``float``/``None``
so it serialises to JSON unchanged.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "regression_metrics",
    "regime_breakdown",
    "baseline_predictions",
    "contiguous_runs",
    "pearson_r",
    "score_split",
    "sign_agreement",
    "smoothness",
    "within_key",
    "REGIME_BINS",
    "SIGN_MIN_ABS_DEG",
    "TOLERANCES_DEG",
]

#: |angle| boundaries for regimes: straight < 5, gentle 5-30, sharp >= 30
REGIME_BINS = (5.0, 30.0)

#: frames steered less than this are left out of the sign agreement: near
#: straight, the sign of the wheel is noise
SIGN_MIN_ABS_DEG = 5.0

#: the errors, in degrees, that the shares of frames within them are given for
#: (``within_1p5_deg``, ``within_5_deg``, ``within_15_deg``)
TOLERANCES_DEG = (1.5, 5.0, 15.0)


def within_key(tolerance_deg: float) -> str:
    """The name of the share of frames whose error is at most ``tolerance_deg``:
    ``within_1p5_deg`` for 1.5, ``within_5_deg`` for 5."""
    return f"within_{tolerance_deg:g}_deg".replace(".", "p")


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Scalar error metrics in degrees."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {target.shape}")
    if pred.size == 0:
        raise ValueError("empty evaluation set")
    err = np.abs(pred - target)
    return {
        "n": int(pred.size),
        "mae_deg": float(err.mean()),
        "rmse_deg": float(np.sqrt(((pred - target) ** 2).mean())),
        "medae_deg": float(np.median(err)),
        **{within_key(tolerance): float((err <= tolerance).mean()) for tolerance in TOLERANCES_DEG},
    }


def regime_breakdown(pred: np.ndarray, target: np.ndarray) -> dict:
    """Per-regime MAE plus the balanced (regime-averaged) MAE.

    Regimes are defined by the magnitude of the HUMAN angle, so every model
    is binned identically. ``balanced_mae_deg`` weights the three regimes
    equally — the number a straight-road-only model cannot game.
    """
    lo, hi = REGIME_BINS
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    t = np.abs(target)
    masks = {
        "straight": t < lo,
        "gentle": (t >= lo) & (t < hi),
        "sharp": t >= hi,
    }
    out: dict = {}
    maes = []
    for name, mask in masks.items():
        if mask.sum() == 0:
            raise ValueError(f"regime {name!r} is empty on this split")
        mae = float(np.abs(pred[mask] - target[mask]).mean())
        out[name] = {"n": int(mask.sum()), "mae_deg": mae}
        maes.append(mae)
    out["balanced_mae_deg"] = float(np.mean(maes))
    return out


def smoothness(values: np.ndarray, dt_s: float | None = None,
               runs: list[np.ndarray] | None = None) -> dict:
    """Mean absolute frame-to-frame change of a chronological signal.

    Only steps between CONSECUTIVE frames count: ``runs`` lists the positions
    in ``values`` of each run of consecutive frames (default: all of
    ``values`` is one run), and the steps inside every run are pooled; a step
    across a gap between runs never counts. With ``dt_s`` the rate is also
    expressed per second.
    """
    values = np.asarray(values, dtype=np.float64)
    runs = [np.arange(values.size)] if runs is None else runs
    steps = [np.diff(values[run]) for run in runs if len(run) >= 2]
    if not steps:
        raise ValueError("need at least two consecutive values")
    step = float(np.abs(np.concatenate(steps)).mean())
    out = {"mean_abs_step_deg": step}
    if dt_s is not None:
        out["mean_abs_rate_deg_per_s"] = step / float(dt_s)
    return out


def baseline_predictions(
    angles: np.ndarray,
    eval_indices: np.ndarray,
    train_mean_deg: float,
) -> dict[str, np.ndarray]:
    """Reference predictors every model must face.

    ``zero``     always predicts 0 (most frames are near-straight).
    ``mean``     always predicts the train-split mean angle.
    ``zoh``      copies the PREVIOUS frame's ground-truth label. This is an
                 oracle — it reads the label stream a vision model never
                 sees — and is published as the upper bound that exposes how
                 much per-frame temporal correlation is worth.
    ``extrap``   linear extrapolation of the two previous labels (oracle,
                 same caveat).
    """
    angles = np.asarray(angles, dtype=np.float64)
    idx = np.asarray(eval_indices, dtype=np.int64)
    if idx.min() < 2:
        raise ValueError("baselines need indices >= 2 for label history")
    return {
        "zero": np.zeros(idx.size),
        "mean": np.full(idx.size, float(train_mean_deg)),
        "zoh": angles[idx - 1],
        "extrap": 2.0 * angles[idx - 1] - angles[idx - 2],
    }


def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation of predictions with the human angle; NaN when either is constant."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape or p.size < 2:
        raise ValueError(f"need two equal-length series of at least 2 values, got {p.shape} and {t.shape}")
    p = p - p.mean()
    t = t - t.mean()
    denom = math.sqrt(float((p * p).sum()) * float((t * t).sum()))
    if denom == 0.0:
        return float("nan")
    return float((p * t).sum() / denom)


def sign_agreement(pred: np.ndarray, target: np.ndarray, min_abs_deg: float = SIGN_MIN_ABS_DEG) -> float:
    """Fraction of frames with ``|human| >= min_abs_deg`` whose predicted sign matches.

    A prediction of exactly 0 matches neither sign. NaN when no frame turns that far.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    turning = np.abs(t) >= min_abs_deg
    if not turning.any():
        return float("nan")
    return float((np.sign(p[turning]) == np.sign(t[turning])).mean())


def contiguous_runs(indices: np.ndarray) -> list[np.ndarray]:
    """Split increasing frame indices into runs of consecutive frames."""
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) != 1) + 1
    return np.split(idx, cuts)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def score_split(
    pred: np.ndarray,
    target: np.ndarray,
    angles_all: np.ndarray,
    indices: np.ndarray,
    train_mean: float,
    dt_s: float | None,
) -> dict:
    """One split's full score: the model beside every baseline, on the same frames.

    ``indices`` are the scored frames in increasing order, ``pred``/``target``
    aligned with them; ``angles_all`` is the whole drive's label stream, which
    the history baselines (``zoh``, ``extrap``) read. ``zero`` and ``mean``
    score every frame, so ``skill_vs_zero = 1 - MAE / zero MAE`` compares like
    with like; the history baselines score the frames from index 2 on.

    Smoothness pools every frame-to-frame step inside the contiguous runs of
    at least two frames (``None`` when no two scored frames are consecutive);
    ``dt_s``, the drive's median frame interval, adds the rate per second.
    ``pearson_r`` and ``sign_agreement`` are ``None`` where they are undefined.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    angles_all = np.asarray(angles_all, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int64)
    if not (pred.shape == target.shape == idx.shape):
        raise ValueError(f"pred, target and indices differ in shape: {pred.shape}, {target.shape}, {idx.shape}")
    if idx.size and np.any(np.diff(idx) <= 0):
        raise ValueError("indices must be strictly increasing")
    train_mean = float(train_mean)

    model = regression_metrics(pred, target)
    baselines = {
        "zero": regression_metrics(np.zeros(idx.size), target),
        "mean": regression_metrics(np.full(idx.size, train_mean), target),
    }
    with_history = idx >= 2
    history = baseline_predictions(angles_all, idx[with_history], train_mean)
    for name in ("zoh", "extrap"):
        baselines[name] = regression_metrics(history[name], target[with_history])

    runs = contiguous_runs(idx)
    offsets = np.cumsum([0] + [run.size for run in runs])
    positions = [np.arange(start, start + run.size) for start, run in zip(offsets, runs) if run.size >= 2]
    if positions:
        dt = None if dt_s is None else float(dt_s)
        smooth = {
            "model": smoothness(pred, dt, positions),
            "human": smoothness(target, dt, positions),
            "n_steps": int(sum(p.size - 1 for p in positions)),
        }
    else:
        smooth = None

    return {
        "n": int(idx.size),
        "model": model,
        "regimes": regime_breakdown(pred, target),
        "zero_regimes": regime_breakdown(np.zeros(idx.size), target),
        "baselines": baselines,
        "pearson_r": _finite_or_none(pearson_r(pred, target)),
        "sign_agreement": _finite_or_none(sign_agreement(pred, target)),
        "skill_vs_zero": 1.0 - model["mae_deg"] / baselines["zero"]["mae_deg"],
        "smoothness": smooth,
    }
