"""Evaluation harness: metrics, baselines, and regime breakdowns.

Steering-angle regression has a trap at each end. Most driving is straight,
so a model that always predicts zero already scores a deceptively decent
MAE; and consecutive frames are so correlated that copying the previous
frame's *label* (zero-order hold — an oracle that reads the answer sheet
one row up) is nearly unbeatable per-frame. Every Sightline result is
therefore published next to both baselines, and broken down by steering
regime so straight-road frames cannot hide curve behavior.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "regression_metrics",
    "regime_breakdown",
    "baseline_predictions",
    "smoothness",
    "REGIME_BINS",
]

#: |angle| boundaries for regimes: straight < 5, gentle 5-30, sharp >= 30
REGIME_BINS = (5.0, 30.0)


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
        "within_1p5_deg": float((err <= 1.5).mean()),
        "within_5_deg": float((err <= 5.0).mean()),
        "within_15_deg": float((err <= 15.0).mean()),
    }


def regime_breakdown(pred: np.ndarray, target: np.ndarray) -> dict:
    """Per-regime MAE plus the balanced (regime-averaged) MAE.

    Regimes are defined by the magnitude of the HUMAN angle, so every model
    is binned identically. ``balanced_mae_deg`` weights the three regimes
    equally — the number a straight-road-only model cannot game.
    """
    lo, hi = REGIME_BINS
    t = np.abs(np.asarray(target, dtype=np.float64))
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


def smoothness(values: np.ndarray, dt_s: float | None = None) -> dict:
    """Mean absolute frame-to-frame change of a chronological signal.

    Only meaningful when ``values`` are predictions over CONSECUTIVE frames.
    With ``dt_s`` the rate is also expressed per second.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        raise ValueError("need at least two consecutive values")
    step = float(np.abs(np.diff(values)).mean())
    out = {"mean_abs_step_deg": step}
    if dt_s is not None:
        out["mean_abs_rate_deg_per_s"] = step / dt_s
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
