"""Run evaluation and report assembly.

``evaluate_run`` loads a trained run, scores it on its own validation and
test splits plus (optionally) an entire held-out cross-drive, always beside
the baselines, and writes one JSON report. Published documents read their
numbers from these reports — nowhere else.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from sightline.data.cache import build_cache
from sightline.data.labels import load_labels, repair_dropouts
from sightline.evaluate import (
    baseline_predictions,
    regime_breakdown,
    regression_metrics,
    smoothness,
)
from sightline.models import build_model
from sightline.paths import MissingDataError, UnusableRunError, Workspace
from sightline.train import pick_device, predict

__all__ = ["evaluate_run", "load_run_model"]


def load_run_model(run_dir: Path) -> tuple[torch.nn.Module, dict]:
    """Construct a run's model from its recorded config and load its best weights."""
    for name in ("history.json", "best.pt"):
        if not (run_dir / name).is_file():
            raise MissingDataError(f"run {run_dir.name!r} not found: expected {run_dir / name}")
    history = json.loads((run_dir / "history.json").read_text())
    config = history["config"]
    if "frame_lags" not in config:
        raise UnusableRunError(
            f"run predates multi-frame support: {run_dir.name!r} records no frame_lags "
            "in history.json, so the input its model expects is unknown; train it again"
        )
    model = build_model(
        config["model"], frame_lags=tuple(config["frame_lags"]), dropout=config["dropout"]
    )
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, history


def _is_contiguous(indices: np.ndarray) -> bool:
    return bool(np.all(np.diff(indices) == 1))


def _score_split(
    model: torch.nn.Module,
    cache,
    labels,
    indices: np.ndarray,
    train_mean: float,
    device: torch.device,
) -> dict:
    angles = cache.angles_deg.astype(np.float64)
    target = angles[indices]
    pred = predict(model, cache, indices, device).astype(np.float64)

    entry = {
        "model": regression_metrics(pred, target),
        "regimes": regime_breakdown(pred, target),
        "baselines": {},
    }
    usable = indices >= 2
    for name, base_pred in baseline_predictions(angles, indices[usable], train_mean).items():
        entry["baselines"][name] = regression_metrics(base_pred, target[usable])

    if _is_contiguous(indices):
        dt = labels.median_dt_s() if labels.has_timestamps else None
        entry["smoothness"] = {
            "model": smoothness(pred, dt),
            "human": smoothness(target, dt),
        }
        if dt is not None:
            entry["median_dt_s"] = dt
    return entry


def evaluate_run(
    run_dir: Path,
    workspace: Workspace,
    cross_drive: str | None = None,
    out_path: Path | None = None,
) -> dict:
    model, history = load_run_model(run_dir)
    device = pick_device()
    model.to(device)

    drive_name = history["config"]["notes"]["drive"]
    labels = repair_dropouts(load_labels(workspace.drive(drive_name)))
    cache = build_cache(labels, workspace.cache_dir(drive_name))

    train_idx = np.load(run_dir / "train_idx.npy")
    val_idx = np.load(run_dir / "val_idx.npy")
    test_idx = np.load(run_dir / "test_idx.npy")
    train_mean = float(cache.angles_deg[train_idx].mean())

    report: dict = {
        "run": run_dir.name,
        "config": history["config"],
        "n_params": history["n_params"],
        "train_mean_deg": train_mean,
        "splits": {
            "train_n": int(train_idx.size),
            "val_n": int(val_idx.size),
            "test_n": int(test_idx.size),
        },
        "val": _score_split(model, cache, labels, np.sort(val_idx), train_mean, device),
        "test": _score_split(model, cache, labels, np.sort(test_idx), train_mean, device),
    }

    if cross_drive is not None:
        c_labels = repair_dropouts(load_labels(workspace.drive(cross_drive)))
        c_cache = build_cache(c_labels, workspace.cache_dir(cross_drive))
        all_idx = np.arange(2, len(c_cache), dtype=np.int64)
        report["cross_drive"] = {
            "drive": cross_drive,
            **_score_split(model, c_cache, c_labels, all_idx, train_mean, device),
        }

    report["headline"] = {
        "val_mae_deg": report["val"]["model"]["mae_deg"],
        "test_mae_deg": report["test"]["model"]["mae_deg"],
        "test_balanced_mae_deg": report["test"]["regimes"]["balanced_mae_deg"],
        "cross_drive_mae_deg": (
            report["cross_drive"]["model"]["mae_deg"] if cross_drive is not None else None
        ),
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
    return report
