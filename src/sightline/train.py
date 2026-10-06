"""Training engine.

Deterministic (seeded) training of a steering regressor over a preprocessed
drive cache. Loss is Huber on angle/30 deg — quadratic within one scale unit,
linear beyond it, so a single 300-degree parking maneuver cannot dominate a
batch the way it does under plain MSE. Model selection is best-validation
MAE, not last checkpoint.

The augmentation recipe and dropout default to the model's registry entry
(``MODEL_SPECS``); ``history.json`` records the resolved values and the
model's ``frame_lags``, which is everything needed to construct the model again,
plus the commit the training code came from (``code_version``).

``history.json`` is written last and atomically, so its presence means the run
finished: a run directory without it is an interrupted run.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sightline.data.cache import DriveCache
from sightline.data.dataset import AUGMENT_RECIPES, SteeringDataset
from sightline.data.frames import stack_frames
from sightline.data.splits import SplitIndices
from sightline.models import MODEL_SPECS, build_model

__all__ = ["DEVICES", "TrainConfig", "code_version", "pick_device", "predict", "resolve_device",
           "train_model"]

HUBER_SCALE_DEG = 30.0
DEVICES = ("auto", "cpu", "mps", "cuda")


@dataclass
class TrainConfig:
    model: str = "sightlinenet"
    epochs: int = 30
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 0
    #: augmentation recipe and dropout; None means the model's registry default
    augment: str | None = None
    dropout: float | None = None
    patience: int = 8
    notes: dict = field(default_factory=dict)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(name: str = "auto") -> torch.device:
    """``auto`` picks the fastest device present; a named device must be present."""
    if name == "auto":
        return pick_device()
    available = {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}
    if name not in available:
        raise ValueError(f"unknown device {name!r}; expected one of {DEVICES}")
    if not available[name]:
        raise ValueError(f"device {name!r} is not available on this machine")
    return torch.device(name)


def code_version(repo: str | Path | None = None) -> str | None:
    """The commit the training code came from: ``git rev-parse HEAD``, plus
    ``+dirty`` when ``git status --porcelain -- src`` lists anything (modified or
    untracked source). ``None`` outside a git checkout.

    ``repo`` defaults to the checkout holding this package (``<repo>/src/sightline``);
    an installed copy has no checkout and gets ``None``.
    """
    if repo is None:
        package = Path(__file__).resolve().parent
        if package.parent.name != "src":
            return None
        repo = package.parent.parent
    repo = Path(repo).resolve()

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                              check=True).stdout.strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != repo:
            return None  # inside some other checkout, not one of its own
        head = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--", "src")
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("+dirty" if dirty else "")


def _resolve_recipe(cfg: TrainConfig) -> TrainConfig:
    """Fill ``augment``/``dropout`` left as None from the model's registry entry."""
    if cfg.model not in MODEL_SPECS:
        raise ValueError(f"unknown model {cfg.model!r}; expected one of {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[cfg.model]
    cfg = replace(
        cfg,
        augment=spec.augment if cfg.augment is None else cfg.augment,
        dropout=spec.dropout if cfg.dropout is None else cfg.dropout,
    )
    if cfg.augment not in AUGMENT_RECIPES:
        raise ValueError(f"unknown augmentation recipe {cfg.augment!r}; expected one of {AUGMENT_RECIPES}")
    return cfg


def _epoch_loader(
    cache: DriveCache,
    idx: np.ndarray,
    cfg: TrainConfig,
    epoch: int,
    train: bool,
    frame_lags: tuple[int, ...],
) -> DataLoader:
    ds = SteeringDataset(
        cache,
        idx,
        augment=cfg.augment if train else "none",
        seed=cfg.seed * 1000 + epoch,
        frame_lags=frame_lags,
    )
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train, num_workers=0, drop_last=train)


@torch.no_grad()
def predict(model: torch.nn.Module, cache: DriveCache, indices: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Predicted angles (degrees) for ``indices``, in order.

    Each frame is fed with the history frames ``model.frame_lags`` asks for.
    """
    model.eval()
    out = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        batch = torch.from_numpy(stack_frames(cache.inputs, chunk, model.frame_lags)).to(device)
        out.append(model(batch)[:, 0].float().cpu().numpy())
    return np.concatenate(out)


def train_model(
    cache: DriveCache,
    splits: SplitIndices,
    cfg: TrainConfig,
    run_dir: str | Path,
    device: torch.device | str | None = None,
) -> dict:
    """Train, select on validation MAE, save best weights + history.

    ``device`` defaults to ``pick_device()``.
    """
    cfg = _resolve_recipe(cfg)
    version = code_version()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device() if device is None else torch.device(device)

    model = build_model(cfg.model, dropout=cfg.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    huber = torch.nn.HuberLoss(delta=1.0)

    angles = cache.angles_deg
    history: list[dict] = []
    best = {"val_mae": float("inf"), "epoch": -1}
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        loader = _epoch_loader(cache, splits.train, cfg, epoch, train=True, frame_lags=model.frame_lags)
        train_loss = 0.0
        n_batches = 0
        for frames, targets in loader:
            frames = frames.to(device)
            targets = targets.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(frames)
            loss = huber(pred / HUBER_SCALE_DEG, targets / HUBER_SCALE_DEG)
            loss.backward()
            opt.step()
            train_loss += float(loss.detach())
            n_batches += 1
        sched.step()

        val_pred = predict(model, cache, splits.val, device)
        val_mae = float(np.abs(val_pred - angles[splits.val]).mean())
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(n_batches, 1),
                "val_mae_deg": val_mae,
                "lr": sched.get_last_lr()[0],
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(f"[{cfg.model}] epoch {epoch + 1}/{cfg.epochs} val MAE {val_mae:.3f} deg ({history[-1]['seconds']}s)", flush=True)

        if val_mae < best["val_mae"]:
            best = {"val_mae": val_mae, "epoch": epoch}
            torch.save(model.state_dict(), run_dir / "best.pt")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"[{cfg.model}] early stop at epoch {epoch + 1}", flush=True)
                break

    if best["epoch"] < 0:
        raise RuntimeError(
            f"[{cfg.model}] no finite validation MAE in {len(history)} epochs; no checkpoint saved"
        )

    summary = {
        "config": {**asdict(cfg), "frame_lags": list(model.frame_lags)},
        "code_version": version,
        "device": str(device),
        "n_params": n_params,
        "best_val_mae_deg": best["val_mae"],
        "best_epoch": best["epoch"],
        "history": history,
    }
    partial = run_dir / "history.json.partial"
    partial.write_text(json.dumps(summary, indent=2))
    os.replace(partial, run_dir / "history.json")
    return summary
