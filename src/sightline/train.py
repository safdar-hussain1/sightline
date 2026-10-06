"""Training engine.

Deterministic (seeded) training of a steering regressor over a preprocessed
drive cache. Loss is Huber on angle/30 deg — quadratic within one scale unit,
linear beyond it, so a single 300-degree parking maneuver cannot dominate a
batch the way it does under plain MSE. Model selection is best-validation
MAE, not last checkpoint.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sightline.data.cache import DriveCache
from sightline.data.dataset import SteeringDataset
from sightline.data.splits import SplitIndices
from sightline.models import build_model

__all__ = ["TrainConfig", "train_model", "pick_device", "predict"]

HUBER_SCALE_DEG = 30.0


@dataclass
class TrainConfig:
    model: str = "sightlinenet"
    epochs: int = 30
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 0
    augment: bool = True
    patience: int = 8
    notes: dict = field(default_factory=dict)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _epoch_loader(cache: DriveCache, idx: np.ndarray, cfg: TrainConfig, epoch: int, train: bool) -> DataLoader:
    ds = SteeringDataset(cache, idx, augment=train and cfg.augment, seed=cfg.seed * 1000 + epoch)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train, num_workers=0, drop_last=train)


@torch.no_grad()
def predict(model: torch.nn.Module, cache: DriveCache, indices: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Predicted angles (degrees) for ``indices``, in order."""
    model.eval()
    out = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        batch = torch.from_numpy(np.ascontiguousarray(cache.inputs[chunk])).to(device)
        out.append(model(batch)[:, 0].float().cpu().numpy())
    return np.concatenate(out)


def train_model(
    cache: DriveCache,
    splits: SplitIndices,
    cfg: TrainConfig,
    run_dir: str | Path,
) -> dict:
    """Train, select on validation MAE, save best weights + history."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device()

    model = build_model(cfg.model).to(device)
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
        loader = _epoch_loader(cache, splits.train, cfg, epoch, train=True)
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

    summary = {
        "config": asdict(cfg),
        "device": str(device),
        "n_params": n_params,
        "best_val_mae_deg": best["val_mae"],
        "best_epoch": best["epoch"],
        "history": history,
    }
    (run_dir / "history.json").write_text(json.dumps(summary, indent=2))
    return summary
