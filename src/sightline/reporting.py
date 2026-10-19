"""Run evaluation and report assembly.

``evaluate_run`` loads a trained run and scores it, on the CPU unless told
otherwise, on its own validation and test splits and on a whole second drive
("another day"), always beside the baselines. Shuffled and blocked runs are
also scored on their seed's paired frames: the frames BOTH protocols hold out
for testing, so the two models of a seed are compared on identical frames.
One JSON report per run.

``aggregate`` combines the run reports into the summary: statistics over
seeds per model and protocol, the paired leakage measurement, the
generalisation ladder (shuffled, blocked, later in the drive, another day),
the shipped model, chosen on validation MAE alone, and the evaluation's
cutoffs (``cutoffs``: the edges of the steering regimes, the smallest angle
the sign agreement counts and the tolerances of the shares of frames within
them, from ``sightline.evaluate``). It refuses reports that were not made
and scored alike: different library versions, drives, split settings,
evaluation device, training recipe (epochs, patience, learning rate, batch
size and weight decay for every run; augmentation and dropout per model),
frame history per model, frames behind a shared baseline, shares of frames
within other tolerances than the evaluation code's, or a checkpoint that is
no longer the run's best.pt.
Published documents read their numbers from these reports, nowhere else.

Reports are written as sorted, indented JSON of plain Python values with a
trailing newline, so evaluating the same run again yields the same text.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from pathlib import Path

import numpy as np
import torch

from sightline.data.cache import build_cache
from sightline.data.labels import load_labels, repair_dropouts
from sightline.data.splits import PROTOCOLS, PUBLIC_NAMES, frames_sha256, make_split, paired_frames
from sightline.evaluate import (
    REGIME_BINS,
    SIGN_MIN_ABS_DEG,
    TOLERANCES_DEG,
    regression_metrics,
    score_split,
    within_key,
)
from sightline.models import MODEL_SPECS, build_model
from sightline.paths import MissingDataError, UnusableRunError, Workspace
from sightline.train import code_version, predict

__all__ = [
    "CROSS_FIRST_FRAME",
    "LADDER",
    "SELECTED_BY",
    "aggregate",
    "checkpoint_sha256",
    "evaluate_run",
    "evaluation_cutoffs",
    "finished_runs",
    "frames_sha256",
    "load_run_model",
    "load_split_indices",
    "seed_stats",
    "select_shipped",
    "stale_reason",
    "write_report",
]

#: the cross drive is scored from this frame on, so every model (history lag up
#: to 2 frames) and every baseline (two labels of history) sees the same frames
CROSS_FIRST_FRAME = 2

#: protocols scored on the paired frames
PAIRED_PROTOCOLS = ("leaky", "blocked")

#: the training recipe a run report copies from its history.json: every run of a summary
#: shares the first five, and every run of one model shares its augmentation and dropout
SHARED_RECIPE = ("epochs", "patience", "lr", "batch_size", "weight_decay")
MODEL_RECIPE = ("augment", "dropout")

#: the generalisation ladder, in order: (rung, protocol, split)
LADDER = (
    (PUBLIC_NAMES["leaky"], "leaky", "test"),
    (PUBLIC_NAMES["blocked"], "blocked", "test"),
    (PUBLIC_NAMES["forward"], "forward", "test"),
    ("another day", "blocked", "cross_drive"),
)

#: the shipped model is the seed of this group with the lowest validation MAE;
#: test and cross-drive scores never take part in the choice
SHIPPED_MODEL, SHIPPED_PROTOCOL = "sightlinenet", "blocked"
SELECTED_BY = "val_mae_deg"

_RUN_NAME = re.compile(
    r"(?P<drive>.+)_(?P<model>{})_(?P<protocol>{})_s(?P<seed>\d+)".format(
        "|".join(re.escape(m) for m in sorted(MODEL_SPECS, key=len, reverse=True)),
        "|".join(PROTOCOLS),
    )
)


def load_run_model(run_dir: Path) -> tuple[torch.nn.Module, dict]:
    """Construct a run's model from its recorded config and load its best weights."""
    for name in ("history.json", "best.pt"):
        if not (run_dir / name).is_file():
            raise MissingDataError(f"run {run_dir.name!r} not found: expected {run_dir / name}")
    history = json.loads((run_dir / "history.json").read_text())
    config = history["config"]
    if "frame_lags" not in config:
        raise UnusableRunError(
            f"run {run_dir.name!r} records no frame_lags in its history.json, so the input "
            "its model expects is unknown; train it again"
        )
    model = build_model(
        config["model"], frame_lags=tuple(config["frame_lags"]), dropout=config["dropout"]
    )
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, history


def load_split_indices(run_dir: Path, split: str) -> np.ndarray:
    """The frame indices a run saved for ``split`` (train, val or test)."""
    path = Path(run_dir) / f"{split}_idx.npy"
    if not path.is_file():
        raise MissingDataError(f"run {Path(run_dir).name!r} is incomplete: expected {path}")
    return np.load(path)


def finished_runs(runs_dir: Path) -> list[Path]:
    """Every ``<drive>_<model>_<protocol>_s<seed>`` folder of ``runs_dir`` holding a history.json."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise MissingDataError(f"no runs folder at {runs_dir}")
    return sorted(
        p for p in runs_dir.iterdir()
        if p.is_dir() and _RUN_NAME.fullmatch(p.name) and (p / "history.json").is_file()
    )


def checkpoint_sha256(run_dir: Path) -> str:
    """sha256 of a run's best.pt: the weights a report describes."""
    digest = hashlib.sha256()
    with open(Path(run_dir) / "best.pt", "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stale_reason(report_path: Path, run_dir: Path) -> str | None:
    """Why the report at ``report_path`` no longer describes ``run_dir``, or ``None``
    when it does (same checkpoint). A report that cannot be read is stale too."""
    try:
        report = json.loads(Path(report_path).read_text())
    except (OSError, ValueError):
        return "its report cannot be read"
    if not (Path(run_dir) / "best.pt").is_file():
        return "its run has no best.pt"  # evaluating it names the missing file
    if not isinstance(report, dict) or report.get("checkpoint_sha256") != checkpoint_sha256(run_dir):
        return "its report was made from another checkpoint"
    if "train_config" not in report:
        return "its report records no training recipe"
    return None


def write_report(path: Path, data: dict) -> None:
    """Write sorted, indented JSON with a trailing newline; replace the file atomically,
    so an interrupted write never leaves a half report that looks finished."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    os.replace(partial, path)


def evaluation_cutoffs() -> dict:
    """The cutoffs every split is scored with, in degrees: the edges of the steering regimes
    (``regime_bins_deg``: straight below the first, sharp from the second on), the smallest
    |angle| the sign agreement counts (``sign_min_abs_deg``) and the tolerances of the shares of
    frames within them (``tolerances_deg``)."""
    return {"regime_bins_deg": [float(edge) for edge in REGIME_BINS], "sign_min_abs_deg": float(SIGN_MIN_ABS_DEG),
            "tolerances_deg": [float(tolerance) for tolerance in TOLERANCES_DEG]}


def _versions() -> dict:
    return {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()}


def _load_drive(workspace: Workspace, name: str):
    labels = repair_dropouts(load_labels(workspace.drive(name)))
    cache = build_cache(labels, workspace.cache_dir(name))
    dt_s = labels.median_dt_s() if labels.has_timestamps else None
    info = {
        "frames": len(cache),
        "repaired_labels": int(labels.repaired_indices.size),
        "median_dt_s": dt_s,
        "label_sha256": cache.meta.get("label_sha256"),
    }
    return cache, dt_s, info


def _score(model, cache, indices: np.ndarray, train_mean: float, dt_s, device) -> dict:
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    angles = cache.angles_deg.astype(np.float64)
    pred = predict(model, cache, indices, device).astype(np.float64)
    return score_split(pred, angles[indices], angles, indices, train_mean, dt_s)


def _check_split(run_name: str, saved: dict, protocol: str, n: int, seed: int, notes: dict) -> None:
    """Refuse a run whose saved split is not the split its protocol makes: its paired
    frames could then include frames it trained on."""
    if "firewall" not in notes or "block" not in notes:
        raise UnusableRunError(
            f"run {run_name!r} records no firewall/block in its history.json, so its split "
            "cannot be checked; train it again"
        )
    try:
        expected = make_split(protocol, n, seed=seed, firewall=notes["firewall"], block=notes["block"])
    except ValueError as exc:
        raise UnusableRunError(f"run {run_name!r}: cannot make its {protocol} split again: {exc}") from None
    for split, indices in saved.items():
        if not np.array_equal(indices, getattr(expected, split)):
            raise UnusableRunError(
                f"run {run_name!r}: {split}_idx.npy is not the {protocol} split of its drive "
                f"({n} frames, seed {seed}, firewall {notes['firewall']}, block {notes['block']})"
            )


def evaluate_run(
    run_dir: Path,
    workspace: Workspace,
    cross_drive: str | None = "drive2017",
    out_path: Path | None = None,
    device: str = "cpu",
) -> dict:
    """Score one run and return (and with ``out_path``, write) its report.

    ``device`` is used as given; evaluation never picks one on its own, so
    published numbers come from the CPU unless a device is named.
    """
    run_dir = Path(run_dir)
    model, history = load_run_model(run_dir)
    saved = {split: load_split_indices(run_dir, split) for split in ("train", "val", "test")}
    if cross_drive is not None:
        workspace.drive(cross_drive)  # a missing cross drive fails before any scoring
    config = history["config"]
    notes = config["notes"]
    protocol, seed = notes["protocol"], int(config["seed"])

    cache, dt_s, drive_info = _load_drive(workspace, notes["drive"])
    n = len(cache)
    _check_split(run_dir.name, saved, protocol, n, seed, notes)
    recorded = notes.get("repaired_labels")
    if recorded is not None and recorded != drive_info["repaired_labels"]:
        raise UnusableRunError(
            f"run {run_dir.name!r} trained on {recorded} repaired labels, but drive "
            f"{notes['drive']!r} now has {drive_info['repaired_labels']}; its labels have changed"
        )

    torch_device = torch.device(device)
    model.to(torch_device)
    angles = cache.angles_deg.astype(np.float64)
    train_mean = float(angles[saved["train"]].mean())

    report: dict = {
        "run": run_dir.name,
        "model": config["model"],
        "protocol": protocol,
        "seed": seed,
        "frame_lags": [int(lag) for lag in config["frame_lags"]],
        "n_params": int(history["n_params"]),
        "best_epoch": int(history["best_epoch"]),
        "epochs_run": len(history["history"]),
        "checkpoint_sha256": checkpoint_sha256(run_dir),
        "code_version": history.get("code_version"),
        "eval_code_version": code_version(),
        "train_device": history.get("device"),
        "train_config": {key: config.get(key) for key in (*SHARED_RECIPE, *MODEL_RECIPE)},
        "device": str(torch_device),
        "versions": _versions(),
        "drive": notes["drive"],
        "drives": {notes["drive"]: drive_info},
        "splits": {
            "train_n": int(saved["train"].size),
            "val_n": int(saved["val"].size),
            "test_n": int(saved["test"].size),
            "firewall": int(notes["firewall"]),
            "block": int(notes["block"]),
        },
        "train_mean_deg": train_mean,
        "val": _score(model, cache, saved["val"], train_mean, dt_s, torch_device),
        "test": _score(model, cache, saved["test"], train_mean, dt_s, torch_device),
    }

    if protocol in PAIRED_PROTOCOLS:
        frames = paired_frames(
            make_split("blocked", n, firewall=notes["firewall"], block=notes["block"]),
            make_split("leaky", n, seed=seed),
        )
        if frames.size == 0:
            raise UnusableRunError(f"run {run_dir.name!r}: the blocked and shuffled test sets share no frame")
        pred = predict(model, cache, frames, torch_device).astype(np.float64)
        report["paired"] = {
            "seed": seed,
            "n": int(frames.size),
            "frames_sha256": frames_sha256(frames),
            "model": regression_metrics(pred, angles[frames]),
            "zero_mae_deg": float(np.abs(angles[frames]).mean()),
        }

    if cross_drive is not None:
        c_cache, c_dt_s, c_info = _load_drive(workspace, cross_drive)
        report["drives"][cross_drive] = c_info
        everything = np.arange(CROSS_FIRST_FRAME, len(c_cache), dtype=np.int64)
        report["cross_drive"] = {
            "drive": cross_drive,
            **_score(model, c_cache, everything, train_mean, c_dt_s, torch_device),
        }

    if out_path is not None:
        write_report(out_path, report)
    return report


# --- the summary ------------------------------------------------------------------------

def seed_stats(values) -> dict:
    """``{"mean", "sd", "values"}`` over seeds; ``sd`` is the sample standard deviation
    (ddof=1), ``None`` for a single seed. Any undefined value makes mean and sd ``None``."""
    vals = [None if v is None else float(v) for v in values]
    if not vals:
        raise ValueError("no values to summarise")
    if any(v is None for v in vals):
        return {"mean": None, "sd": None, "values": vals}
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    return {"mean": float(np.mean(vals)), "sd": sd, "values": vals}


def _best_on_validation(group: dict) -> int:
    values = group[SELECTED_BY]["values"]
    if any(v is None for v in values):
        raise ValueError(f"a {SELECTED_BY} value is missing; cannot choose a seed")
    return int(np.argmin(values))  # ties go to the first, i.e. the lowest seed


def select_shipped(groups: dict) -> dict:
    """The ``sightlinenet/blocked`` seed with the lowest validation MAE."""
    key = f"{SHIPPED_MODEL}/{SHIPPED_PROTOCOL}"
    if key not in groups:
        raise ValueError(f"no {key} runs to choose the shipped model from")
    group = groups[key]
    best = _best_on_validation(group)
    return {
        "model": SHIPPED_MODEL,
        "protocol": SHIPPED_PROTOCOL,
        "seed": group["seeds"][best],
        "run": group["runs"][best],
        "selected_by": SELECTED_BY,
    }


def _runs_text(runs: list[str]) -> str:
    shown = ", ".join(runs[:3])
    return shown + (f" and {len(runs) - 3} more" if len(runs) > 3 else "")


def _require_one(reports: list[dict], value_of, what: str):
    """The one value ``value_of`` gives for every report, or a one-line ValueError
    naming each differing value with its runs (the rarest value first)."""
    runs_by_value: dict[str, list[str]] = {}
    for r in reports:
        runs_by_value.setdefault(json.dumps(value_of(r), sort_keys=True), []).append(r["run"])
    if len(runs_by_value) > 1:
        parts = sorted(runs_by_value.items(), key=lambda item: (len(item[1]), item[0]))
        raise ValueError(f"the reports disagree on the {what}: "
                         + "; ".join(f"{value} in {_runs_text(runs)}" for value, runs in parts))
    return value_of(reports[0])


def _group(members: list[dict]) -> dict:
    first = members[0]
    group = {
        "model": first["model"],
        "protocol": first["protocol"],
        "seeds": [r["seed"] for r in members],
        "runs": [r["run"] for r in members],
        "val_mae_deg": seed_stats(r["val"]["model"]["mae_deg"] for r in members),
        "test_mae_deg": seed_stats(r["test"]["model"]["mae_deg"] for r in members),
        "test_r": seed_stats(r["test"]["pearson_r"] for r in members),
        "test_skill": seed_stats(r["test"]["skill_vs_zero"] for r in members),
    }
    key = f"{first['model']}/{first['protocol']}"
    zero = [r["test"]["baselines"]["zero"]["mae_deg"] for r in members]
    if first["protocol"] == "leaky":
        # a shuffled split's test set is the seed's own, so its zero baseline is too
        group["test_zero_mae_deg"] = float(np.mean(zero))
    else:
        # every seed of a blocked or forward split is tested on the same frames
        group["test_zero_mae_deg"] = _require_one(
            members, lambda r: r["test"]["baselines"]["zero"]["mae_deg"],
            f"zero baseline on the {key} test split (the same frames for every seed)")
    if "cross_drive" in first:
        group["cross_mae_deg"] = seed_stats(r["cross_drive"]["model"]["mae_deg"] for r in members)
        group["cross_r"] = seed_stats(r["cross_drive"]["pearson_r"] for r in members)
        group["cross_skill"] = seed_stats(r["cross_drive"]["skill_vs_zero"] for r in members)
        group["cross_zero_mae_deg"] = first["cross_drive"]["baselines"]["zero"]["mae_deg"]
    return group


def _paired_leakage(model: str, by_key: dict, groups: dict) -> dict | None:
    leaky, blocked = groups.get(f"{model}/leaky"), groups.get(f"{model}/blocked")
    if leaky is None or blocked is None:
        return None
    seeds = sorted(set(leaky["seeds"]) & set(blocked["seeds"]))
    if not seeds:
        return None
    per_seed = []
    for seed in seeds:
        shuffled_score = by_key[(model, "leaky", seed)].get("paired")
        blocked_score = by_key[(model, "blocked", seed)].get("paired")
        if shuffled_score is None or blocked_score is None:
            raise ValueError(f"{model} seed {seed}: a shuffled or blocked report has no paired score")
        same = ("seed", "n", "frames_sha256")
        if any(shuffled_score[k] != blocked_score[k] for k in same):
            raise ValueError(f"{model} seed {seed}: the paired scores were taken on different frames")
        s_mae, b_mae = shuffled_score["model"]["mae_deg"], blocked_score["model"]["mae_deg"]
        per_seed.append({
            "seed": seed,
            "n": shuffled_score["n"],
            "shuffled_mae_deg": s_mae,
            "blocked_mae_deg": b_mae,
            "ratio": b_mae / s_mae,
            "difference_deg": b_mae - s_mae,
            "zero_mae_deg": shuffled_score["zero_mae_deg"],
        })
    return {
        "per_seed": per_seed,
        "ratio": seed_stats(p["ratio"] for p in per_seed),
        "difference_deg": seed_stats(p["difference_deg"] for p in per_seed),
        "shuffled_mae_deg": seed_stats(p["shuffled_mae_deg"] for p in per_seed),
        "blocked_mae_deg": seed_stats(p["blocked_mae_deg"] for p in per_seed),
    }


def _ladder(model: str, groups: dict, train_drive: str, cross_drive: str | None) -> list[dict]:
    rungs = []
    for rung, protocol, split in LADDER:
        group = groups.get(f"{model}/{protocol}")
        if group is None:
            continue
        if split == "test":
            mae, zero, drive = group["test_mae_deg"], group["test_zero_mae_deg"], train_drive
        elif "cross_mae_deg" in group:
            mae, zero, drive = group["cross_mae_deg"], group["cross_zero_mae_deg"], cross_drive
        else:
            continue
        rungs.append({"rung": rung, "protocol": protocol, "split": split, "drive": drive,
                      "seeds": group["seeds"], "mae_deg": mae, "zero_mae_deg": zero})
    return rungs


def _check_checkpoints(reports: list[dict], runs_dir: Path) -> None:
    stale = [r["run"] for r in reports
             if (Path(runs_dir) / r["run"] / "best.pt").is_file()
             and r.get("checkpoint_sha256") != checkpoint_sha256(Path(runs_dir) / r["run"])]
    if stale:
        raise ValueError(f"the reports of {_runs_text(stale)} were made from another checkpoint than "
                         f"the best.pt now in {runs_dir}; evaluate them again")


def aggregate(run_reports: list[dict], runs_dir: Path | None = None) -> dict:
    """The summary of a set of run reports (one per model, protocol and seed).

    With ``runs_dir``, every report whose run folder is there must describe the
    best.pt in it. A run folder that is absent (``runs/`` is not part of a
    clone) cannot be checked, and its report is taken as written.
    """
    reports = sorted(run_reports, key=lambda r: (r["model"], r["protocol"], r["seed"]))
    if not reports:
        raise ValueError("no run reports to summarise")
    by_key: dict = {}
    for r in reports:
        key = (r["model"], r["protocol"], r["seed"])
        if key in by_key:
            raise ValueError(f"{r['model']}/{r['protocol']} seed {r['seed']} is reported twice "
                             f"({by_key[key]['run']!r} and {r['run']!r})")
        by_key[key] = r

    if runs_dir is not None:
        _check_checkpoints(reports, runs_dir)
    versions = _require_one(reports, lambda r: r["versions"], "library versions")
    within = _require_one(reports, lambda r: sorted(k for k in r["test"]["model"] if k.startswith("within_")),
                          "shares of frames within a tolerance")
    if within != sorted(within_key(tolerance) for tolerance in TOLERANCES_DEG):
        raise ValueError(f"the reports give the shares {', '.join(within)}, but the evaluation code scores "
                         f"{', '.join(within_key(tolerance) for tolerance in TOLERANCES_DEG)}; evaluate them again")
    train_drive = _require_one(reports, lambda r: r["drive"], "training drive")
    device = _require_one(reports, lambda r: r["device"], "evaluation device")
    _require_one(reports, lambda r: {"firewall": r["splits"]["firewall"], "block": r["splits"]["block"]},
                 "split settings (firewall, block)")
    unrecorded = [r["run"] for r in reports if not isinstance(r.get("train_config"), dict)]
    if unrecorded:
        raise ValueError(f"the reports of {_runs_text(unrecorded)} record no training recipe; evaluate them again")
    recipe = _require_one(reports, lambda r: {k: r["train_config"].get(k) for k in SHARED_RECIPE},
                          f"training recipe ({', '.join(SHARED_RECIPE)})")
    cross_drive = _require_one(reports, lambda r: r.get("cross_drive", {}).get("drive"), "cross drive")
    if cross_drive is not None:
        _require_one(reports, lambda r: r["cross_drive"]["baselines"]["zero"]["mae_deg"],
                     "zero baseline on the cross drive (the same frames for every run)")
    recipe["models"] = {}
    for model in sorted({r["model"] for r in reports}):
        members = [r for r in reports if r["model"] == model]
        _require_one(members, lambda r: r["frame_lags"], f"frame_lags of {model}")
        recipe["models"][model] = _require_one(
            members, lambda r: {k: r["train_config"].get(k) for k in MODEL_RECIPE},
            f"augmentation and dropout of {model}")
    drives = {
        name: _require_one([r for r in reports if name in r["drives"]], lambda r: r["drives"][name],
                           f"drive {name!r}")
        for name in sorted({name for r in reports for name in r["drives"]})
    }

    groups: dict = {}
    for (model, protocol, _seed), r in by_key.items():
        groups.setdefault(f"{model}/{protocol}", []).append(r)
    groups = {key: _group(members) for key, members in groups.items()}
    models = sorted({r["model"] for r in reports})

    shipped = select_shipped(groups)
    twin_key = (SHIPPED_MODEL, "leaky", shipped["seed"])
    if twin_key not in by_key:
        raise ValueError(f"the shipped run {shipped['run']!r} has no shuffled-split twin "
                         f"({SHIPPED_MODEL}/leaky seed {shipped['seed']})")
    twin = by_key[twin_key]

    regimes: dict = {"protocol": SHIPPED_PROTOCOL, "split": "test", "selected_by": SELECTED_BY,
                     "zero": by_key[(SHIPPED_MODEL, SHIPPED_PROTOCOL, shipped["seed"])]["test"]["zero_regimes"],
                     "models": {}}
    for model in models:
        group = groups.get(f"{model}/{SHIPPED_PROTOCOL}")
        if group is not None:
            seed = group["seeds"][_best_on_validation(group)]
            chosen = by_key[(model, SHIPPED_PROTOCOL, seed)]
            regimes["models"][model] = {"run": chosen["run"], "seed": seed, "regimes": chosen["test"]["regimes"]}

    paired = {m: p for m in models if (p := _paired_leakage(m, by_key, groups)) is not None}

    def ordered_unique(values):
        return sorted(set(values), key=lambda v: (v is None, v or ""))

    return {
        "versions": versions,
        "device": device,
        "code_versions": {
            "train": ordered_unique(r["code_version"] for r in reports),
            "eval": ordered_unique(r.get("eval_code_version") for r in reports),
        },
        "runs": sorted(r["run"] for r in reports),
        "train_config": recipe,
        "groups": groups,
        "paired_leakage": paired,
        "ladder": {m: _ladder(m, groups, train_drive, cross_drive) for m in models},
        "shipped": shipped,
        "twin": {"run": twin["run"], "model": twin["model"], "protocol": twin["protocol"], "seed": twin["seed"]},
        "regimes": regimes,
        "drives": drives,
        "cutoffs": evaluation_cutoffs(),
    }
