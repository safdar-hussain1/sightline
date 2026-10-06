"""Command-line interface.

Subcommands only — no top-level option shares a name with a subcommand
option, so nothing can shadow silently.

    sightline data verify [--drive drive2018 ...]
    sightline cache   --drive drive2018
    sightline train   --drive drive2018 --model sightlinenet --protocol honest
    sightline eval    --run runs/drive2018_sightlinenet_honest_s0 [--cross-drive drive2017]
    sightline export  --run runs/... --out web/public/models/sightline.onnx
    sightline demo    --drive drive2017 --run runs/... --out demo.mp4

``--drive`` takes a drive name, looked up under ``--data-dir`` (default
``data``); caches and training runs live under ``--runs-dir`` (default
``runs``). Both options belong to each subcommand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sightline.paths import MissingDataError, Workspace


def _workspace(args: argparse.Namespace) -> Workspace:
    return Workspace(data_dir=Path(args.data_dir), runs_dir=Path(args.runs_dir))


def _load_drive(ws: Workspace, name: str, repair: bool = True):
    from sightline.data.cache import build_cache
    from sightline.data.labels import load_labels, repair_dropouts

    labels = load_labels(ws.drive(name))
    if repair:
        labels = repair_dropouts(labels)
    cache = build_cache(labels, ws.cache_dir(name))
    return labels, cache


def cmd_cache(args: argparse.Namespace) -> int:
    labels, cache = _load_drive(_workspace(args), args.drive, repair=not args.raw_labels)
    print(f"cached {len(cache)} frames from {args.drive} "
          f"({labels.repaired_indices.size} dropout repairs)")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    import numpy as np

    from sightline.data.splits import chronological_split, random_split
    from sightline.train import TrainConfig, train_model

    ws = _workspace(args)
    labels, cache = _load_drive(ws, args.drive)
    n = len(cache)
    if args.protocol == "honest":
        splits = chronological_split(n, firewall=args.firewall)
    elif args.protocol == "leaky":
        splits = random_split(n, seed=args.seed)
    else:
        raise ValueError(f"unknown protocol {args.protocol}")

    cfg = TrainConfig(
        model=args.model,
        epochs=args.epochs,
        seed=args.seed,
        notes={
            "drive": args.drive,
            "protocol": args.protocol,
            "firewall": args.firewall if args.protocol == "honest" else None,
            "repaired_labels": int(labels.repaired_indices.size),
        },
    )
    run_dir = ws.runs_dir / f"{args.drive}_{args.model}_{args.protocol}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "train_idx.npy", splits.train)
    np.save(run_dir / "val_idx.npy", splits.val)
    np.save(run_dir / "test_idx.npy", splits.test)
    summary = train_model(cache, splits, cfg, run_dir)
    print(json.dumps({k: summary[k] for k in ("best_val_mae_deg", "best_epoch", "n_params")}, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from sightline.reporting import evaluate_run

    report = evaluate_run(
        run_dir=Path(args.run),
        workspace=_workspace(args),
        cross_drive=args.cross_drive,
        out_path=Path(args.out) if args.out else None,
    )
    print(json.dumps(report["headline"], indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from sightline.export.onnx_export import export_run

    result = export_run(
        Path(args.run), Path(args.out), _workspace(args), check_frames=args.check_frames
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from sightline.demo import render_demo

    out = render_demo(
        drive_name=args.drive,
        run_dir=Path(args.run),
        out_path=Path(args.out),
        workspace=_workspace(args),
        start=args.start,
        num_frames=args.frames,
        fps=args.fps,
    )
    print(f"wrote {out}")
    return 0


def cmd_data_verify(args: argparse.Namespace) -> int:
    from sightline.data.verify import KNOWN_DRIVES, verify_drive

    ws = _workspace(args)
    failed = False
    for name in args.drive or sorted(KNOWN_DRIVES):
        try:
            report = verify_drive(ws.drive(name), name)
        except MissingDataError as exc:
            report = {"ok": False, "problems": [str(exc)]}
        if report["ok"]:
            print(f"{name}: OK {report['frames']} frames sha256 {report['label_sha256'][:8]}\u2026")
        else:
            failed = True
            print(f"{name}: PROBLEM {report['problems'][0]}")
    return 1 if failed else 0


def _add_locations(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", default="data", help="folder holding the drive folders")
    p.add_argument("--runs-dir", default="runs", help="folder for caches and training runs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sightline", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="check the dataset on disk")
    data_sub = p.add_subparsers(dest="data_command", required=True)
    p = data_sub.add_parser("verify", help="compare drives with the published checksums")
    p.add_argument("--drive", action="append", help="drive name; repeat for several (default: all known)")
    _add_locations(p)
    p.set_defaults(fn=cmd_data_verify)

    p = sub.add_parser("cache", help="preprocess a drive into the training cache")
    p.add_argument("--drive", required=True)
    p.add_argument("--raw-labels", action="store_true", help="skip dropout repair")
    _add_locations(p)
    p.set_defaults(fn=cmd_cache)

    p = sub.add_parser("train", help="train a steering model")
    p.add_argument("--drive", required=True)
    p.add_argument("--model", default="sightlinenet", choices=["sightlinenet", "pilotnet"])
    p.add_argument("--protocol", default="honest", choices=["honest", "leaky"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--firewall", type=int, default=250)
    _add_locations(p)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("eval", help="evaluate a trained run against the baselines")
    p.add_argument("--run", required=True)
    p.add_argument("--cross-drive", default=None, help="drive name to evaluate as held-out cross-drive test")
    p.add_argument("--out", default=None, help="write the full report JSON here")
    _add_locations(p)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("export", help="export a run to ONNX with saliency output")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--check-frames", type=int, default=64)
    _add_locations(p)
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("demo", help="render the cockpit demo video")
    p.add_argument("--drive", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", default="demo.mp4")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--fps", type=float, default=17.5)
    _add_locations(p)
    p.set_defaults(fn=cmd_demo)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except MissingDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
