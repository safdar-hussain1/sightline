"""Command-line interface.

Subcommands only — no top-level option shares a name with a subcommand
option, so nothing can shadow silently.

    sightline data verify [--drive drive2018 ...]
    sightline cache   --drive drive2018
    sightline train   --drive drive2018 --model sightlinenet --protocol blocked --seed 0
    sightline eval    --run runs/drive2018_sightlinenet_blocked_s0 [--cross-drive drive2017]
    sightline export  --run runs/... --out web/public/models/sightline.onnx
    sightline demo    --drive drive2017 --run runs/... --out demo.mp4

``--drive`` takes a drive name, looked up under ``--data-dir`` (default
``data``); caches and training runs live under ``--runs-dir`` (default
``runs``). Both options belong to each subcommand.

``train`` writes ``<runs-dir>/<drive>_<model>_<protocol>_s<seed>`` and never
touches an existing run unless given ``--overwrite``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sightline.paths import MissingDataError, UnusableRunError, Workspace

#: what ``train`` writes into a run directory; history.json is listed first
#: because a run without it reads as unfinished
RUN_FILES = ("history.json", "history.json.partial", "best.pt",
             "train_idx.npy", "val_idx.npy", "test_idx.npy")


class CommandError(Exception):
    """A request the command cannot carry out; reported as one line, exit 2."""


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

    from sightline.data.splits import make_split
    from sightline.train import TrainConfig, resolve_device, train_model

    if args.drive in ("", ".", "..") or Path(args.drive).name != args.drive:
        raise CommandError(f"--drive takes a drive name under --data-dir, not a path: {args.drive!r}")
    try:
        device = resolve_device(args.device)
    except ValueError as exc:
        raise CommandError(str(exc)) from None

    ws = _workspace(args)
    name = f"{args.drive}_{args.model}_{args.protocol}_s{args.seed}"
    run_dir = ws.runs_dir / name
    if run_dir.exists() and not args.overwrite:
        raise CommandError(f"run {name!r} already exists at {run_dir}; pass --overwrite to replace it")

    labels, cache = _load_drive(ws, args.drive)
    n = len(cache)
    try:
        splits = make_split(args.protocol, n, seed=args.seed, firewall=args.firewall, block=args.block)
    except ValueError as exc:
        raise CommandError(f"cannot split {args.drive!r} for --protocol {args.protocol}: {exc}") from None

    cfg = TrainConfig(
        model=args.model,
        epochs=args.epochs,
        seed=args.seed,
        patience=args.patience,
        notes={
            "drive": args.drive,
            "protocol": args.protocol,
            "firewall": args.firewall,
            "block": args.block,
            "repaired_labels": int(labels.repaired_indices.size),
        },
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in RUN_FILES:
        (run_dir / stale).unlink(missing_ok=True)
    np.save(run_dir / "train_idx.npy", splits.train)
    np.save(run_dir / "val_idx.npy", splits.val)
    np.save(run_dir / "test_idx.npy", splits.test)
    summary = train_model(cache, splits, cfg, run_dir, device=device)
    print(json.dumps({k: summary[k] for k in ("best_val_mae_deg", "best_epoch", "n_params", "code_version")},
                     indent=2))
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
            report = verify_drive(ws.drive(name, verify_hint=False), name)
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


def _int_from(minimum: int):
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {value}")
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    from sightline.data.splits import PROTOCOLS
    from sightline.models import MODEL_SPECS
    from sightline.train import DEVICES

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
    p.add_argument("--drive", required=True, help="drive name under --data-dir")
    p.add_argument("--model", default="sightlinenet", choices=list(MODEL_SPECS))
    p.add_argument("--protocol", default="blocked", choices=PROTOCOLS,
                   help="leaky = shuffled frames, blocked = interleaved blocks, forward = later in the drive")
    p.add_argument("--seed", type=_int_from(0), default=0)
    p.add_argument("--epochs", type=_int_from(1), default=30)
    p.add_argument("--patience", type=_int_from(1), default=8,
                   help="stop after this many epochs without a better validation MAE")
    p.add_argument("--firewall", type=_int_from(0), default=250,
                   help="frames dropped on each side of every train/evaluation boundary")
    p.add_argument("--block", type=_int_from(1), default=1500,
                   help="block length of the blocked split (recorded for every protocol)")
    p.add_argument("--device", default="auto", choices=DEVICES)
    p.add_argument("--overwrite", action="store_true", help="replace an existing run of the same name")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (MissingDataError, UnusableRunError, CommandError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
