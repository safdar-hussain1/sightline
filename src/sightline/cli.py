"""Command-line interface.

Subcommands only — no top-level option shares a name with a subcommand
option, so nothing can shadow silently.

    sightline data verify [--drive drive2018 ...]
    sightline cache   --drive drive2018
    sightline train   --drive drive2018 --model sightlinenet --protocol blocked --seed 0
    sightline eval    --run drive2018_sightlinenet_blocked_s0 [--cross-drive drive2017|none] [--out PATH]
    sightline eval    --all [--out-dir reports/runs] [--overwrite]
    sightline report  [--in-dir reports/runs] [--out reports/summary.json]
    sightline figures [--summary reports/summary.json] [--out-dir reports/figures]
    sightline leakage --drive drive2018 --seed 0 [--out reports/leakage.json]
    sightline legacy  --weights PATH [--out reports/legacy.json]
    sightline export  --run drive2018_sightlinenet_blocked_s0 --out PATH.onnx
    sightline site-data [--summary reports/summary.json] [--out-dir web/public] [--reports-dir reports]
    sightline demo    --drive drive2017 --run runs/... --out demo.mp4

``--drive`` takes a drive name, looked up under ``--data-dir`` (default
``data``); caches and training runs live under ``--runs-dir`` (default
``runs``). Both options belong to each subcommand.

``train`` writes ``<runs-dir>/<drive>_<model>_<protocol>_s<seed>`` and never
touches an existing run unless given ``--overwrite``. ``eval`` scores runs on
the CPU unless ``--device`` names another; ``eval --all`` writes one report per
finished run and skips a run whose report describes its current best.pt
(``--overwrite`` evaluates every run again). ``report`` summarises the run
reports and ``figures`` draws the summary. ``leakage`` measures how close the
training frames of each split sit to its test frames, in time and in pixels,
and scores the lookup table the shuffled split rewards: copy the label of the
most similar training frame. ``legacy`` scores the standard-recipe PilotNet
(shuffled split, MSE in radians, atan head) from its TF1 weights, which are
not part of the repository: ``--weights`` or ``SIGHTLINE_LEGACY_WEIGHTS``
names their .npz. ``export`` writes one run as an ONNX graph (steering angle and saliency)
after checking it against PyTorch; ``site-data`` writes everything the browser page loads
(both models, the demo clips, results.json, the self-test fixture and the leakage pairs)
from the summary, the reports and the checkpoints, and records it in reports/export.json.
"""

from __future__ import annotations

import argparse
import json
import os
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


def _drive_name(text: str) -> str:
    """``text`` if it names a drive under --data-dir; a path is refused, so nothing is
    read or cached outside the workspace."""
    if text in ("", ".", "..") or Path(text).name != text:
        raise CommandError(f"--drive takes a drive name under --data-dir, not a path: {text!r}")
    return text


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

    _drive_name(args.drive)
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


def _run_path(ws: Workspace, text: str) -> Path:
    """A bare run name is looked up under --runs-dir; anything with a slash is a path."""
    path = Path(text)
    return ws.runs_dir / text if len(path.parts) == 1 else path


def _headline(report: dict) -> dict:
    line = {
        "run": report["run"],
        "val_mae_deg": report["val"]["model"]["mae_deg"],
        "test_mae_deg": report["test"]["model"]["mae_deg"],
        "test_zero_mae_deg": report["test"]["baselines"]["zero"]["mae_deg"],
    }
    if "paired" in report:
        line["paired_mae_deg"] = report["paired"]["model"]["mae_deg"]
    if "cross_drive" in report:
        line["cross_drive_mae_deg"] = report["cross_drive"]["model"]["mae_deg"]
    return line


def cmd_eval(args: argparse.Namespace) -> int:
    from sightline.reporting import evaluate_run, finished_runs, stale_reason
    from sightline.train import resolve_device

    try:
        resolve_device(args.device)
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    if args.all and args.out:
        raise CommandError("--out goes with --run; --all writes one report per run into --out-dir")
    ws = _workspace(args)
    cross = None if args.cross_drive.lower() == "none" else args.cross_drive

    if args.run is not None:
        report = evaluate_run(_run_path(ws, args.run), ws, cross_drive=cross,
                              out_path=Path(args.out) if args.out else None, device=args.device)
        print(json.dumps(_headline(report), indent=2))
        return 0

    out_dir = Path(args.out_dir)
    runs = finished_runs(ws.runs_dir)
    if not runs:
        raise MissingDataError(f"no finished runs under {ws.runs_dir}; train one with `sightline train`")
    evaluated = 0
    for run in runs:
        out = out_dir / f"{run.name}.json"
        if out.exists() and not args.overwrite:
            # a run trained again since its report gets a new report: the old one
            # describes weights that are gone and could no longer be reproduced
            reason = stale_reason(out, run)
            if reason is None:
                print(f"skipped {run.name}: {out} describes its best.pt (--overwrite replaces it)", flush=True)
                continue
            print(f"re-evaluating {run.name}: {reason}", flush=True)
        report = evaluate_run(run, ws, cross_drive=cross, out_path=out, device=args.device)
        evaluated += 1
        print(f"evaluated {run.name}: test MAE {report['test']['model']['mae_deg']:.3f} deg -> {out}",
              flush=True)
    print(f"{evaluated} of {len(runs)} runs evaluated into {out_dir}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from sightline.reporting import aggregate, write_report

    in_dir = Path(args.in_dir)
    paths = sorted(in_dir.glob("*.json")) if in_dir.is_dir() else []
    if not paths:
        raise MissingDataError(f"no run reports in {in_dir}; write them with `sightline eval --all`")
    reports = []
    for path in paths:
        try:
            report = json.loads(path.read_text())
        except ValueError as exc:
            raise CommandError(f"{path} is not a JSON report: {exc}") from None
        if not isinstance(report, dict) or not {"run", "model", "protocol", "seed", "val", "test"} <= report.keys():
            raise CommandError(f"{path} is not a run report (`sightline eval` writes those)")
        reports.append(report)
    runs_dir = Path(args.runs_dir)
    try:
        summary = aggregate(reports, runs_dir=runs_dir)
    except ValueError as exc:
        raise CommandError(f"cannot summarise {in_dir}: {exc}") from None
    write_report(Path(args.out), summary)
    checked = sum((runs_dir / r["run"] / "best.pt").is_file() for r in reports)
    print(f"summarised {len(reports)} run reports into {args.out}; shipped {summary['shipped']['run']}")
    print(f"checked the checkpoints of {checked} of {len(reports)} runs against {runs_dir}"
          + ("" if checked == len(reports) else " (the others have no run folder here)"))
    for model, entry in summary["paired_leakage"].items():
        print(f"{model}: paired blocked/shuffled MAE ratio {entry['ratio']['mean']:.2f} "
              f"over seeds {[p['seed'] for p in entry['per_seed']]}")
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    summary_path = Path(args.summary)
    if not summary_path.is_file():
        raise MissingDataError(f"no summary at {summary_path}; write it with `sightline report`")
    try:
        from sightline.figures import write_figures
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] != "matplotlib":
            raise
        raise CommandError("figures need matplotlib: pip install -e '.[figures]'") from None
    try:
        written = write_figures(json.loads(summary_path.read_text()), Path(args.out_dir))
    except ValueError as exc:
        raise CommandError(f"cannot draw {summary_path}: {exc}") from None
    for path in written:
        print(f"wrote {path}")
    return 0


def cmd_leakage(args: argparse.Namespace) -> int:
    from sightline.data.splits import PUBLIC_NAMES
    from sightline.leakage import LOOKUP_PROTOCOLS, leakage_report
    from sightline.reporting import write_report

    drive = _drive_name(args.drive)
    labels, cache = _load_drive(_workspace(args), drive)
    try:
        report = leakage_report(cache, drive=drive, seed=args.seed, timestamps_s=labels.timestamps_s,
                                firewall=args.firewall, block=args.block)
    except ValueError as exc:
        raise CommandError(f"cannot analyse {drive!r} with --block {args.block} "
                           f"--firewall {args.firewall}: {exc}") from None
    write_report(Path(args.out), report)
    for protocol in LOOKUP_PROTOCOLS:
        gap, lookup = report["gaps"][protocol]["test"]["median_frames"], report["memoriser"][protocol]
        print(f"{PUBLIC_NAMES[protocol]} split: the median test frame is {gap:g} frame{'' if gap == 1 else 's'} "
              f"from training; copying the label of the most similar training frame scores MAE "
              f"{lookup['mae_deg']:.2f} deg on the {lookup['n']} paired frames")
    print(f"wrote {args.out}")
    return 0


def cmd_legacy(args: argparse.Namespace) -> int:
    from sightline.legacy import WEIGHTS_ENV, evaluate_legacy
    from sightline.reporting import write_report

    weights = args.weights or os.environ.get(WEIGHTS_ENV)
    if not weights:
        raise CommandError(f"no weights to score: pass --weights PATH (an .npz of the TF1 tensors "
                           f"Variable .. Variable_19) or set {WEIGHTS_ENV}")
    try:
        report = evaluate_legacy(Path(weights), Workspace(data_dir=Path(args.data_dir)),
                                 firewall=args.firewall, block=args.block)
    except ValueError as exc:
        raise CommandError(f"cannot evaluate the standard-recipe PilotNet: {exc}") from None
    write_report(Path(args.out), report)
    parity = report["port_parity"]
    print(f"the PyTorch port is within {parity['max_abs_diff_deg']:.1e} deg of the TF1 graph "
          f"on {parity['frames']} frames")
    for key, score in report["eval"].items():
        print(f"{key}: MAE {score['model']['mae_deg']:.2f} deg on {score['model']['n']} frames "
              f"(always straight ahead: {score['zero_mae_deg']:.2f} deg)")
    cap = report["atan_cap"]
    print(f"largest |prediction| {cap['max_abs_pred_deg']:.1f} deg; labels past 180 deg: "
          + ", ".join(f"{drive} {count['n']}" for drive, count in sorted(cap["labels_beyond_180"].items())))
    print(f"wrote {args.out}")
    return 0


def _need_onnx(command: str) -> None:
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] not in ("onnx", "onnxruntime"):
            raise
        raise CommandError(f"{command} needs onnx and onnxruntime: pip install -e '.[export]'") from None


def cmd_export(args: argparse.Namespace) -> int:
    from sightline.export.onnx_export import export_run

    _need_onnx("export")
    ws = _workspace(args)
    run = _run_path(ws, args.run)
    try:
        result = export_run(run, Path(args.out), ws, check_frames=args.check_frames)
    except (ValueError, RuntimeError) as exc:
        raise CommandError(f"cannot export run {run.name!r}: {exc}") from None
    print(json.dumps(result, indent=2))
    return 0


def cmd_site_data(args: argparse.Namespace) -> int:
    _need_onnx("site-data")
    from sightline.export.site_data import build_site_data

    summary = Path(args.summary)
    if not summary.is_file():
        raise MissingDataError(f"no summary at {summary}; write it with `sightline report`")
    reports_dir = Path(args.reports_dir)
    try:
        export = build_site_data(summary, _workspace(args), out_dir=Path(args.out_dir), reports_dir=reports_dir,
                                 clip_frames=args.clip_frames, video=not args.no_video)
    except (ValueError, RuntimeError) as exc:
        raise CommandError(f"cannot build the site data: {exc}") from None
    for key, model in export["models"].items():
        parity = model["parity"]
        print(f"{key}: {model['run']} -> {model['file']} ({model['bytes']:,} bytes); onnxruntime within "
              f"{parity['max_angle_diff_deg']:.1e} deg and {parity['max_saliency_diff']:.1e} saliency of "
              f"PyTorch on {parity['frames']} frame stacks")
    for cid, clip in export["clips"].items():
        mae = clip["clip_mae_deg"]
        print(f"{cid}: {clip['drive']} frames {clip['first_frame']}..{clip['first_frame'] + clip['n'] - 1}; "
              f"MAE on this clip: sightline {mae['sightline']:.2f} deg, shuffled {mae['shuffled']:.2f} deg, "
              f"straight ahead {mae['zero']:.2f} deg")
    print(f"wrote {args.out_dir} and {reports_dir / 'export.json'}")
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
    p.add_argument("--epochs", type=_int_from(1), default=20,
                   help="length of the cosine learning-rate schedule")
    p.add_argument("--patience", type=_int_from(0), default=0,
                   help="stop after this many epochs without a better validation MAE; "
                        "0 trains every epoch (the best-validation checkpoint is kept either way)")
    p.add_argument("--firewall", type=_int_from(0), default=250,
                   help="frames dropped on each side of every train/evaluation boundary")
    p.add_argument("--block", type=_int_from(1), default=1500,
                   help="block length of the blocked split (recorded for every protocol)")
    p.add_argument("--device", default="auto", choices=DEVICES)
    p.add_argument("--overwrite", action="store_true", help="replace an existing run of the same name")
    _add_locations(p)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("eval", help="evaluate trained runs against the baselines")
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", help="run name under --runs-dir, or a path to a run folder")
    which.add_argument("--all", action="store_true",
                       help="every finished run under --runs-dir, one report each in --out-dir")
    p.add_argument("--cross-drive", default="drive2017",
                   help="drive scored whole as 'another day', or 'none' (default: drive2017)")
    p.add_argument("--out", default=None, help="with --run: write the report JSON here")
    p.add_argument("--out-dir", default="reports/runs", help="with --all: folder for the reports")
    p.add_argument("--overwrite", action="store_true", help="with --all: evaluate runs already reported")
    p.add_argument("--device", default="cpu", choices=[d for d in DEVICES if d != "auto"],
                   help="device to evaluate on (default: cpu, the device published numbers come from)")
    _add_locations(p)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("report", help="summarise the run reports over seeds")
    p.add_argument("--in-dir", default="reports/runs", help="folder of run reports")
    p.add_argument("--out", default="reports/summary.json", help="where to write the summary")
    p.add_argument("--runs-dir", default="runs",
                   help="training runs, to check each report still describes its run's best.pt")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("figures", help="draw the summary's figures")
    p.add_argument("--summary", default="reports/summary.json", help="the summary to draw")
    p.add_argument("--out-dir", default="reports/figures", help="folder for the PNG figures")
    p.set_defaults(fn=cmd_figures)

    p = sub.add_parser("leakage", help="measure how close training frames sit to test frames, in time and pixels")
    p.add_argument("--drive", required=True, help="drive name under --data-dir")
    p.add_argument("--seed", type=_int_from(0), default=0, help="seed of the shuffled split")
    p.add_argument("--firewall", type=_int_from(0), default=250,
                   help="frames dropped on each side of every train/evaluation boundary, as in training")
    p.add_argument("--block", type=_int_from(1), default=1500,
                   help="block length of the blocked split, as in training")
    p.add_argument("--out", default="reports/leakage.json", help="where to write the report")
    _add_locations(p)
    p.set_defaults(fn=cmd_leakage)

    p = sub.add_parser("legacy", help="score the standard-recipe PilotNet from its TF1 weights")
    p.add_argument("--weights", default=None,
                   help="the .npz of its TF1 tensors (default: $SIGHTLINE_LEGACY_WEIGHTS)")
    p.add_argument("--firewall", type=_int_from(0), default=250,
                   help="firewall of the drive2018 test splits, as in training")
    p.add_argument("--block", type=_int_from(1), default=1500,
                   help="block length of the drive2018 blocked split, as in training")
    p.add_argument("--out", default="reports/legacy.json", help="where to write the report")
    p.add_argument("--data-dir", default="data", help="folder holding the drive folders")
    p.set_defaults(fn=cmd_legacy)

    p = sub.add_parser("export", help="export a run to ONNX with saliency output")
    p.add_argument("--run", required=True, help="run name under --runs-dir, or a path to a run folder")
    p.add_argument("--out", required=True, help="where to write the .onnx file")
    p.add_argument("--check-frames", type=_int_from(1), default=64,
                   help="test-split frames onnxruntime must reproduce PyTorch on")
    _add_locations(p)
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("site-data", help="write the browser page's models, demo clips and results from the reports")
    p.add_argument("--summary", default="reports/summary.json", help="the summary naming the shipped run")
    p.add_argument("--out-dir", default="web/public", help="the page's public folder")
    p.add_argument("--reports-dir", default="reports",
                   help="folder of the run, leakage and standard-recipe reports; export.json is written here")
    p.add_argument("--clip-frames", type=_int_from(8), default=420, help="scored frames per demo clip")
    p.add_argument("--no-video", action="store_true", help="write everything but the clip videos (they need ffmpeg)")
    _add_locations(p)
    p.set_defaults(fn=cmd_site_data)

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
