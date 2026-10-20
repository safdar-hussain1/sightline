#!/usr/bin/env python3
"""Train the published matrix of runs, a few at a time, each in its own process.

    python scripts/train_matrix.py --runs-dir runs --parallel 3 --epochs 20 --patience 0 [--only REGEX] [--dry-run]

Twenty-one runs on drive2018: PilotNet and SightlineNet under each split
protocol (leaky, blocked, forward) and the single-frame SightlineNet under the
blocked protocol, each for seeds 0, 1 and 2. All seed-0 runs start first, then
seed 1, then seed 2, so a matrix stopped part-way still holds a complete
single-seed result.

Every run trains the full cosine schedule (``--epochs``, default 20) with no early
stopping (``--patience 0``, passed explicitly to each run): the schedule only reaches
low learning rates in its last epochs, and the best-validation checkpoint is kept.

Each run is ``python -m sightline.cli train ...`` with this checkout's ``src``
first on PYTHONPATH, logging to ``<runs-dir>/logs/<run>.log``. Before any run
starts, ``sightline cache`` builds (or checks) the drive's cache once, so
parallel runs never write it at the same time.

No run is ever trained twice. A run whose ``history.json`` exists is skipped;
if it was trained with other settings (epochs, patience, firewall, block, ...) it is
reported as a conflict, left untouched, and the matrix exits non-zero. A run
directory without ``history.json`` is an interrupted run: it is trained again
from scratch with ``--overwrite``, and both the log and the output say so.

The runner writes its pid to ``<runs-dir>/logs/matrix.pid`` and refuses to start
while another runner is alive. Stop it with ``kill <pid>`` (SIGTERM): the runs
in progress are stopped as well, and start from scratch on the next launch.
Exit status: 0 when every run is finished, 1 when any run failed, conflicted or
was stopped, 2 for unusable arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
# every run is trained by this checkout's code (see _child_env), so its split settings default to this
# checkout's splits module as well
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from sightline.data.splits import BLOCK, FIREWALL  # noqa: E402

SEEDS = (0, 1, 2)
#: (model, protocols), in the order each seed's runs start
MATRIX = (
    ("pilotnet", ("leaky", "blocked", "forward")),
    ("sightlinenet", ("leaky", "blocked", "forward")),
    ("sightlinenet-1f", ("blocked",)),
)
DEVICES = ("auto", "cpu", "mps", "cuda")
POLL_SECONDS = 1.0
STOP_GRACE_SECONDS = 30.0


class Job(NamedTuple):
    drive: str
    model: str
    protocol: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.drive}_{self.model}_{self.protocol}_s{self.seed}"


def matrix_jobs(drive: str) -> list[Job]:
    """Every run of the matrix, seed-major."""
    return [Job(drive, model, protocol, seed)
            for seed in SEEDS for model, protocols in MATRIX for protocol in protocols]


def cache_command(opts: argparse.Namespace) -> list[str]:
    return [sys.executable, "-m", "sightline.cli", "cache", "--drive", opts.drive,
            "--data-dir", opts.data_dir, "--runs-dir", opts.runs_dir]


def train_command(job: Job, opts: argparse.Namespace, overwrite: bool) -> list[str]:
    cmd = [sys.executable, "-m", "sightline.cli", "train",
           "--drive", job.drive, "--model", job.model, "--protocol", job.protocol,
           "--seed", str(job.seed), "--epochs", str(opts.epochs),
           "--patience", str(opts.patience),
           "--firewall", str(opts.firewall), "--block", str(opts.block), "--device", opts.device,
           "--data-dir", opts.data_dir, "--runs-dir", opts.runs_dir]
    return [*cmd, "--overwrite"] if overwrite else cmd


def run_state(job: Job, opts: argparse.Namespace) -> tuple[str, str]:
    """What is on disk for ``job``: ``new``, ``unfinished``, ``finished`` or ``conflict``, with a note."""
    run_dir = Path(opts.runs_dir) / job.name
    history = run_dir / "history.json"
    if os.path.lexists(history):
        try:
            recorded = json.loads(history.read_text())
            config, notes = recorded["config"], recorded["config"]["notes"]
            got = {"model": config.get("model"), "seed": config.get("seed"),
                   "epochs": config.get("epochs"), "patience": config.get("patience"), "drive": notes.get("drive"),
                   "protocol": notes.get("protocol"), "firewall": notes.get("firewall"),
                   "block": notes.get("block")}
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            return "conflict", f"has an unreadable history.json ({type(exc).__name__}: {exc})"
        wanted = {"model": job.model, "seed": job.seed, "epochs": opts.epochs, "patience": opts.patience, "drive": job.drive,
                  "protocol": job.protocol, "firewall": opts.firewall, "block": opts.block}
        differ = [f"{key}={got[key]!r} (matrix: {wanted[key]!r})" for key in wanted if got[key] != wanted[key]]
        if differ:
            return "conflict", "finished with " + ", ".join(differ)
        return "finished", f"finished, code {recorded.get('code_version')}"
    if os.path.lexists(run_dir):
        return "unfinished", "unfinished run without history.json; retraining from scratch with --overwrite"
    return "new", ""


def _say(tag: str, text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {tag:<8} {text}", flush=True)


def _live_runner(pid_file: Path) -> int | None:
    """The pid recorded in ``pid_file`` if that process is alive and is a matrix runner."""
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None
    if pid <= 0 or pid == os.getpid():
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass  # alive, owned by someone else
    try:
        ps = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True)
    except OSError:
        return pid  # cannot tell what it is, so assume it is a runner
    return pid if ps.returncode == 0 and "train_matrix" in ps.stdout else None


def _child_env() -> dict[str, str]:
    paths = [str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]
    return {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in paths if p), "PYTHONUNBUFFERED": "1"}


def _launch(cmd: list[str], log_path: Path, notes: list[str]):
    log = open(log_path, "a")
    for line in [*notes, shlex.join(cmd)]:
        log.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
    log.flush()
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                env=_child_env())
    except OSError:
        log.close()
        raise
    return proc, log


def _stop(proc: subprocess.Popen) -> int:
    proc.terminate()
    try:
        return proc.wait(timeout=STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait()


def _run(jobs: list[Job], opts: argparse.Namespace, logs: Path, stop: list[int]) -> int:
    t_start = time.monotonic()
    counts: Counter[str] = Counter()
    pending = []
    for job in jobs:
        state, note = run_state(job, opts)
        if state == "finished":
            _say("skip", f"{job.name} ({note})")
            counts["skipped"] += 1
        elif state == "conflict":
            _say("CONFLICT", f"{job.name} {note}; left untouched, move it aside to retrain it")
            counts["conflicts"] += 1
        else:
            pending.append(job)
    _say("matrix", f"pid {os.getpid()}: {len(pending)} to train, {counts['skipped']} finished, "
                   f"{counts['conflicts']} conflicts, parallel {opts.parallel}")

    if pending:
        log_path = logs / f"cache-{opts.drive}.log"
        t0 = time.monotonic()
        try:
            proc, log = _launch(cache_command(opts), log_path, [])
        except OSError as exc:
            _say("FAILED", f"cache {opts.drive} could not start: {exc}")
            return 1
        with log:
            while proc.poll() is None and not stop:
                time.sleep(POLL_SECONDS)
            code = _stop(proc) if stop else proc.returncode
        if stop:
            _say("matrix", f"stopped by signal {stop[0]} while building the cache; no run started")
            return 1
        if code != 0:
            _say("FAILED", f"cache {opts.drive} exit {code} in {time.monotonic() - t0:.1f}s (log {log_path})")
            _say("matrix", "no run started")
            return 1
        _say("done", f"cache {opts.drive} exit 0 in {time.monotonic() - t0:.1f}s")

    queue = list(pending)
    running: dict[str, tuple] = {}
    while (queue or running) and not stop:
        while queue and len(running) < opts.parallel and not stop:
            job = queue.pop(0)
            state, note = run_state(job, opts)  # again: the disk is the truth at launch time
            if state == "finished":
                _say("skip", f"{job.name} ({note})")
                counts["skipped"] += 1
                continue
            if state == "conflict":
                _say("CONFLICT", f"{job.name} {note}; left untouched, move it aside to retrain it")
                counts["conflicts"] += 1
                continue
            log_path = logs / f"{job.name}.log"
            try:
                proc, log = _launch(train_command(job, opts, overwrite=state == "unfinished"),
                                    log_path, [note] if state == "unfinished" else [])
            except OSError as exc:
                _say("FAILED", f"{job.name} could not start: {exc}")
                counts["failed"] += 1
                continue
            running[job.name] = (proc, log, log_path, time.monotonic())
            _say("start", job.name + (f" ({note})" if state == "unfinished" else ""))
        for name, (proc, log, log_path, t0) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del running[name]
            if code == 0:
                _say("done", f"{name} exit 0 in {time.monotonic() - t0:.1f}s")
                counts["ok"] += 1
            else:
                _say("FAILED", f"{name} exit {code} in {time.monotonic() - t0:.1f}s (log {log_path})")
                counts["failed"] += 1
        if (queue or running) and not stop:
            time.sleep(POLL_SECONDS)

    if stop:
        for proc, _, _, _ in running.values():
            proc.terminate()
        for name, (proc, log, log_path, t0) in running.items():
            code = _stop(proc)
            seconds = time.monotonic() - t0
            if code in (-signal.SIGTERM, -signal.SIGKILL):
                log.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} stopped by the matrix runner\n")
                _say("stopped", f"{name} after {seconds:.1f}s; it starts from scratch next launch")
                counts["stopped"] += 1
            elif code == 0:  # finished just before the stop
                _say("done", f"{name} exit 0 in {seconds:.1f}s")
                counts["ok"] += 1
            else:
                _say("FAILED", f"{name} exit {code} in {seconds:.1f}s (log {log_path})")
                counts["failed"] += 1
            log.close()
        _say("matrix", f"stopped by signal {stop[0]}: {counts['ok']} ok, {counts['failed']} failed, "
                       f"{counts['stopped']} stopped, {len(queue)} not started")
        return 1

    _say("matrix", f"finished in {time.monotonic() - t_start:.0f}s: {counts['ok']} ok, "
                   f"{counts['failed']} failed, {counts['skipped']} skipped, {counts['conflicts']} conflicts")
    return 1 if counts["failed"] or counts["conflicts"] else 0


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


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="train_matrix.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs", help="where the runs, caches and logs go")
    p.add_argument("--data-dir", default="data", help="folder holding the drive folders")
    p.add_argument("--drive", default="drive2018")
    p.add_argument("--parallel", type=_int_from(1), default=3, help="runs trained at the same time")
    p.add_argument("--epochs", type=_int_from(1), default=20)
    p.add_argument("--patience", type=_int_from(0), default=0,
                   help="early-stopping patience passed to every run; 0 trains every epoch")
    p.add_argument("--firewall", type=_int_from(0), default=FIREWALL)
    p.add_argument("--block", type=_int_from(1), default=BLOCK)
    p.add_argument("--device", default="auto", choices=DEVICES)
    p.add_argument("--only", metavar="REGEX", help="train only the runs whose name matches")
    p.add_argument("--dry-run", action="store_true", help="print what would run; change nothing")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    opts = parse_args(argv)
    jobs = matrix_jobs(opts.drive)
    if opts.only is not None:
        try:
            pattern = re.compile(opts.only)
        except re.error as exc:
            print(f"error: --only is not a regular expression: {exc}", file=sys.stderr)
            return 2
        jobs = [job for job in jobs if pattern.search(job.name)]
        if not jobs:
            print(f"error: --only {opts.only!r} matches no run of the matrix", file=sys.stderr)
            return 2

    if opts.dry_run:
        states = [(job, *run_state(job, opts)) for job in jobs]
        print(f"# {len(jobs)} runs; each command runs with PYTHONPATH={ROOT / 'src'}")
        if any(state in ("new", "unfinished") for _, state, _ in states):
            print(shlex.join(cache_command(opts)))
        for job, state, note in states:
            if state in ("new", "unfinished"):
                print(shlex.join(train_command(job, opts, overwrite=state == "unfinished")))
            else:
                print(f"{'skip' if state == 'finished' else 'CONFLICT'} {job.name} ({note})")
        return 1 if any(state == "conflict" for _, state, _ in states) else 0

    logs = Path(opts.runs_dir) / "logs"
    pid_file = logs / "matrix.pid"
    other = _live_runner(pid_file)
    if other is not None:
        print(f"error: a training matrix is already running (pid {other}); "
              f"if it is not, delete {pid_file}", file=sys.stderr)
        return 2
    logs.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{os.getpid()}\n")

    stop: list[int] = []
    handlers = {sig: signal.signal(sig, lambda signum, frame: stop.append(signum))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        return _run(jobs, opts, logs, stop)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        try:
            if pid_file.read_text().strip() == str(os.getpid()):
                pid_file.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
