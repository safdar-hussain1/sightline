"""Figures drawn from the summary.

* ``ladder.png`` -- MAE on each rung of the generalisation ladder per model
  (the test split; for "another day" the other drive from its third frame on,
  as the run reports score it), error bars one seed standard deviation, and on
  every bar a black tick at what always predicting 0 deg scores on that bar's
  own frames (its model's seeds).
* ``paired-leakage.png`` -- the paired measurement: per seed, the MAE of the
  shuffled-split model and of the blocked-split model on identical frames.
* ``regimes.png`` -- MAE on straight, gentle and sharp frames of the blocked
  test split for the shipped model, PilotNet and always predicting 0 deg; the
  regimes' labels state the edges ``sightline.evaluate.REGIME_BINS`` cuts at.

Every plotted value is read from the summary; nothing is recomputed. Sizes
are fixed (``FIGURE_SIZES``, in pixels at ``DPI``) and the PNGs carry no
software tag. Each model keeps one colour in every figure; the zero baseline
is neutral grey or black, never a model colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sightline.evaluate import REGIME_BINS

__all__ = ["DPI", "FIGURE_SIZES", "MODEL_COLOURS", "draw", "write_figures"]

DPI = 144
_INCHES = {"ladder.png": (8.0, 4.5), "paired-leakage.png": (8.0, 4.5), "regimes.png": (7.0, 4.0)}
#: pixel size of every figure
FIGURE_SIZES = {name: (round(w * DPI), round(h * DPI)) for name, (w, h) in _INCHES.items()}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: points left between the plot and its subtitle for the legend row
LEGEND_ROOM_PT = 26

#: one categorical palette, in a fixed order checked for colour-blind separation
MODEL_COLOURS = {"sightlinenet": "#2a78d6", "pilotnet": "#eb6834", "sightlinenet-1f": "#1baf7a"}
MODEL_LABELS = {"sightlinenet": "SightlineNet", "pilotnet": "PilotNet",
                "sightlinenet-1f": "SightlineNet, single frame"}
RUNGS = ("shuffled", "blocked", "later in the drive", "another day")
#: the steering regimes by their key in the reports, each with the label under its bars, which
#: states the regime's edges as the evaluation cuts them
REGIMES = (("straight", f"straight\n|angle| < {REGIME_BINS[0]:g}°"),
           ("gentle", f"gentle\n{REGIME_BINS[0]:g}–{REGIME_BINS[1]:g}°"),
           ("sharp", f"sharp\n≥ {REGIME_BINS[1]:g}°"))
ZERO_LABEL = "always predicting 0°"

_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 9,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "legend.frameon": False,
    "legend.fontsize": 9,
}


def _figure(name: str, title: str, subtitle: str):
    fig = Figure(figsize=_INCHES[name], dpi=DPI, facecolor=SURFACE, layout="constrained")
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_title(subtitle, loc="left", color=INK_SECONDARY, pad=LEGEND_ROOM_PT)
    return fig, ax


def _models(present) -> list[str]:
    unknown = set(present) - set(MODEL_COLOURS)
    if unknown:
        raise ValueError(f"no colour assigned to model(s) {sorted(unknown)}")
    return [m for m in MODEL_COLOURS if m in present]


def _legend(ax, handles: list) -> None:
    """One row of legend between the subtitle and the plot, clear of the data."""
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncols=len(handles),
              borderaxespad=0.4, handlelength=1.6, columnspacing=1.6)


def _value_label(ax, x: float, top: float, value: float) -> None:
    ax.annotate(f"{value:.1f}", (x, top), xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)


def _ladder(summary: dict, name: str) -> Figure:
    ladder = summary["ladder"]
    models = _models(ladder)
    if not models:
        raise ValueError("the summary has no ladder")
    fig, ax = _figure(name, "Steering error on each rung of the generalisation ladder",
                      "MAE, mean over seeds (±1 sd); on the test split, "
                      "or for another day on the other drive from its third frame")
    width, gap = 0.2, 0.03
    offsets = {m: (j - (len(models) - 1) / 2) * (width + gap) for j, m in enumerate(models)}
    for model in models:
        entries = {e["rung"]: e for e in ladder[model]}
        for i, rung in enumerate(RUNGS):
            entry = entries.get(rung)
            if entry is None:
                continue
            x, mean, sd = i + offsets[model], entry["mae_deg"]["mean"], entry["mae_deg"]["sd"]
            ax.bar(x, mean, width, color=MODEL_COLOURS[model], zorder=2)
            if sd is not None:
                ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor=INK_SECONDARY, elinewidth=1,
                            capsize=3, zorder=3)
            # the zero baseline of this bar's own frames: a shuffled rung's test
            # frames depend on the seed, so models with other seeds get other ticks
            half = (width + gap) / 2
            ax.hlines(entry["zero_mae_deg"], x - half, x + half, colors=INK, linewidth=2, zorder=4,
                      gid="zero-baseline")
    ax.set_xticks(range(len(RUNGS)), labels=list(RUNGS))
    ax.set_xlim(-0.6, len(RUNGS) - 0.4)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("mean absolute error (degrees)")
    handles = [Patch(color=MODEL_COLOURS[m], label=MODEL_LABELS[m]) for m in models]
    handles.append(Line2D([0], [0], color=INK, linewidth=2, label=f"{ZERO_LABEL}, same frames"))
    _legend(ax, handles)
    return fig


def _paired(summary: dict, name: str) -> Figure:
    paired = summary["paired_leakage"]
    models = _models(paired)
    if not models:
        raise ValueError("the summary has no paired leakage measurement")
    fig, ax = _figure(name, "Paired leakage: the same held-out frames, two training splits",
                      "MAE on the frames both splits hold out for testing; one line per seed")
    handles = []
    for model in models:
        colour = MODEL_COLOURS[model]
        for p in paired[model]["per_seed"]:
            ys = [p["shuffled_mae_deg"], p["blocked_mae_deg"]]
            ax.plot([0, 1], ys, color=colour, linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter([0, 1], ys, s=64, color=colour, edgecolors=SURFACE, linewidths=2, zorder=3)
        ratio = paired[model]["ratio"]["mean"]
        handles.append(Line2D([0], [0], color=colour, linewidth=2, marker="o", markersize=7,
                              markeredgecolor=SURFACE, markeredgewidth=2,
                              label=f"{MODEL_LABELS[model]}: blocked error {ratio:.2f}× the shuffled one"))
    ax.set_xticks([0, 1], labels=["trained on a shuffled split", "trained on a blocked split"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("MAE on the paired frames (degrees)")
    _legend(ax, handles)
    return fig


def _regimes(summary: dict, name: str) -> Figure:
    regimes = summary["regimes"]
    shipped = summary["shipped"]
    series = [(f"{MODEL_LABELS[shipped['model']]} (shipped, seed {shipped['seed']})",
               MODEL_COLOURS[shipped["model"]], regimes["models"][shipped["model"]]["regimes"])]
    if "pilotnet" in regimes["models"] and shipped["model"] != "pilotnet":
        series.append((MODEL_LABELS["pilotnet"], MODEL_COLOURS["pilotnet"],
                       regimes["models"]["pilotnet"]["regimes"]))
    series.append((ZERO_LABEL, MUTED, regimes["zero"]))
    fig, ax = _figure(name, "Error by steering regime",
                      f"MAE on the {regimes['protocol']} {regimes['split']} split by the human's angle; "
                      "seeds chosen on validation")
    width, gap = 0.22, 0.03
    for j, (label, colour, values) in enumerate(series):
        offset = (j - (len(series) - 1) / 2) * (width + gap)
        for i, (key, _) in enumerate(REGIMES):
            mae = values[key]["mae_deg"]
            ax.bar(i + offset, mae, width, color=colour, zorder=2)
            _value_label(ax, i + offset, mae, mae)
    counts = regimes["zero"]
    ax.set_xticks(range(len(REGIMES)),
                  labels=[f"{text}\n{counts[key]['n']:,} frames" for key, text in REGIMES])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("mean absolute error (degrees)")
    _legend(ax, [Patch(color=colour, label=label) for label, colour, _ in series])
    return fig


_DRAW = {"ladder.png": _ladder, "paired-leakage.png": _paired, "regimes.png": _regimes}


def draw(summary: dict, name: str) -> Figure:
    """The figure ``name`` (a key of ``FIGURE_SIZES``) drawn from ``summary``."""
    if name not in _DRAW:
        raise ValueError(f"unknown figure {name!r}; expected one of {sorted(_DRAW)}")
    with matplotlib.rc_context(_RC):
        return _DRAW[name](summary, name)


def write_figures(summary: dict, out_dir: Path) -> list[Path]:
    """Draw every figure of ``summary`` into ``out_dir``; return the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in _DRAW:
        path = out_dir / name
        fig = draw(summary, name)
        with matplotlib.rc_context(_RC):
            fig.savefig(path, dpi=DPI, facecolor=SURFACE, metadata={"Software": None})
        written.append(path)
    return written
