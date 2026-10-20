"""The narrative notebooks, read as JSON: running them needs Jupyter, and the first one the dataset.

Both were run top to bottom with their outputs saved. Their text cells describe and never type a
number; code cells print every value, from the reports. The values saved in their outputs are the
reports' current ones, so a report that changes without the notebooks being run again fails here.
The claims the results notebook's text makes about models and rungs are checked against the run
reports as well, so a text that no longer holds is found before a reader finds it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from sightline.data.labels import load_labels
from sightline.data.splits import make_split
from sightline.evaluate import TOLERANCES_DEG, within_key

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("01-data-and-leakage.ipynb", "02-results.ipynb")
LABELS = {"pilotnet": "PilotNet", "sightlinenet": "SightlineNet", "sightlinenet-1f": "SightlineNet, single frame"}
RUNGS = {"leaky": "shuffled", "blocked": "blocked", "forward": "later in the drive"}
DRIVE = "drive2018"


def _notebook(name: str) -> dict:
    return json.loads((REPO / "notebooks" / name).read_text())


def _code(name: str) -> list[dict]:
    return [cell for cell in _notebook(name)["cells"] if cell["cell_type"] == "code"]


def _shown(name: str) -> str:
    """Every text the notebook's code cells printed or displayed."""
    parts = []
    for cell in _code(name):
        for output in cell["outputs"]:
            if output["output_type"] == "stream":
                parts.append("".join(output["text"]))
            parts += ["".join(output.get("data", {}).get(kind, "")) for kind in ("text/markdown", "text/plain")]
    return "\n".join(parts)


def _rows(name: str) -> list[list[str]]:
    """Every row of every markdown table the notebook displayed, as its cells."""
    rows = []
    for line in _shown(name).splitlines():
        if line.startswith("|"):
            rows.append([cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))])
    return rows


def _shows(shown: str, value: str) -> bool:
    """Whether ``value`` stands in ``shown`` as a whole token: never as the head or tail of a longer
    number (``0.54`` is not found in ``10.54`` or in ``0.543``, nor ``5.74`` in ``-5.74``)."""
    return re.search(r"(?<![\w.,+-])" + re.escape(value) + r"(?!\w)", shown) is not None


def _has_row(rows: list[list[str]], cells: list[str]) -> bool:
    """Whether some table row holds ``cells`` as consecutive cells, each exactly."""
    n = len(cells)
    return any(row[i:i + n] == cells for row in rows for i in range(len(row) - n + 1))


def _assert_shows(name: str, tokens=(), rows=()) -> None:
    shown, table = _shown(name), _rows(name)
    missing = [t for t in tokens if not _shows(shown, t)] + [list(r) for r in rows if not _has_row(table, list(r))]
    assert not missing, f"{name} shows other values than the reports (run it again): {missing}"


def _report(path: str) -> dict:
    return json.loads((REPO / "reports" / path).read_text())


def _runs() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted((REPO / "reports" / "runs").glob("*.json"))}


def _pm(stat: dict, digits: int = 2) -> str:
    return f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f}"


def _per_seed(stat: dict, digits: int = 2) -> str:
    return ", ".join(f"{value:.{digits}f}" for value in stat["values"])


# --- the notebooks as files -------------------------------------------------------------------

@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_code_cell_ran_once_in_order_without_an_error_or_a_warning(name):
    code = _code(name)
    assert code and [cell["execution_count"] for cell in code] == list(range(1, len(code) + 1))
    for cell in code:
        assert cell["outputs"], "".join(cell["source"])[:80]
        for output in cell["outputs"]:
            assert output["output_type"] != "error" and output.get("name") != "stderr", output


NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SIBLINGS = {path.name for path in (REPO / "notebooks").glob("*.ipynb")}


def typed_digits(text: str, siblings=SIBLINGS) -> list[str]:
    """The digits a text cell types. Prose may carry none. Inline code may carry them only inside a
    code name, which is an identifier (``drive2018``, ``within_1p5_deg``, also in a path or a command
    line) or the file name of a notebook beside this one; ``1.5`` and ``17.5 fps`` are numbers in code
    dress and count."""
    found = []
    for piece in re.split(r"(`[^`]*`)", text):
        if len(piece) >= 2 and piece[0] == piece[-1] == "`":
            code = piece[1:-1]
            if code in siblings:
                continue
            piece = NAME.sub("", code)
        found += re.findall(r"\d+", piece)
    return found


@pytest.mark.parametrize("text, digits", [
    ("a `drive2018`, `data/drive2017` and `sightline cache --drive drive2018`", []),
    ("the field `within_1p5_deg` and the folder `reports/runs/`", []),
    ("the other notebook, `01-data-and-leakage.ipynb`", []),
    ("a number typed in prose, 3.17", ["3", "17"]),
    ("the same, in code dress: `3.17`", ["3", "17"]),
    ("a rate: `17.5 fps`", ["17", "5"]),
    ("a tolerance: `1.5`", ["1", "5"]),
    ("a name that starts with digits: `2x`", ["2"]),
    ("a notebook that is not there: `99-missing.ipynb`", ["99"]),
    ("a lone tick ` and a 4", ["4"]),
])
def test_the_rule_for_numbers_in_text_cells(text, digits):
    assert typed_digits(text) == digits


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_text_cells_type_no_numbers(name):
    for cell in _notebook(name)["cells"]:
        if cell["cell_type"] == "markdown":
            text = "".join(cell["source"])
            assert "```" not in text, "markdown cells describe; code goes in code cells"
            assert not typed_digits(text), text


def test_the_matching_of_shown_values_is_exact_to_the_token():
    shown = "| a | 10.54 | 0.543 | -5.74 | 4.01 ± 0.21 |\nfrom 6,000 to 16,000"
    assert _shows(shown, "10.54") and _shows(shown, "-5.74") and _shows(shown, "4.01 ± 0.21") and _shows(shown, "6,000")
    assert not _shows(shown, "0.54") and not _shows(shown, "5.74") and not _shows(shown, "4.01 ± 0.2")
    assert not _shows(shown, "0,000") and not _shows(shown, "0.5")
    rows = [["x", "1", "2", "3"], ["y", "2", "3", "4"]]
    assert _has_row(rows, ["1", "2"]) and _has_row(rows, ["x", "1", "2", "3"]) and not _has_row(rows, ["1", "3"])


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_notebooks_name_the_standard_kernel_and_record_no_timings(name):
    notebook = _notebook(name)
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert all(not cell.get("metadata") for cell in notebook["cells"])


def test_the_results_notebook_reads_the_reports_alone():
    """No dataset, cache, run folder or weights: the reports, and the training module's source to check the
    learning-rate schedule the recipe line names."""
    code = "\n".join("".join(cell["source"]) for cell in _code("02-results.ipynb"))
    assert not re.search(r"(?<![\w/])(data|runs)/", code)  # reports/runs is a reports folder
    assert not re.search(r"\b(Workspace|load_labels|open_cache|torch)\b", code)


# --- notebook 02: what it shows is what the reports hold -----------------------------------------

def test_the_results_notebook_shows_the_recipe_the_reports_record():
    summary, runs = _report("summary.json"), _runs()
    cfg = summary["train_config"]
    tokens = [f"{len(runs)} run reports: trained on {DRIVE}, another day is drive2017",
              f"recipe: {cfg['epochs']} epochs, batch {cfg['batch_size']}, learning rate {cfg['lr']:g} at the start, "
              f"on the cosine schedule of sightline.train, weight decay {cfg['weight_decay']:g}, no early stopping"]
    assert cfg["patience"] == 0
    for model, recipe in cfg["models"].items():
        first = runs[f"{DRIVE}_{model}_blocked_s0"]
        tokens.append(f"{LABELS[model]}: " + ("frames t and t-2" if first["frame_lags"] == [2] else "frame t alone")
                      + f", {first['n_params']:,} parameters, {recipe['augment']} augmentation, dropout {recipe['dropout']:g}")
    _assert_shows("02-results.ipynb", tokens=tokens)


def test_the_results_notebook_shows_where_and_with_what_the_runs_were_trained_and_evaluated():
    summary, runs = _report("summary.json"), _runs()
    devices: dict = {}
    for r in runs.values():  # in the order of the run reports' names, as the notebook counts them
        devices[r["train_device"]] = devices.get(r["train_device"], 0) + 1
    assert {r["code_version"] for r in runs.values()} == set(summary["code_versions"]["train"])
    assert {r["eval_code_version"] for r in runs.values()} == set(summary["code_versions"]["eval"])
    tokens = [f"trained at commit {', '.join(c[:7] for c in summary['code_versions']['train'])} on "
              + ", ".join(f"{device} ({count} runs)" for device, count in devices.items())
              + f"; evaluated at commit {', '.join(c[:7] for c in summary['code_versions']['eval'])} on the "
              f"{summary['device']}",
              "libraries: " + ", ".join(f"{name} {version}" for name, version in sorted(summary["versions"].items()))]
    _assert_shows("02-results.ipynb", tokens=tokens)


def test_the_results_notebook_shows_that_each_paired_comparison_is_on_one_set_of_frames():
    summary, leakage, runs = _report("summary.json"), _report("leakage.json"), _runs()
    compared = [(runs[f"{DRIVE}_{model}_leaky_s{p['seed']}"]["paired"]["frames_sha256"],
                 runs[f"{DRIVE}_{model}_blocked_s{p['seed']}"]["paired"]["frames_sha256"])
                for model, entry in summary["paired_leakage"].items() for p in entry["per_seed"]]
    # what the notebook prints as True holds in the reports, and is not true of nothing
    assert compared and all(shuffled == blocked for shuffled, blocked in compared)
    first = {runs[f"{DRIVE}_{model}_{protocol}_s{leakage['seed']}"]["paired"]["frames_sha256"]
             for model in summary["paired_leakage"] for protocol in ("leaky", "blocked")}
    assert first == {leakage["paired"]["frames_sha256"]}
    _assert_shows("02-results.ipynb", tokens=[
        f"each seed's two models were scored on the same frames: True ({len(compared)} pairs of run reports)",
        f"seed {leakage['seed']}'s paired frames are the ones reports/leakage.json analyses: True"])


def test_the_results_notebook_shows_the_splits_and_their_gaps():
    summary, leakage, runs = _report("summary.json"), _report("leakage.json"), _runs()
    rows, medians = [], []
    for protocol, rung in RUNGS.items():
        sizes = runs[f"{DRIVE}_sightlinenet_{protocol}_s{leakage['seed']}"]["splits"]
        gap = leakage["gaps"][protocol]["test"]
        rows.append([rung, f"{sizes['train_n']:,}", f"{sizes['val_n']:,}", f"{sizes['test_n']:,}", f"{gap['min_frames']:,}",
                     f"{gap['median_frames']:,g}", f"{gap['frac_within_1']:.1%}"])
        medians.append(f"{gap['median_frames']:,g}")
    blocked = runs[f"{DRIVE}_sightlinenet_blocked_s{leakage['seed']}"]["splits"]
    need = 2 * blocked["firewall"] + 1
    other = summary["drives"]["drive2017"]["frames"]
    (scored,) = {r["cross_drive"]["n"] for r in runs.values()}  # every run scores the same frames of the other drive
    counts = ", ".join(f"{rung} in {sum(r['protocol'] == protocol for r in runs.values())} runs" for protocol, rung in RUNGS.items())
    tokens = [
        "median distance to the nearest training frame, from the shuffled split to the later-in-the-drive split: "
        + ", ".join(medians) + " frames; another day is a different drive",
        f"blocked: blocks of {blocked['block']:,} frames; {blocked['firewall']} frames dropped on each side of every "
        f"training/evaluation boundary, so every evaluation frame is at least {need} frames from a training frame; "
        "01-data-and-leakage.ipynb measures how far that is in seconds, from the drive's timestamps",
        f"another day: {scored:,} of the {other:,} frames of drive2017, all but the first {other - scored}",
        f"the shuffled split above is seed {leakage['seed']}'s; every run of a split deals the same numbers of frames: {counts}",
    ]
    _assert_shows("02-results.ipynb", tokens=tokens, rows=rows)
    assert float(medians[0].replace(",", "")) < float(medians[1].replace(",", "")) < float(medians[2].replace(",", ""))


def test_the_results_notebook_leaves_the_gap_in_seconds_to_the_data_notebook():
    """The reports hold the drive's median frame interval, not the time between two frames, and the
    interval varies along the drive: a distance in frames times the median interval is not how far
    apart two frames are. The results notebook, which reads the reports alone, gives the distance in
    frames; the data notebook, which has the timestamps, gives it in seconds."""
    code = "\n".join("".join(cell["source"]) for cell in _code("02-results.ipynb"))
    assert "median_dt_s" not in code
    assert not re.search(r"\d s at the median frame interval", _shown("02-results.ipynb"))


def test_the_results_notebook_shows_the_shipped_choice():
    summary, runs = _report("summary.json"), _runs()
    shipped, group = summary["shipped"], summary["groups"][f"{summary['shipped']['model']}/{summary['shipped']['protocol']}"]
    rows = []
    for seed, run in zip(group["seeds"], group["runs"]):
        r = runs[run]
        rows.append([str(seed), f"{r['val']['model']['mae_deg']:.3f}", f"{r['test']['model']['mae_deg']:.3f}",
                     f"{r['cross_drive']['model']['mae_deg']:.3f}", "shipped" if run == shipped["run"] else ""])
    best_on_test = group["seeds"][min(range(len(group["seeds"])), key=lambda i: group["test_mae_deg"]["values"][i])]
    tokens = [f"shipped: {shipped['run']}, the lowest {shipped['selected_by']}; its shuffled-split twin (same model and "
              f"seed): {summary['twin']['run']}",
              f"the lowest test MAE belongs to seed {best_on_test}, not the shipped one; test scores never choose"]
    assert best_on_test != shipped["seed"]
    _assert_shows("02-results.ipynb", tokens=tokens, rows=rows)


def test_the_results_notebook_shows_the_ladder_the_paired_measurement_and_the_groups():
    summary, runs = _report("summary.json"), _runs()
    rows, tokens = [], []
    for model, ladder in summary["ladder"].items():
        skills = []
        for entry in ladder:
            group = summary["groups"][f"{model}/{entry['protocol']}"]
            split, skill = ("test", group["test_skill"]) if entry["split"] == "test" else ("cross_drive", group["cross_skill"])
            wins = sum(runs[run][split]["model"]["mae_deg"] < runs[run][split]["baselines"]["zero"]["mae_deg"]
                       for run in group["runs"])
            rows.append([LABELS[model], entry["rung"], _pm(entry["mae_deg"]), _per_seed(entry["mae_deg"]),
                         f"{entry['zero_mae_deg']:.2f}", _pm(skill, 3), f"{wins} of {len(entry['seeds'])}"])
            skills.append(f"{entry['rung']} {skill['mean']:.3f}")
        tokens.append(f"{LABELS[model]}: skill on each rung, " + ", ".join(skills))
        for entry in (e for e in ladder if e["protocol"] == "leaky"):
            zeros = [runs[run]["test"]["baselines"]["zero"]["mae_deg"] for run in summary["groups"][f"{model}/leaky"]["runs"]]
            assert len(zeros) == len(entry["seeds"]) > 1  # a mean over the seeds' own test sets
            tokens.append(f"{LABELS[model]}, shuffled rung: always straight ahead on each seed's own test frames "
                          + ", ".join(f"{z:.2f}" for z in zeros) + f", mean {entry['zero_mae_deg']:.2f}")
    for model, entry in summary["paired_leakage"].items():
        tokens.append(f"{LABELS[model]}: on identical frames the blocked-split error is {_pm(entry['ratio'])} times the "
                      f"shuffled-split error (per seed: {_per_seed(entry['ratio'])}); "
                      f"MAE {_pm(entry['shuffled_mae_deg'])} vs {_pm(entry['blocked_mae_deg'])} deg")
        for seed in entry["per_seed"]:
            rows.append([LABELS[model], str(seed["seed"]), f"{seed['n']:,}", f"{seed['shuffled_mae_deg']:.3f}",
                         f"{seed['blocked_mae_deg']:.3f}", f"{seed['ratio']:.3f}", f"{seed['difference_deg']:.2f}",
                         f"{seed['zero_mae_deg']:.2f}"])
    for key, group in summary["groups"].items():
        model, protocol = key.split("/")
        promise = group["cross_mae_deg"]["mean"] / group["test_mae_deg"]["mean"]
        rows.append([LABELS[model], RUNGS[protocol], _pm(group["test_mae_deg"]), _pm(group["cross_mae_deg"]),
                     f"{promise:.2f}", _pm(group["cross_skill"], 3)])
    _assert_shows("02-results.ipynb", tokens=tokens, rows=rows)


def test_the_results_notebook_shows_each_split_trained_model_on_another_day():
    summary = _report("summary.json")
    straight = summary["groups"]["sightlinenet/blocked"]["cross_zero_mae_deg"]
    rows, tokens = [], []
    for model in ("sightlinenet", "pilotnet"):
        shuffled, blocked = summary["groups"][f"{model}/leaky"], summary["groups"][f"{model}/blocked"]
        pairs = list(zip(shuffled["cross_mae_deg"]["values"], blocked["cross_mae_deg"]["values"]))
        for seed, (s, b) in zip(blocked["seeds"], pairs):
            rows.append([LABELS[model], str(seed), f"{s:.2f}", f"{b:.2f}", f"{s - b:+.2f}", f"{straight:.2f}"])
        tokens.append(f"{LABELS[model]}: the model trained on shuffled frames is worse on another day in "
                      f"{sum(s > b for s, b in pairs)} of {len(pairs)} seeds; above always straight ahead in "
                      f"{sum(s > straight for s, _ in pairs)} of {len(pairs)} seeds when trained on shuffled frames, "
                      f"{sum(b > straight for _, b in pairs)} of {len(pairs)} when trained on blocked frames")
    tokens.append(f"always straight ahead on another day: {straight:.2f} deg")
    _assert_shows("02-results.ipynb", tokens=tokens, rows=rows)


def test_the_claims_in_the_results_text_hold_in_the_run_reports():
    """What the results notebook's text says about rungs and models, checked on each run's own report."""
    runs = _runs()

    def scores(model, protocol, split):
        found = [(r[split]["model"]["mae_deg"], r[split]["baselines"]["zero"]["mae_deg"])
                 for r in runs.values() if r["model"] == model and r["protocol"] == protocol]
        assert found, f"no {model} run on the {protocol} split"  # every seed of nothing is not every seed
        return found

    def ahead_in_every_seed(model, protocol, split="test"):
        return all(mae < zero for mae, zero in scores(model, protocol, split))

    def mean_ahead(model, protocol, split="test"):
        pairs = scores(model, protocol, split)
        return sum(mae for mae, _ in pairs) / len(pairs) < sum(zero for _, zero in pairs) / len(pairs)

    honest = (("blocked", "test"), ("forward", "test"), ("blocked", "cross_drive"))
    # SightlineNet beats straight ahead on every honest rung, in every seed
    assert all(ahead_in_every_seed("sightlinenet", protocol, split) for protocol, split in honest)
    # PilotNet's mean error is below it on the blocked split only, of the honest rungs
    assert [(p, s) for p, s in honest if mean_ahead("pilotnet", p, s)] == [("blocked", "test")]
    # both models beat it in every seed on the shuffled split, which flatters them
    assert ahead_in_every_seed("sightlinenet", "leaky") and ahead_in_every_seed("pilotnet", "leaky")
    # the model a shuffled split trains: SightlineNet's is no worse on another day in any seed, PilotNet's worse in every
    # seed, and neither of PilotNet's two beats straight ahead there
    seeds = sorted({r["seed"] for r in runs.values()})
    assert len(seeds) > 1
    for seed in seeds:
        sl = [runs[f"{DRIVE}_sightlinenet_{protocol}_s{seed}"]["cross_drive"]["model"]["mae_deg"] for protocol in ("leaky", "blocked")]
        pn = [runs[f"{DRIVE}_pilotnet_{protocol}_s{seed}"]["cross_drive"]["model"]["mae_deg"] for protocol in ("leaky", "blocked")]
        zero = runs[f"{DRIVE}_pilotnet_blocked_s{seed}"]["cross_drive"]["baselines"]["zero"]["mae_deg"]
        assert sl[0] <= sl[1]
        assert pn[0] > pn[1] and min(pn) > zero


def test_the_results_notebook_shows_the_baselines_and_the_regimes():
    summary, runs = _report("summary.json"), _runs()
    run = runs[summary["shipped"]["run"]]
    rows = []
    for split, title in (("val", "validation"), ("test", "test"), ("cross_drive", "another day")):
        s, b = run[split], run[split]["baselines"]
        rows.append([title, f"{s['n']:,}", f"{s['model']['mae_deg']:.2f}", f"{b['zero']['mae_deg']:.2f}",
                     f"{b['mean']['mae_deg']:.2f}", f"{b['zoh']['mae_deg']:.2f}", f"{b['extrap']['mae_deg']:.2f}",
                     f"{s['pearson_r']:.3f}", f"{s['sign_agreement']:.1%}"])
    t, smooth, cutoffs = run["test"]["model"], run["test"]["smoothness"], summary["cutoffs"]
    tolerances = cutoffs["tolerances_deg"]
    tokens = [f"sign agreement: the share of frames steered at least {cutoffs['sign_min_abs_deg']:g} deg either way on "
              "which the model turns the wheel the same way as the human",
              "test frames within " + " / ".join(f"{x:g}" for x in tolerances) + " deg of the human: "
              + " / ".join(f"{t[within_key(x)]:.1%}" for x in tolerances)
              + f"; median error {t['medae_deg']:.2f} deg, RMSE {t['rmse_deg']:.2f} deg",
              f"frame-to-frame change on the test split: model {smooth['model']['mean_abs_step_deg']:.2f} deg, "
              f"human {smooth['human']['mean_abs_step_deg']:.2f} deg on average"]
    regimes, (low, high) = summary["regimes"], cutoffs["regime_bins_deg"]
    columns = [entry["regimes"] for entry in regimes["models"].values()] + [regimes["zero"]]
    names = {"straight": f"straight, under {low:g} deg either way", "gentle": f"gentle, {low:g} to {high:g} deg",
             "sharp": f"sharp, {high:g} deg or more"}
    rows.append(["regime", "frames"] + [f"{LABELS[model]} (seed {entry['seed']})" for model, entry in regimes["models"].items()]
                + ["always straight ahead"])
    for key, name in names.items():
        rows.append([name, f"{regimes['zero'][key]['n']:,}"] + [f"{values[key]['mae_deg']:.2f}" for values in columns])
    rows.append(["balanced (the three weighted equally)", ""] + [f"{values['balanced_mae_deg']:.2f}" for values in columns])
    _assert_shows("02-results.ipynb", tokens=tokens, rows=rows)


def test_the_results_notebook_shows_the_ablation_the_standard_recipe_and_the_export():
    summary, legacy, export = _report("summary.json"), _report("legacy.json"), _report("export.json")
    two, one = summary["groups"]["sightlinenet/blocked"], summary["groups"]["sightlinenet-1f/blocked"]
    rows = [["trained on the blocked split", "SightlineNet (two frames)", "SightlineNet, single frame"]]
    for title, key, digits in (("validation MAE", "val_mae_deg", 2), ("test MAE", "test_mae_deg", 2),
                               ("test Pearson r", "test_r", 3), ("test skill", "test_skill", 3),
                               ("another-day MAE", "cross_mae_deg", 2), ("another-day Pearson r", "cross_r", 3),
                               ("another-day skill", "cross_skill", 3)):
        rows.append([title, _pm(two[key], digits), _pm(one[key], digits)])
    tokens = [f"another-day skill over always straight ahead: two frames {_pm(two['cross_skill'], 3)}, "
              f"one frame {_pm(one['cross_skill'], 3)} (per seed: {_per_seed(one['cross_skill'], 3)})"]
    parity = legacy["port_parity"]
    tokens += [legacy["recipe"],
               f"trained on {legacy['trained_on']}; the PyTorch port is within {parity['max_abs_diff_deg']:.1e} deg of the "
               f"TF1 graph on {parity['frames']} frames; scored on the {legacy['device']}"]
    for score in legacy["eval"].values():
        rows.append([f"{score['model']['n']:,}", f"{score['model']['mae_deg']:.2f}", f"{score['zero_mae_deg']:.2f}",
                     f"{score['pearson_r']:.3f}", f"{score['regimes']['balanced_mae_deg']:.2f}"])
    blocked = legacy["eval"][f"{DRIVE}_blocked_test"]["model"]["mae_deg"]
    pilot, sightline = summary["groups"]["pilotnet/blocked"], summary["groups"]["sightlinenet/blocked"]
    cap = legacy["atan_cap"]
    tokens += [f"standard recipe {blocked:.2f} deg, PilotNet {_pm(pilot['test_mae_deg'])} deg, "
               f"SightlineNet {_pm(sightline['test_mae_deg'])} deg",
               f"largest |prediction|: {cap['max_abs_pred_deg']:.1f} deg; labels steered past 180 deg, out of the head's "
               "reach: " + ", ".join(f"{drive} {c['n']} ({c['frac']:.2%})" for drive, c in sorted(cap["labels_beyond_180"].items()))]
    for key, model in export["models"].items():
        p = model["parity"]
        rows.append([key, model["run"], f"{model['bytes']:,}", f"{p['frames']:,}", f"{p['max_angle_diff_deg']:.1e}",
                     f"{p['angle_tolerance_deg']:g}", f"{p['max_saliency_diff']:.1e}", f"{p['saliency_tolerance']:g}"])
    tokens.append("the browser's models are the shipped run and its twin")
    _assert_shows("02-results.ipynb", tokens=tokens, rows=rows)


# --- notebook 01 -------------------------------------------------------------------------------

def test_the_data_notebook_gives_the_memoriser_one_column_per_tolerance_of_the_evaluation_code():
    leakage = _report("leakage.json")
    rows = [["training frames searched", "frames", "MAE (deg)", "RMSE", "median error",
             *(f"within {x:g} deg" for x in TOLERANCES_DEG)]]
    for protocol, m in leakage["memoriser"].items():
        rows.append([RUNGS[protocol], str(m["n"]), f"{m['mae_deg']:.3f}", f"{m['rmse_deg']:.2f}", f"{m['medae_deg']:.2f}",
                     *(f"{m[within_key(x)]:.1%}" for x in TOLERANCES_DEG)])
    _assert_shows("01-data-and-leakage.ipynb", rows=rows)


#: what the data notebook prints of the blocked split's closest validation and test frames
CLOSEST = re.compile(r"blocked (validation|test) frames: the closest to a training frame by the drive's timestamps is "
                     r"frame ([\d,]+), (\d+\.\d) s from training frame ([\d,]+) \(([\d,]+) frames away\)")


def _closest_in_seconds(stamps: np.ndarray, held: np.ndarray, trained: np.ndarray) -> tuple[float, int, int]:
    """The time from the evaluation frame of ``held`` nearest a training frame of ``trained`` to that
    training frame, in seconds by ``stamps``, with the two frames: checked against every training frame,
    not only its neighbours."""
    best = (np.inf, -1, -1)
    for frame in held:
        apart = np.abs(stamps[trained] - stamps[frame])
        k = int(np.argmin(apart))
        if apart[k] < best[0]:
            best = (float(apart[k]), int(frame), int(trained[k]))
    return best


def test_the_data_notebook_gives_the_closest_blocked_evaluation_frames_in_seconds_from_the_timestamps():
    leakage, summary = _report("leakage.json"), _report("summary.json")
    split = make_split("blocked", summary["drives"][DRIVE]["frames"], firewall=leakage["splits"]["firewall"],
                       block=leakage["splits"]["block"])
    found = {m[1]: m for m in CLOSEST.finditer(_shown("01-data-and-leakage.ipynb"))}
    assert sorted(found) == ["test", "validation"], "the closest validation and test frames, in seconds"
    labels = REPO / "data" / DRIVE / "data.txt"
    stamps = load_labels(labels.parent).timestamps_s if labels.is_file() else None
    for which, title in (("val", "validation"), ("test", "test")):
        line = found[title]
        frame, trained, apart = (int(line[k].replace(",", "")) for k in (2, 4, 5))
        seconds = float(line[3])
        # an evaluation frame and the training frame next to it, no closer than the firewall allows
        assert frame in getattr(split, which) and trained in split.train and apart == abs(frame - trained)
        assert not np.isin(np.arange(min(frame, trained) + 1, max(frame, trained)), split.train).any()
        assert apart >= leakage["gaps"]["blocked"][which]["min_frames"] and seconds > 0
        if stamps is not None:  # the dataset is here: the time is measured again from its timestamps
            measured, closest, nearest = _closest_in_seconds(stamps, getattr(split, which), np.sort(split.train))
            assert (f"{measured:.1f}", closest, nearest) == (line[3], frame, trained), title


def test_the_data_notebook_prints_the_closest_blocked_frames_the_site_data_gives():
    """The page reads the blocked split's closest validation and test frames from results.json's timing;
    the data notebook prints them from the same timestamps. Both name the same two frames, and the same
    time at the precision the notebook prints it to."""
    results = json.loads((REPO / "web" / "public" / "data" / "results.json").read_text())
    closest = results["timing"][DRIVE]["closest_eval_s"]["blocked"]
    found = {m[1]: m for m in CLOSEST.finditer(_shown("01-data-and-leakage.ipynb"))}
    assert sorted(found) == ["test", "validation"], "the closest validation and test frames, in seconds"
    for which, title in (("val", "validation"), ("test", "test")):
        line, entry = found[title], closest[which]
        frame, trained, apart = (int(line[k].replace(",", "")) for k in (2, 4, 5))
        assert (frame, trained, apart) == \
            (entry["eval_frame"], entry["train_frame"], abs(entry["eval_frame"] - entry["train_frame"])), title
        decimals = len(line[3].split(".")[1])
        assert f"{entry['gap_s']:.{decimals}f}" == line[3], title


def test_the_data_notebook_shows_the_reports_current_values():
    leakage, summary = _report("leakage.json"), _report("summary.json")
    expected = [f"paired frames: {leakage['paired']['n']}"]
    expected += [f"{memo['mae_deg']:.3f}" for memo in leakage["memoriser"].values()]
    expected += [f"{gaps['test']['median_frames']:,g}" for gaps in leakage["gaps"].values()]
    expected += [f"{pair['frame']:,}" for pair in leakage["pairs"]]
    expected += [f"{drive['frames']:,}" for drive in summary["drives"].values()]
    expected += [f"{name} | {drive['repaired_labels']} |" for name, drive in summary["drives"].items()]
    _assert_shows("01-data-and-leakage.ipynb", tokens=expected)
