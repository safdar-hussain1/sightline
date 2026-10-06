"""The narrative notebooks, read as JSON: running them needs Jupyter, and the first one the dataset.

Both were run top to bottom with their outputs saved. Their text cells describe and never type a
number; code cells print every value, from the reports. The values saved in their outputs are the
reports' current ones, so a report that changes without the notebooks being run again fails here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("01-data-and-leakage.ipynb", "02-results.ipynb")


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


def _report(path: str) -> dict:
    return json.loads((REPO / "reports" / path).read_text())


def _pm(stat: dict, digits: int = 2) -> str:
    return f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_code_cell_ran_once_in_order_without_an_error_or_a_warning(name):
    code = _code(name)
    assert code and [cell["execution_count"] for cell in code] == list(range(1, len(code) + 1))
    for cell in code:
        assert cell["outputs"], "".join(cell["source"])[:80]
        for output in cell["outputs"]:
            assert output["output_type"] != "error" and output.get("name") != "stderr", output


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_text_cells_type_no_numbers(name):
    for cell in _notebook(name)["cells"]:
        if cell["cell_type"] == "markdown":
            text = re.sub(r"`[^`]*`", "", "".join(cell["source"]))  # inline code: file and drive names
            assert not re.search(r"\d", text), text


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_notebooks_name_the_standard_kernel_and_record_no_timings(name):
    notebook = _notebook(name)
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert all(not cell.get("metadata") for cell in notebook["cells"])


def test_the_results_notebook_reads_the_reports_alone():
    code = "\n".join("".join(cell["source"]) for cell in _code("02-results.ipynb"))
    assert not re.search(r"(?<![\w/])(data|runs)/", code)  # reports/runs is a reports folder
    assert not re.search(r"\b(Workspace|load_labels|open_cache|torch)\b", code)


def test_the_results_notebook_shows_the_reports_current_values():
    shown = _shown("02-results.ipynb")
    summary, legacy = _report("summary.json"), _report("legacy.json")
    expected = [summary["shipped"]["run"], summary["twin"]["run"]]
    for ladder in summary["ladder"].values():
        expected += [_pm(entry["mae_deg"]) for entry in ladder]
    for entry in summary["paired_leakage"].values():
        expected.append(_pm(entry["ratio"]))
        for seed in entry["per_seed"]:
            expected += [f"{seed['shuffled_mae_deg']:.3f}", f"{seed['blocked_mae_deg']:.3f}", f"{seed['ratio']:.3f}"]
    for group in summary["groups"].values():
        expected += [_pm(group["test_mae_deg"]), _pm(group["cross_mae_deg"]), _pm(group["cross_skill"], 3)]
    expected += [f"{score['model']['mae_deg']:.2f}" for score in legacy["eval"].values()]
    missing = [value for value in expected if value not in shown]
    assert not missing, f"02-results.ipynb shows other values than the reports (run it again): {missing}"


def test_the_data_notebook_shows_the_reports_current_values():
    shown = _shown("01-data-and-leakage.ipynb")
    leakage, summary = _report("leakage.json"), _report("summary.json")
    expected = [f"paired frames: {leakage['paired']['n']}"]
    expected += [f"{memo['mae_deg']:.3f}" for memo in leakage["memoriser"].values()]
    expected += [f"{gaps['test']['median_frames']:,g}" for gaps in leakage["gaps"].values()]
    expected += [f"{pair['frame']:,}" for pair in leakage["pairs"]]
    expected += [f"{drive['frames']:,}" for drive in summary["drives"].values()]
    expected += [f"{name} | {drive['repaired_labels']} |" for name, drive in summary["drives"].items()]
    missing = [value for value in expected if value not in shown]
    assert not missing, f"01-data-and-leakage.ipynb shows other values than the reports (run it again): {missing}"
