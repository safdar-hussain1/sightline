"""The standard-recipe PilotNet: its PyTorch port against an independent NumPy implementation
of the TF1 graph, the recipe's own preprocessing, the atan cap and the evaluation.

Seeded random weights of the TF1 tensor shapes check the port without any private file; the
same check with the real weights runs when SIGHTLINE_LEGACY_WEIGHTS names their .npz. The
evaluation runs on two synthetic drives named like the real ones, with small blocks and
firewalls.
"""
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from conftest import make_drive
from sightline.cli import main
from sightline.data.labels import load_labels, repair_dropouts
from sightline.data.preprocess import preprocess_frame
from sightline.data.splits import make_split
from sightline.data.verify import label_sha256
from sightline.evaluate import pearson_r, regime_breakdown, regression_metrics
from sightline.legacy import (
    PARITY_FRAMES_PER_SET,
    PARITY_LIMIT_DEG,
    RECIPE,
    TENSOR_SHAPES,
    WEIGHTS_ENV,
    conv2d_valid,
    load_weights,
    preprocess_legacy,
    reference_forward,
    to_model_input,
)
from sightline.models.pilotnet import PilotNet
from sightline.reporting import CROSS_FIRST_FRAME

#: the port must agree with the TF1 graph to this many degrees on every frame
TOL_DEG = 1e-3


def no_cache(*args, **kwargs):
    raise AssertionError("the standard-recipe PilotNet is scored on decoded JPEGs, never on a drive cache")


def random_weights(seed: int = 0) -> dict[str, np.ndarray]:
    """Seeded float32 tensors of the TF1 shapes: He-scaled kernels keep every layer's
    activations in range, and non-zero biases make the port's bias handling count."""
    rng = np.random.default_rng(seed)
    weights = {}
    for name, shape in TENSOR_SHAPES.items():
        if len(shape) == 1:
            weights[name] = rng.normal(0.0, 0.1, shape).astype(np.float32)
        else:
            fan_in = int(np.prod(shape[:-1]))
            weights[name] = rng.normal(0.0, np.sqrt(2.0 / fan_in), shape).astype(np.float32)
    return weights


def random_frames(n: int = 8, seed: int = 1) -> np.ndarray:
    """``n`` decoded-frame-sized (256 x 455) BGR uint8 noise images, each with its own
    brightness per channel, so frames differ and the channel order matters."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (n, 256, 455, 3)).astype(np.float64)
    return (noise * rng.uniform(0.2, 1.0, (n, 1, 1, 3))).astype(np.uint8)


def recipe_input(path) -> np.ndarray:
    """The recipe's preprocessing, verbatim."""
    return cv2.resize(cv2.imread(str(path))[-150:], (200, 66)) / 255.0


def port_from(path) -> PilotNet:
    model = PilotNet().eval()
    model.load_legacy_npz(path)
    return model


def port_predict(model: PilotNet, frames: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(np.stack([to_model_input(f) for f in frames]))
        return model(x)[:, 0].double().numpy()


def reference_predict(weights: dict, frames: np.ndarray) -> np.ndarray:
    return reference_forward(weights, np.stack([preprocess_legacy(f) for f in frames]))


def error_line(err: str) -> str:
    """The CLI's one error line."""
    assert "Traceback" not in err
    lines = [line for line in err.splitlines() if line.startswith("error: ")]
    assert len(lines) == 1, err
    return lines[0]


# --- the port against the TF1 graph -------------------------------------------------------

def test_the_port_matches_the_tf1_graph_with_random_weights(tmp_path):
    weights = random_weights()
    np.savez(tmp_path / "weights.npz", **weights)
    frames = random_frames()
    ref = reference_predict(weights, frames)
    port = port_predict(port_from(tmp_path / "weights.npz"), frames)
    # the outputs sit on the steep part of 2*atan and differ from frame to frame, so a
    # wrong channel order, kernel layout or flatten order cannot hide behind the cap
    assert np.abs(ref).max() < 150.0 and np.ptp(ref) > 5.0
    np.testing.assert_allclose(port, ref, rtol=0, atol=TOL_DEG)


#: the real weights' .npz when SIGHTLINE_LEGACY_WEIGHTS names it, resolved from the folder
#: pytest starts in, before any test moves into its tmp folder
REAL_WEIGHTS = Path(os.environ[WEIGHTS_ENV]).resolve() if os.environ.get(WEIGHTS_ENV) else None


@pytest.mark.skipif(
    REAL_WEIGHTS is None,
    reason=f"{WEIGHTS_ENV} is not set: the standard-recipe PilotNet weights are not part of "
           "the repository; set it to their .npz to check the port with them",
)
def test_the_port_matches_the_tf1_graph_with_the_real_weights():
    frames = random_frames()
    ref = reference_predict(load_weights(REAL_WEIGHTS), frames)
    port = port_predict(port_from(REAL_WEIGHTS), frames)
    np.testing.assert_allclose(port, ref, rtol=0, atol=TOL_DEG)


@pytest.mark.parametrize("bias", [1e6, -1e6])
def test_no_input_steers_the_tf1_graph_past_half_a_turn(bias):
    weights = random_weights()
    weights["Variable_19"] = np.array([bias], np.float32)  # the head's bias: z near +-1e6
    extreme = np.stack([np.zeros((256, 455, 3), np.uint8), np.full((256, 455, 3), 255, np.uint8),
                        *random_frames(n=4)])
    out = reference_predict(weights, extreme)
    assert np.all(np.abs(out) < 180.0)
    assert np.all(np.abs(out) > 179.9) and np.all(np.sign(out) == np.sign(bias))


@pytest.mark.parametrize("bias", [1.0, -1.0, 0.25, 0.0])
def test_the_head_is_twice_atan_in_radians_reported_in_degrees(bias):
    weights = {name: np.zeros(shape, np.float32) for name, shape in TENSOR_SHAPES.items()}
    weights["Variable_19"] = np.array([bias], np.float32)
    out = reference_forward(weights, np.zeros((3, 66, 200, 3), np.float32))
    z = float(np.float32(bias))  # the head's output: its bias, as stored
    np.testing.assert_allclose(out, np.full(3, 2.0 * math.atan(z) * 180.0 / math.pi), rtol=1e-12, atol=1e-12)


def test_the_tf1_graph_flattens_row_then_column_then_channel(tmp_path):
    # conv5's bias alone sets its 1x18x64 output: channel c holds c + 1 everywhere. The first
    # fully-connected unit reads flat element 1 and every later layer passes it on, so the
    # head's z is that element: 2 (row 0, column 0, channel 1) in TF's NHWC order, where a
    # channel-first flatten would read 1 (channel 0, column 1)
    weights = {name: np.zeros(shape, np.float32) for name, shape in TENSOR_SHAPES.items()}
    weights["Variable_9"] = np.arange(1, 65, dtype=np.float32)
    weights["Variable_10"][1, 0] = 1.0
    for name in ("Variable_12", "Variable_14", "Variable_16", "Variable_18"):
        weights[name][0, 0] = 1.0
    expected = 2.0 * math.atan(2.0) * 180.0 / math.pi
    np.testing.assert_allclose(reference_forward(weights, np.zeros((2, 66, 200, 3), np.float32)),
                               [expected, expected], rtol=1e-12)
    np.savez(tmp_path / "weights.npz", **weights)  # the port reads the same element
    np.testing.assert_allclose(port_predict(port_from(tmp_path / "weights.npz"), random_frames(n=2)),
                               [expected, expected], rtol=1e-6)


@pytest.mark.parametrize("stride", [1, 2])
def test_conv2d_valid_is_tf_valid_convolution(stride):
    rng = np.random.default_rng(3)
    x, kernel, bias = rng.normal(size=(2, 9, 12, 3)), rng.normal(size=(3, 5, 3, 4)), rng.normal(size=4)
    out = conv2d_valid(x, kernel, bias, stride)
    rows, cols = -(-(9 - 3 + 1) // stride), -(-(12 - 5 + 1) // stride)  # TF VALID: ceil((in - k + 1) / s)
    assert out.shape == (2, rows, cols, 4)
    expected = np.empty_like(out)
    for n in range(2):
        for i in range(rows):
            for j in range(cols):
                patch = x[n, i * stride:i * stride + 3, j * stride:j * stride + 5, :]
                expected[n, i, j] = np.einsum("hwc,hwco->o", patch, kernel) + bias
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


# --- the recipe's preprocessing -------------------------------------------------------------

def test_the_preprocessing_is_the_recipes_own(tmp_path):
    path = tmp_path / "0.jpg"
    cv2.imwrite(str(path), random_frames(n=1)[0])
    frame = cv2.imread(str(path))
    recipe = recipe_input(path)

    x = preprocess_legacy(frame)
    assert x.dtype == np.float32 and x.shape == (66, 200, 3)
    np.testing.assert_array_equal(x, recipe.astype(np.float32))  # BGR kept, [0, 1]

    u = to_model_input(frame)
    assert u.dtype == np.uint8 and u.shape == (66, 200, 3)
    np.testing.assert_array_equal(u, np.rint(recipe[:, :, ::-1] * 255.0).astype(np.uint8))  # RGB
    # Sightline's own input is resized with INTER_AREA; this model never sees it
    assert not np.array_equal(u, preprocess_frame(frame))


def test_weights_missing_a_tensor_or_of_the_wrong_shape_are_refused(tmp_path):
    weights = random_weights()
    del weights["Variable_7"]
    np.savez(tmp_path / "missing.npz", **weights)
    with pytest.raises(ValueError, match="Variable_7"):
        load_weights(tmp_path / "missing.npz")

    weights = random_weights()
    weights["Variable_10"] = weights["Variable_10"].T
    np.savez(tmp_path / "transposed.npz", **weights)
    with pytest.raises(ValueError, match=r"Variable_10 .*\(1164, 1152\).*\(1152, 1164\)"):
        load_weights(tmp_path / "transposed.npz")


# --- the evaluation on synthetic drives ---------------------------------------------------

N2018, N2017, BLOCK, FIREWALL = 1300, 300, 100, 20
#: a single-frame sensor dropout inside the 2018 blocked test frames (520..579 and 1220..1299;
#: the forward test frames are 1125..1299, so the two sets share 80 frames, as on the real drive)
DROPOUT = 525
#: 2017 labels at and past half a turn; exactly 180 is not beyond it
EXTREME_LABELS = {10: 250.0, 11: -200.5, 12: 180.0}


def cached(angles: np.ndarray) -> np.ndarray:
    """Angles as the drive cache stores them (float32), read back as float64 as the run reports do."""
    return np.asarray(angles).astype(np.float32).astype(np.float64)


def spread(frames: np.ndarray) -> np.ndarray:
    """The parity sample of a set: PARITY_FRAMES_PER_SET frames spread evenly over it (all, if fewer)."""
    take = min(PARITY_FRAMES_PER_SET, frames.size)
    return frames[np.unique(np.linspace(0, frames.size - 1, take).round().astype(int))]


def set_angles(drive, angles: dict[int, float]) -> None:
    """Rewrite the angle of the given rows of a drive's label file."""
    lines = (drive / "data.txt").read_text().splitlines()
    for row, angle in angles.items():
        name, rest = lines[row].split(maxsplit=1)
        stamp = rest.split(",", 1)[1] if "," in rest else None
        lines[row] = f"{name} {angle:.6f}" + (f",{stamp}" if stamp else "")
    (drive / "data.txt").write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    root = tmp_path_factory.mktemp("legacy")
    data = root / "data"
    make_drive(data, "drive2018", n=N2018, dropouts=(DROPOUT,))
    set_angles(make_drive(data, "drive2017", n=N2017, timestamps=False), EXTREME_LABELS)
    weights = random_weights()
    path = root / "private" / "weights.npz"
    path.parent.mkdir()
    np.savez(path, **weights)
    out = root / "reports" / "legacy.json"
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)  # a relative default folder would land here, where a test checks none was made
        # every route to a drive cache (building one, opening one, mapping its file) raises
        # instead of touching a file, so the evaluation succeeds only without one
        mp.setattr("sightline.data.cache.build_cache", no_cache)
        mp.setattr("sightline.data.cache.DriveCache", no_cache)
        mp.setattr(np, "memmap", no_cache)
        assert main(["legacy", "--weights", str(path), "--out", str(out), "--data-dir", str(data),
                     "--block", str(BLOCK), "--firewall", str(FIREWALL)]) == 0
    ctx = SimpleNamespace(root=root, data=data, path=path, text=out.read_text())
    ctx.report = json.loads(ctx.text)
    ctx.labels = {name: repair_dropouts(load_labels(data / name)) for name in ("drive2018", "drive2017")}
    ctx.sets = {
        "drive2018_blocked_test": ("drive2018", make_split("blocked", N2018, firewall=FIREWALL, block=BLOCK).test),
        "drive2018_forward_test": ("drive2018", make_split("forward", N2018, firewall=FIREWALL, block=BLOCK).test),
        "drive2017_all_seen": ("drive2017", np.arange(CROSS_FIRST_FRAME, N2017)),
    }
    # the TF1 graph on every scored frame, decoded and preprocessed exactly as the recipe did
    ctx.reference = {
        key: reference_forward(weights, np.stack([recipe_input(ctx.labels[drive].frame_path(i)) for i in frames]))
        for key, (drive, frames) in ctx.sets.items()
    }
    return ctx


def test_the_report_names_the_recipe_and_has_the_contract_keys(evaluated):
    r = evaluated.report
    assert {"recipe", "trained_on", "port_parity", "eval", "atan_cap"} <= r.keys()
    assert r["recipe"] == RECIPE == (
        "PilotNet in TensorFlow 1: shuffled 80/20 frame split, MSE in radians plus L2, atan output head limited "
        "to ±180°")
    assert "\u00b7" not in RECIPE  # the page shows it: no middle dot
    assert r["trained_on"] == "drive2017"
    assert r["eval"].keys() == {"drive2018_blocked_test", "drive2018_forward_test", "drive2017_all_seen"}
    for key, score in r["eval"].items():
        assert score.keys() == {"model", "regimes", "pearson_r", "zero_mae_deg"}, key
        assert score["regimes"].keys() == {"straight", "gentle", "sharp", "balanced_mae_deg"}, key
    assert r["port_parity"].keys() == {"frames", "max_abs_diff_deg"}
    assert r["atan_cap"].keys() == {"max_abs_pred_deg", "labels_beyond_180"}
    assert (r["device"], r["splits"]) == ("cpu", {"firewall": FIREWALL, "block": BLOCK})
    assert r["seen_first_frame"] == CROSS_FIRST_FRAME  # the first drive2017 frame the run reports score
    dt_2018 = evaluated.labels["drive2018"].median_dt_s()
    assert dt_2018 == pytest.approx(0.057, abs=1e-6)
    for name, n, dt_s in (("drive2018", N2018, dt_2018), ("drive2017", N2017, None)):
        assert r["drives"][name] == {"frames": n, "repaired_labels": int(name == "drive2018"),
                                     "median_dt_s": dt_s,
                                     "label_sha256": label_sha256(evaluated.data / name / "data.txt")}
    assert r["weights_sha256"] == hashlib.sha256(evaluated.path.read_bytes()).hexdigest()
    assert evaluated.text == json.dumps(r, indent=2, sort_keys=True) + "\n"  # plain sorted JSON


def test_each_set_is_scored_on_its_frames_against_the_repaired_labels(evaluated):
    raw = load_labels(evaluated.data / "drive2018").angles_deg
    repaired = evaluated.labels["drive2018"].angles_deg
    assert DROPOUT in evaluated.sets["drive2018_blocked_test"][1]
    assert raw[DROPOUT] == 0.0 and abs(repaired[DROPOUT]) > 15.0  # the repair moves the zero baseline
    for key, (drive, frames) in evaluated.sets.items():
        labels = evaluated.labels[drive].angles_deg[frames]
        target = cached(labels)  # what the run reports score against on the same frames
        ref = evaluated.reference[key]
        score = evaluated.report["eval"][key]
        expected = regression_metrics(ref, target)
        assert score["model"]["n"] == frames.size, key
        for metric in ("mae_deg", "rmse_deg", "medae_deg"):
            assert score["model"][metric] == pytest.approx(expected[metric], abs=TOL_DEG), (key, metric)
        assert score["zero_mae_deg"] == float(np.abs(target).mean()), key
        assert score["zero_mae_deg"] != float(np.abs(labels).mean()), key  # the rounding shows here
        regimes = regime_breakdown(ref, target)
        for regime in ("straight", "gentle", "sharp"):
            assert score["regimes"][regime]["n"] == regimes[regime]["n"], (key, regime)
            assert score["regimes"][regime]["mae_deg"] == pytest.approx(regimes[regime]["mae_deg"], abs=TOL_DEG)
        assert score["pearson_r"] == pytest.approx(pearson_r(ref, target), abs=1e-5), key


def test_the_port_is_checked_against_the_tf1_graph_on_frames_of_every_set(evaluated):
    blocked, forward = (evaluated.sets[f"drive2018_{protocol}_test"][1] for protocol in ("blocked", "forward"))
    assert np.intersect1d(blocked, forward).size == 80  # as on the real drive, the test sets overlap
    sampled = {drive: np.unique(np.concatenate([spread(frames) for d, frames in evaluated.sets.values() if d == drive]))
               for drive in ("drive2018", "drive2017")}
    assert sampled["drive2018"].size < 2 * PARITY_FRAMES_PER_SET  # some sampled frames are in both sets
    parity = evaluated.report["port_parity"]
    assert parity["frames"] == sum(frames.size for frames in sampled.values())
    # the port runs in float32 and the graph in float64, so frames compared are never all equal
    assert 0.0 < parity["max_abs_diff_deg"] < TOL_DEG


def test_the_cap_and_the_labels_beyond_it(evaluated):
    cap = evaluated.report["atan_cap"]
    assert cap["labels_beyond_180"] == {"drive2017": {"n": 2, "frac": 2 / N2017},
                                        "drive2018": {"n": 0, "frac": 0.0}}
    largest = max(float(np.abs(ref).max()) for ref in evaluated.reference.values())
    assert cap["max_abs_pred_deg"] == pytest.approx(largest, abs=TOL_DEG) and cap["max_abs_pred_deg"] < 180.0


def test_the_evaluation_reads_the_jpegs_and_keeps_the_weights_file_private(evaluated):
    # it succeeded while every route to a cache raised (see the fixture) and made no cache folder
    assert not (evaluated.root / "runs").exists()
    for private in (str(evaluated.path), evaluated.path.name, str(evaluated.root)):
        assert private not in evaluated.text


# --- the command's errors -----------------------------------------------------------------

def run_legacy(tmp_path, *extra):
    return main(["legacy", "--out", str(tmp_path / "legacy.json"), "--data-dir", str(tmp_path / "data"), *extra])


@pytest.fixture
def weights_file(tmp_path):
    path = tmp_path / "weights.npz"
    np.savez(path, **random_weights())
    return path


def test_without_weights_the_command_says_how_to_give_them(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    code = run_legacy(tmp_path)
    line = error_line(capsys.readouterr().err)
    assert code == 2 and "--weights" in line and WEIGHTS_ENV in line


def test_the_weights_path_can_come_from_the_environment(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path / "elsewhere.npz"))
    code = run_legacy(tmp_path)
    line = error_line(capsys.readouterr().err)
    assert code == 2 and "elsewhere.npz" in line and "not found" in line


@pytest.mark.parametrize("damage, phrase", [
    ("absent", "not found"),
    ("missing tensor", "Variable_7"),
    ("wrong shape", "Variable_10"),
    ("not an npz", "not an .npz"),
])
def test_unusable_weights_are_a_one_line_error(tmp_path, capsys, damage, phrase):
    path = tmp_path / "weights.npz"
    weights = random_weights()
    if damage == "missing tensor":
        del weights["Variable_7"]
    elif damage == "wrong shape":
        weights["Variable_10"] = weights["Variable_10"].T
    if damage == "not an npz":
        path.write_text("not weights\n")
    elif damage != "absent":
        np.savez(path, **weights)
    make_drive(tmp_path / "data", "drive2018", n=5)
    make_drive(tmp_path / "data", "drive2017", n=5, timestamps=False)
    code = run_legacy(tmp_path, "--weights", str(path))
    line = error_line(capsys.readouterr().err)
    assert code == 2 and phrase in line, damage


@pytest.mark.parametrize("offset", [1.0, float("nan")])
def test_a_port_that_disagrees_with_the_tf1_graph_writes_no_report(tmp_path, capsys, monkeypatch, weights_file,
                                                                   offset):
    import sightline.legacy

    make_drive(tmp_path / "data", "drive2018", n=100)
    make_drive(tmp_path / "data", "drive2017", n=10, timestamps=False)
    graph = sightline.legacy.reference_forward
    monkeypatch.setattr(sightline.legacy, "reference_forward", lambda weights, x: graph(weights, x) + offset)
    code = run_legacy(tmp_path, "--weights", str(weights_file), "--block", "10", "--firewall", "2")
    line = error_line(capsys.readouterr().err)
    assert code == 2 and "differs from the TF1 graph" in line and f"limit {PARITY_LIMIT_DEG:g} deg" in line
    assert not (tmp_path / "legacy.json").exists()


def test_a_missing_drive_is_a_one_line_error(tmp_path, capsys, weights_file):
    make_drive(tmp_path / "data", "drive2018", n=5)
    code = run_legacy(tmp_path, "--weights", str(weights_file))
    line = error_line(capsys.readouterr().err)
    assert code == 2 and "drive 'drive2017' not found" in line


def test_a_drive_too_short_for_the_splits_is_a_one_line_error(tmp_path, capsys, weights_file):
    make_drive(tmp_path / "data", "drive2018", n=50)
    make_drive(tmp_path / "data", "drive2017", n=5, timestamps=False)
    code = run_legacy(tmp_path, "--weights", str(weights_file))
    line = error_line(capsys.readouterr().err)
    assert code == 2 and "cannot split 'drive2018'" in line and "too small" in line
    assert "block=1500, firewall=250" in line
    assert not (tmp_path / "legacy.json").exists()
