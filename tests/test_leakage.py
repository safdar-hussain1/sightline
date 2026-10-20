"""The leakage analysis: temporal gaps, the pixel lookup table and its example frame pairs.

A module-scoped fixture runs ``sightline leakage`` on a synthetic drive with scenery: its
background drifts slowly from frame to frame, so neighbouring frames look almost alike and
distant ones do not, as on a real drive. (Without scenery, every frame showing the same bar
position is pixel-identical, so a lookup finds an exact copy under any split.)
"""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import REPO, make_drive
from sightline import leakage
from sightline.cli import main
from sightline.data.cache import DriveCache
from sightline.data.labels import load_labels
from sightline.data.splits import SplitIndices, make_split, paired_frames
from sightline.leakage import downsample, nearest_neighbours, pair_positions, temporal_gaps

N, BLOCK, FIREWALL, SEED = 1600, 100, 20, 0
GAPS = {"min_frames", "median_frames", "p95_frames", "frac_within_1", "median_s"}
SIDE = {"nn_frame", "gap_frames", "gap_s", "pixel_rmse", "nn_angle_deg"}


def _args(drive, out, seed=SEED):
    return ["leakage", "--drive", drive.name, "--seed", str(seed), "--block", str(BLOCK),
            "--firewall", str(FIREWALL), "--out", str(out), *drive.loc]


def _digest(frames):
    """The run reports' frame digest, written out: sha256 of little-endian int64 indices."""
    return hashlib.sha256(np.asarray(frames).astype("<i8").tobytes()).hexdigest()


@pytest.fixture(scope="module")
def drive(tmp_path_factory):
    root = tmp_path_factory.mktemp("leakage")
    make_drive(root / "data", "drivetest", n=N, scenery=True)
    ctx = SimpleNamespace(name="drivetest", root=root,
                          loc=["--data-dir", str(root / "data"), "--runs-dir", str(root / "runs")],
                          out=root / "reports" / "leakage.json")
    assert main(_args(ctx, ctx.out)) == 0
    ctx.text = ctx.out.read_text()
    ctx.report = json.loads(ctx.text)
    ctx.cache = DriveCache(root / "runs" / "cache" / "drivetest")
    ctx.timestamps = load_labels(root / "data" / "drivetest").timestamps_s
    ctx.splits = {p: make_split(p, N, seed=SEED, firewall=FIREWALL, block=BLOCK)
                  for p in ("leaky", "blocked", "forward")}
    ctx.paired = paired_frames(ctx.splits["blocked"], ctx.splits["leaky"])
    return ctx


# --- nearest neighbours -----------------------------------------------------------------

def test_an_exact_copy_in_the_pool_is_found_at_rmse_zero():
    rng = np.random.default_rng(0)
    pool = rng.uniform(0, 255, (40, 3300)).astype(np.float32)
    pool[33] = pool[9]  # the same frame twice: the lower row wins
    rows, rmse = nearest_neighbours(pool[[9, 21]], pool, chunk=8)
    assert rows.tolist() == [9, 21]
    assert rmse.tolist() == [0.0, 0.0]


def test_rows_the_expansion_cannot_tell_apart_are_compared_directly():
    # |q|^2 is about 1e18 here, so the expansion's rounding (hundreds of units) swamps
    # squared distances of 4, 1 and 9; the direct comparison still ranks them
    base = np.random.default_rng(3).integers(10**7, 2 * 10**7, 3300).astype(np.float64)
    pool = np.stack([base, base, base])
    pool[:, 0] += [2.0, 1.0, 3.0]
    rows, rmse = nearest_neighbours(base[None], pool)
    assert rows.tolist() == [1]
    assert rmse[0] == np.sqrt(1 / 3300)


@pytest.mark.parametrize("chunk", [7, 512])
def test_nearest_neighbours_agree_with_brute_force(chunk):
    rng = np.random.default_rng(1)
    q, p = rng.uniform(0, 255, (23, 300)), rng.uniform(0, 255, (101, 300))
    dist = ((q[:, None] - p[None]) ** 2).sum(-1)
    rows, rmse = nearest_neighbours(q, p, chunk=chunk)
    np.testing.assert_array_equal(rows, np.argmin(dist, 1))
    np.testing.assert_allclose(rmse, np.sqrt(dist.min(1) / q.shape[1]), rtol=1e-12)


def test_downsample_is_bt601_luma_halved_by_area():
    rng = np.random.default_rng(2)
    frames = rng.integers(0, 256, (4, 66, 200, 3), dtype=np.uint8)
    frames[0] = (255, 0, 0)
    frames[1] = (0, 255, 0)
    frames[2] = (0, 0, 255)
    rows = downsample(frames, np.array([0, 1, 2, 3]), chunk=3)
    assert rows.shape == (4, 3300) and rows.dtype == np.float32
    np.testing.assert_allclose(rows[:3].mean(axis=1), [0.299 * 255, 0.587 * 255, 0.114 * 255], rtol=1e-6)
    luma = frames[3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    np.testing.assert_allclose(rows[3], luma.reshape(33, 2, 100, 2).mean(axis=(1, 3)).ravel(), atol=1e-3)
    np.testing.assert_array_equal(downsample(frames, np.array([3, 0])), rows[[3, 0]])  # rows follow the indices


# --- temporal gaps and the example frames -----------------------------------------------

def test_temporal_gaps_summarise_each_frames_distance_to_training():
    split = SplitIndices(train=np.array([10, 0, 1, 2]), val=np.array([5]), test=np.array([3, 4, 20, 6]))
    # test frames sit 1, 2, 10 and 4 frames from the nearest training frame
    assert temporal_gaps(split, "test", 0.5) == {
        "min_frames": 1, "median_frames": 3.0, "p95_frames": pytest.approx(9.1),
        "frac_within_1": 0.25, "median_s": 1.5}
    assert temporal_gaps(split, "val", None)["median_s"] is None
    with pytest.raises(ValueError, match="'val' or 'test'"):
        temporal_gaps(split, "train", 0.5)


def test_example_frames_sit_at_angle_quantiles_then_the_largest_angle():
    # |angle| sorted: 0.5 1 2 3 4 4 6 8 9 10; quantiles 0.725, 2.25, 4.0, 7.5, 9.55
    angles = np.array([-0.5, 1.0, 9.0, -2.0, 4.0, -4.0, 6.0, 3.0, -10.0, 8.0])
    # 4.0 ties positions 4 and 5 (the earlier wins); the largest |angle| was already picked
    assert pair_positions(angles) == [0, 3, 4, 9, 8]


# --- the report on a synthetic drive ----------------------------------------------------

def test_the_report_has_the_contract_keys(drive):
    r = drive.report
    assert {"drive", "seed", "dt_s", "paired", "gaps", "memoriser", "pairs"} <= r.keys()
    assert (r["drive"], r["seed"], r["splits"]) == ("drivetest", SEED, {"firewall": FIREWALL, "block": BLOCK})
    assert r["dt_s"] == pytest.approx(0.057, abs=1e-6)
    assert r["gaps"].keys() == {"leaky", "blocked", "forward"}
    for protocol, gaps in r["gaps"].items():
        assert gaps.keys() == {"val", "test"} and all(g.keys() == GAPS for g in gaps.values()), protocol
    assert r["memoriser"].keys() == {"leaky", "blocked"}
    for protocol, lookup in r["memoriser"].items():
        assert {"n", "mae_deg", "rmse_deg"} <= lookup.keys(), protocol
        assert lookup["n"] == drive.paired.size > 0
    for pair in r["pairs"]:
        assert pair.keys() == {"frame", "angle_deg", "leaky", "blocked"}
        assert pair["leaky"].keys() == pair["blocked"].keys() == SIDE


def test_blocked_test_frames_sit_beyond_the_firewall_and_shuffled_ones_next_to_training(drive):
    gaps = drive.report["gaps"]
    assert gaps["blocked"]["test"]["min_frames"] >= 2 * FIREWALL + 1 == 41
    assert gaps["leaky"]["test"]["min_frames"] == 1
    assert gaps["leaky"]["test"]["median_frames"] == 1 and gaps["leaky"]["test"]["frac_within_1"] > 0.8
    for protocol in ("blocked", "forward"):
        for which in ("val", "test"):
            assert gaps[protocol][which]["min_frames"] >= 41 and gaps[protocol][which]["frac_within_1"] == 0
    assert gaps["leaky"]["test"]["median_s"] == pytest.approx(drive.report["dt_s"])


def test_the_lookup_table_is_far_better_under_the_shuffled_split(drive):
    lookup = drive.report["memoriser"]
    assert lookup["blocked"]["mae_deg"] >= 3 * lookup["leaky"]["mae_deg"]
    # under the shuffled split the most similar training frame is the next one in time
    assert lookup["leaky"]["median_gap_frames"] <= 2 < 2 * FIREWALL + 1 <= lookup["blocked"]["median_gap_frames"]


def test_six_distinct_example_pairs_each_matched_to_a_training_frame(drive):
    pairs = drive.report["pairs"]
    frames = [p["frame"] for p in pairs]
    assert len(pairs) == 6 and len(set(frames)) == 6
    assert set(frames) <= set(drive.paired.tolist())
    angles, ts = drive.cache.angles_deg, drive.timestamps
    for protocol in ("leaky", "blocked"):
        train = set(drive.splits[protocol].train.tolist())
        for p in pairs:
            side = p[protocol]
            assert side["nn_frame"] in train, (protocol, p)
            assert side["gap_frames"] == abs(side["nn_frame"] - p["frame"])
            assert side["gap_s"] == pytest.approx(abs(ts[side["nn_frame"]] - ts[p["frame"]]), rel=1e-12)
            assert side["nn_angle_deg"] == float(angles[side["nn_frame"]])
            assert p["angle_deg"] == float(angles[p["frame"]])
    assert [p["frame"] for p in pairs] == drive.paired[pair_positions(angles[drive.paired])].tolist()


def test_the_report_matches_a_brute_force_lookup(drive):
    angles = drive.cache.angles_deg.astype(np.float64)

    def thumbs(idx):
        luma = drive.cache.inputs[idx].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
        return luma.reshape(len(idx), 33, 2, 100, 2).mean(axis=(2, 4)).reshape(len(idx), -1)

    query = thumbs(drive.paired)
    by_frame = {p["frame"]: p for p in drive.report["pairs"]}
    for protocol in ("leaky", "blocked"):
        train = np.sort(drive.splits[protocol].train)
        pool = thumbs(train)
        nn = np.array([train[np.argmin(((pool - q) ** 2).sum(1))] for q in query])
        err = angles[nn] - angles[drive.paired]
        lookup = drive.report["memoriser"][protocol]
        assert lookup["mae_deg"] == pytest.approx(np.abs(err).mean(), abs=1e-9), protocol
        assert lookup["rmse_deg"] == pytest.approx(np.sqrt((err ** 2).mean()), abs=1e-9), protocol
        for frame, neighbour in zip(drive.paired.tolist(), nn.tolist()):
            if frame in by_frame:
                assert by_frame[frame][protocol]["nn_frame"] == neighbour, (protocol, frame)


def test_the_report_is_the_same_every_time_as_plain_sorted_json(drive, tmp_path):
    def walk(value):
        if isinstance(value, dict):
            assert all(isinstance(k, str) for k in value)
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        else:
            assert type(value) in (int, float, str, type(None)), (type(value), value)

    walk(drive.report)
    assert drive.text == json.dumps(drive.report, indent=2, sort_keys=True) + "\n"
    again = tmp_path / "again.json"
    assert main(_args(drive, again)) == 0
    assert again.read_text() == drive.text


def test_pair_gaps_in_seconds_follow_the_timestamps_not_the_median_interval(tmp_path):
    # frame intervals alternate 40 and 74 ms: 300 of 40 ms and 299 of 74 ms, so the median
    # interval is 40 ms while the frames are 57 ms apart on average
    make_drive(tmp_path / "data", "driveuneven", n=600, intervals_ms=(40, 74))
    out = tmp_path / "leakage.json"
    ctx = SimpleNamespace(name="driveuneven", loc=["--data-dir", str(tmp_path / "data"),
                                                   "--runs-dir", str(tmp_path / "runs")])
    assert main(_args(ctx, out)) == 0
    report = json.loads(out.read_text())
    ts = load_labels(tmp_path / "data" / "driveuneven").timestamps_s
    assert report["dt_s"] == pytest.approx(0.040, abs=1e-6)
    sides = [(p["frame"], p[protocol]) for p in report["pairs"] for protocol in ("leaky", "blocked")]
    for frame, side in sides:
        assert side["gap_s"] == pytest.approx(abs(ts[side["nn_frame"]] - ts[frame]), rel=1e-12)
    # over two frames or more the median interval would understate every gap by 34 ms or more
    longer = [side for _, side in sides if side["gap_frames"] >= 2]
    assert longer and all(side["gap_s"] - side["gap_frames"] * report["dt_s"] > 0.030 for side in longer)
    # the gap summaries stay in frames, and their median_s is the median in frames times dt_s
    for gaps in report["gaps"].values():
        for g in gaps.values():
            assert g["median_s"] == pytest.approx(g["median_frames"] * report["dt_s"], rel=1e-12)


def test_the_module_gives_the_median_frame_interval_the_site_data_records_and_leaves_the_spread_to_it():
    """The intervals between frames are spread wide, so the module's opening paragraph names its figure
    as the median, the one results.json records under ``timing``, and points there for the rest."""
    timing = json.loads((REPO / "web" / "public" / "data" / "results.json").read_text())["timing"]
    median_ms = round(timing["drive2018"]["frame_interval_s"]["median"] * 1000)
    opening = " ".join(leakage.__doc__.split("\n\n")[1].split())
    assert f"{median_ms} ms apart at the median" in opening and "results.json ``timing``" in opening, opening


def test_the_seed_picks_the_shuffled_split_and_so_the_paired_frames(drive, tmp_path):
    out = tmp_path / "seed1.json"
    assert main(_args(drive, out, seed=1)) == 0
    report = json.loads(out.read_text())
    leaky = make_split("leaky", N, seed=1)
    paired = paired_frames(drive.splits["blocked"], leaky)
    assert paired.size != drive.paired.size  # seed 1's paired frames are not seed 0's
    assert report["seed"] == 1
    assert report["paired"] == {"n": int(paired.size), "frames_sha256": _digest(paired)}
    assert all(lookup["n"] == paired.size for lookup in report["memoriser"].values())
    assert {p["frame"] for p in report["pairs"]} <= set(paired.tolist())
    assert all(p["leaky"]["nn_frame"] in set(leaky.train.tolist()) for p in report["pairs"])


def test_the_paired_frames_carry_the_digest_the_run_reports_use(drive):
    from sightline import reporting
    from sightline.data import splits

    assert reporting.frames_sha256 is splits.frames_sha256  # one digest function for every report
    assert drive.report["paired"] == {"n": int(drive.paired.size), "frames_sha256": _digest(drive.paired)}
    assert drive.report["paired"]["frames_sha256"] == reporting.frames_sha256(drive.paired)


def test_a_drive_without_timestamps_reports_gaps_in_frames_only(tmp_path):
    make_drive(tmp_path / "data", "drivenotime", n=600, timestamps=False)
    out = tmp_path / "leakage.json"
    ctx = SimpleNamespace(name="drivenotime", loc=["--data-dir", str(tmp_path / "data"),
                                                   "--runs-dir", str(tmp_path / "runs")])
    assert main(_args(ctx, out)) == 0
    report = json.loads(out.read_text())
    assert report["dt_s"] is None
    assert all(g["median_s"] is None for gaps in report["gaps"].values() for g in gaps.values())
    assert report["pairs"] and all(p[s]["gap_s"] is None for p in report["pairs"] for s in ("leaky", "blocked"))


def _error_line(err: str) -> str:
    assert "Traceback" not in err
    lines = [line for line in err.splitlines() if line.startswith("error: ")]
    assert len(lines) == 1, err
    return lines[0]


def test_unusable_split_settings_are_a_one_line_error(drive, tmp_path, capsys):
    out = tmp_path / "leakage.json"
    assert main(["leakage", "--drive", drive.name, "--block", "30", "--firewall", "20",
                 "--out", str(out), *drive.loc]) == 2
    assert "block (30) must exceed twice the firewall (20)" in _error_line(capsys.readouterr().err)
    assert not out.exists()


@pytest.mark.parametrize("value", ["../drivetest", "data/drivetest", ".."])
def test_leakage_takes_a_drive_name_not_a_path(tmp_path, capsys, value):
    code = main(["leakage", "--drive", value, "--data-dir", str(tmp_path), "--runs-dir", str(tmp_path / "runs"),
                 "--out", str(tmp_path / "leakage.json")])
    assert code == 2 and "drive name" in _error_line(capsys.readouterr().err)


def test_a_missing_drive_is_a_clear_error(tmp_path, capsys):
    code = main(["leakage", "--drive", "drive2018", "--data-dir", str(tmp_path / "nowhere"),
                 "--runs-dir", str(tmp_path / "runs"), "--out", str(tmp_path / "leakage.json")])
    assert code == 2 and "drive 'drive2018' not found" in _error_line(capsys.readouterr().err)
