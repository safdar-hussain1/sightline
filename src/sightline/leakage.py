"""Why a shuffled split flatters a steering model, measured in time and in pixels.

Consecutive dashcam frames are near-duplicates: about 57 ms apart, same
road, same steering. Shuffle the frames before splitting and almost every
test frame has a training frame right next to it; split the drive into
blocks with a firewall and none is closer than ``2 * firewall + 1`` frames.

``temporal_gaps``
    For every validation or test frame, the distance in frames to the
    nearest training frame, summarised.
the lookup table the shuffled split rewards (``memoriser`` in the report)
    A predictor with no model at all: find the training frame that looks
    most like the test frame and copy its steering label. Frames are
    compared as 100x33 grayscale images (BT.601 luma of the cached RGB
    model input, area-resized) by pixel RMSE in gray levels (0-255). It is
    scored on the paired frames P(seed), the frames that both the shuffled
    split of that seed and the blocked split hold out for testing, so both
    splits are judged on identical frames.

``leakage_report`` assembles both, plus up to six example frames with the
most similar training frame under each split: how far apart they are in
frames and seconds, how alike they look, and what each was steering.
"""

from __future__ import annotations

import cv2
import numpy as np

from sightline.data.splits import PROTOCOLS, SplitIndices, make_split, paired_frames
from sightline.evaluate import regression_metrics

__all__ = [
    "LOOKUP_PROTOCOLS",
    "LUMA_BT601",
    "PAIR_QUANTILES",
    "THUMB_SIZE",
    "downsample",
    "frame_gaps",
    "leakage_report",
    "nearest_neighbours",
    "pair_positions",
    "temporal_gaps",
]

#: weights of the R, G and B channels in BT.601 luma
LUMA_BT601 = (0.299, 0.587, 0.114)
#: (width, height) the frames are compared at: the 200x66 model input halved
THUMB_SIZE = (100, 33)
#: the example frames sit at these quantiles of |angle| over the paired
#: frames; the frame with the largest |angle| follows them
PAIR_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
#: the splits whose training frames the lookup table searches
LOOKUP_PROTOCOLS = ("leaky", "blocked")


def frame_gaps(train: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """For each of ``frames``, the distance in frames to the nearest of ``train``,
    which may come in any order (a shuffled split's does)."""
    train = np.sort(np.asarray(train, dtype=np.int64))
    frames = np.asarray(frames, dtype=np.int64)
    if train.size == 0:
        raise ValueError("there are no training frames")
    pos = np.searchsorted(train, frames)
    before = train[np.maximum(pos - 1, 0)]
    after = train[np.minimum(pos, train.size - 1)]
    return np.minimum(np.abs(frames - before), np.abs(after - frames))


def temporal_gaps(split: SplitIndices, which: str, dt_s: float | None) -> dict:
    """How far the ``which`` frames (``"val"`` or ``"test"``) of ``split`` sit from its
    training frames: the minimum, median and 95th percentile of each frame's
    distance to the nearest training frame, the share of frames with a training
    frame right next to them (``frac_within_1``), and the median in seconds at
    ``dt_s`` seconds per frame (``None`` when the drive has no timestamps)."""
    if which not in ("val", "test"):
        raise ValueError(f"which must be 'val' or 'test', got {which!r}")
    gaps = frame_gaps(split.train, getattr(split, which))
    median = float(np.median(gaps))
    return {
        "min_frames": int(gaps.min()),
        "median_frames": median,
        "p95_frames": float(np.percentile(gaps, 95)),
        "frac_within_1": float(np.mean(gaps <= 1)),
        "median_s": None if dt_s is None else median * float(dt_s),
    }


def downsample(inputs: np.ndarray, indices: np.ndarray, chunk: int = 128) -> np.ndarray:
    """``inputs[indices]`` as float32 rows of 33x100 gray levels, one row per index.

    ``inputs`` holds uint8 RGB frames (N x H x W x 3, e.g. the drive cache's
    memmap). Each frame becomes BT.601 luma, resized to 100x33 with
    ``cv2.INTER_AREA`` (for the 200x66 model input: the mean of each 2x2
    block). Frames are read ``chunk`` at a time, so a drive is never loaded whole.
    """
    if inputs.ndim != 4 or inputs.shape[-1] != 3:
        raise ValueError(f"expected N x H x W x 3 RGB frames, got shape {inputs.shape}")
    indices = np.asarray(indices, dtype=np.int64)
    width, height = THUMB_SIZE
    red, green, blue = LUMA_BT601
    out = np.empty((indices.size, height * width), dtype=np.float32)
    for start in range(0, indices.size, chunk):
        rgb = np.asarray(inputs[indices[start:start + chunk]], dtype=np.float64)
        luma = (rgb[..., 0] * red + rgb[..., 1] * green + rgb[..., 2] * blue).astype(np.float32)
        for k, frame in enumerate(luma):
            out[start + k] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA).ravel()
    return out


def _squared_norms(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    return np.einsum("ij,ij->i", rows, rows)


def nearest_neighbours(query: np.ndarray, pool: np.ndarray, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """For each row of ``query``, the row of ``pool`` nearest in Euclidean distance.

    Returns the pool row of each query row's neighbour and the RMSE between the
    two rows. The pool is searched ``chunk`` rows at a time in float64 with the
    expansion ``|q|^2 - 2 q.p + |p|^2`` (clipped at 0), so neither a full
    distance matrix nor a float64 copy of the pool is ever held. Any pool row
    the expansion cannot rank against the best within its rounding error is
    compared again directly: the nearest wins, ties go to the lowest pool row,
    and the RMSE comes from the direct comparison, so an exact copy scores 0
    and the answer depends on neither ``chunk`` nor the order of floating-point
    sums.
    """
    query, pool = np.asarray(query), np.asarray(pool)
    if query.ndim != 2 or pool.ndim != 2 or query.shape[1] != pool.shape[1]:
        raise ValueError(f"query {query.shape} and pool {pool.shape} must be 2-D with the same width")
    if pool.shape[0] == 0:
        raise ValueError("the pool is empty")
    if chunk < 1:
        raise ValueError(f"chunk must be at least 1, got {chunk}")
    if query.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    width = query.shape[1]
    q = query.astype(np.float64)
    q_sq = _squared_norms(q)
    starts = range(0, pool.shape[0], chunk)
    p_sq = np.concatenate([_squared_norms(pool[s:s + chunk]) for s in starts])
    # each expanded distance is off by at most (width + 2) * eps * (|q|^2 + max |p|^2);
    # a row within four times that of the best might truly be the nearest, so it is
    # kept for the direct comparison
    slack = 4.0 * (width + 2) * np.finfo(np.float64).eps * (q_sq + p_sq.max())

    best = np.full(q.shape[0], np.inf)
    kept: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for s in starts:
        p = pool[s:s + chunk].astype(np.float64)
        dist = q_sq[:, None] - 2.0 * (q @ p.T) + p_sq[None, s:s + p.shape[0]]
        np.maximum(dist, 0.0, out=dist)
        best = np.minimum(best, dist.min(axis=1))
        qi, pj = np.nonzero(dist <= (best + slack)[:, None])
        kept.append((qi, pj + s, dist[qi, pj]))
    qi, pj, dist = (np.concatenate(parts) for parts in zip(*kept))
    close = dist <= best[qi] + slack[qi]
    qi, pj = qi[close], pj[close]

    exact = np.empty(qi.size, dtype=np.float64)
    for s in range(0, qi.size, chunk):
        diff = q[qi[s:s + chunk]] - pool[pj[s:s + chunk]].astype(np.float64)
        exact[s:s + chunk] = (diff * diff).sum(axis=1)
    order = np.lexsort((pj, exact, qi))  # by query row, then distance, then pool row
    rows_in_order = qi[order]
    first = order[np.r_[True, rows_in_order[1:] != rows_in_order[:-1]]]
    return pj[first], np.sqrt(exact[first] / width)


def pair_positions(angles_deg: np.ndarray) -> list[int]:
    """Positions of the example frames among ``angles_deg``.

    For each of ``PAIR_QUANTILES``, the frame whose |angle| is nearest that
    quantile of |angle|; then the frame with the largest |angle|. Ties go to
    the earliest position and a frame picked twice is kept once, at its first
    place.
    """
    magnitude = np.abs(np.asarray(angles_deg, dtype=np.float64))
    if magnitude.size == 0:
        raise ValueError("there are no frames to pick from")
    picks = [int(np.argmin(np.abs(magnitude - np.quantile(magnitude, q)))) for q in PAIR_QUANTILES]
    picks.append(int(np.argmax(magnitude)))
    return list(dict.fromkeys(picks))


def leakage_report(cache, drive: str, seed: int, dt_s: float | None,
                   firewall: int = 250, block: int = 1500) -> dict:
    """The leakage report of a cached drive (see the module docstring).

    ``cache`` is a drive cache (``inputs`` and ``angles_deg``); ``seed`` picks the
    shuffled split, ``firewall`` and ``block`` the blocked and forward splits,
    exactly as in training. ``dt_s`` is the drive's median frame interval, or
    ``None`` when it has no timestamps. Every value is a plain int, float, str
    or None.
    """
    n = len(cache)
    splits = {p: make_split(p, n, seed=seed, firewall=firewall, block=block) for p in PROTOCOLS}
    frames = paired_frames(splits["blocked"], splits["leaky"])
    if frames.size == 0:
        raise ValueError("the blocked and shuffled test sets share no frame")
    angles = np.asarray(cache.angles_deg, dtype=np.float64)
    query = downsample(cache.inputs, frames)

    memoriser: dict = {}
    nearest: dict = {}
    for protocol in LOOKUP_PROTOCOLS:
        pool = np.sort(splits[protocol].train)
        rows, rmse = nearest_neighbours(query, downsample(cache.inputs, pool))
        nn = pool[rows]
        gaps = np.abs(nn - frames)
        memoriser[protocol] = {
            **regression_metrics(angles[nn], angles[frames]),
            "median_gap_frames": float(np.median(gaps)),
            "median_pixel_rmse": float(np.median(rmse)),
        }
        nearest[protocol] = (nn, gaps, rmse)

    pairs = []
    for i in pair_positions(angles[frames]):
        pair: dict = {"frame": int(frames[i]), "angle_deg": float(angles[frames[i]])}
        for protocol, (nn, gaps, rmse) in nearest.items():
            pair[protocol] = {
                "nn_frame": int(nn[i]),
                "gap_frames": int(gaps[i]),
                "gap_s": None if dt_s is None else int(gaps[i]) * float(dt_s),
                "pixel_rmse": float(rmse[i]),
                "nn_angle_deg": float(angles[nn[i]]),
            }
        pairs.append(pair)

    return {
        "drive": drive,
        "seed": int(seed),
        "dt_s": None if dt_s is None else float(dt_s),
        "splits": {"firewall": int(firewall), "block": int(block)},
        "gaps": {p: {which: temporal_gaps(s, which, dt_s) for which in ("val", "test")}
                 for p, s in splits.items()},
        "memoriser": memoriser,
        "pairs": pairs,
    }
