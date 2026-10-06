"""ONNX export with measured parity.

The exported graph takes raw uint8 RGB frames (N, 66, 200, 3k) — the current
frame and any history frames stacked along channels, each exactly what a
browser canvas yields — and returns steering degrees plus the VisualBackProp
saliency map from one forward pass. Normalization lives inside the graph, so
there is no client-side preprocessing to drift. The graph's metadata names the
model's ``frame_lags`` (a JSON list), so a client knows which earlier frames to
stack, its ``model`` and, when given, the ``run`` it was trained in.

An export is kept only when onnxruntime on the CPU reproduces the PyTorch
outputs on the given frames within tolerance; otherwise no file is left behind.
The measured maximum differences are returned and published, not assumed. The
same weights always export to the same bytes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import warnings
from pathlib import Path

import numpy as np
import torch

from sightline.paths import MissingDataError, Workspace
from sightline.saliency import SteeringWithSaliency

__all__ = [
    "ADAPTER_CHECK_WARNING",
    "ANGLE_TOLERANCE_DEG",
    "INPUT_NAME",
    "OPSET",
    "OUTPUT_NAMES",
    "SALIENCY_TOLERANCE",
    "export_onnx",
    "export_run",
    "open_drive_cache",
]

#: parity gate, degrees. fp32 graph replayed by a different runtime lands
#: far below this; the gate exists to catch structural export bugs.
ANGLE_TOLERANCE_DEG = 1e-3
SALIENCY_TOLERANCE = 1e-3
OPSET = 17
INPUT_NAME = "frames"
OUTPUT_NAMES = ("steering_deg", "saliency")
#: the tracer's warning about the input adapter's channel-count check, which runs once while
#: tracing on the example's fixed shape; the graph's channel count is fixed, so it does not
#: apply. Every other tracer warning is left to show.
ADAPTER_CHECK_WARNING = {"message": "Converting a tensor to a Python boolean",
                         "category": torch.jit.TracerWarning, "module": r"sightline\.models\.adapter$"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _max_differences(wrapper: torch.nn.Module, session, frames: np.ndarray, batch_size: int) -> tuple[float, float]:
    """The largest |PyTorch - onnxruntime| difference of the angle and of the saliency; NaN
    when either runtime produced a NaN (np.maximum carries it through every later batch)."""
    angle_diff = saliency_diff = 0.0
    for start in range(0, len(frames), batch_size):
        chunk = np.ascontiguousarray(frames[start:start + batch_size])
        with torch.no_grad():
            torch_angle, torch_saliency = wrapper(torch.from_numpy(chunk.copy()))
        ort_angle, ort_saliency = session.run(None, {INPUT_NAME: chunk})
        angle_diff = float(np.maximum(angle_diff, np.abs(torch_angle.numpy() - ort_angle).max()))
        saliency_diff = float(np.maximum(saliency_diff, np.abs(torch_saliency.numpy() - ort_saliency).max()))
    return angle_diff, saliency_diff


def export_onnx(
    model: torch.nn.Module,
    out_path: str | Path,
    check_inputs: np.ndarray,
    *,
    model_name: str | None = None,
    run: str | None = None,
    max_bytes: int | None = None,
    batch_size: int = 64,
    angle_tolerance_deg: float | None = None,
    saliency_tolerance: float | None = None,
) -> dict:
    """Export ``model`` (+saliency) to ONNX and verify it against onnxruntime on the CPU.

    ``check_inputs`` are uint8 frame stacks (N, 66, 200, 3·(1+len(frame_lags))); both
    runtimes score all of them, ``batch_size`` at a time, and must agree within
    ``angle_tolerance_deg`` and ``saliency_tolerance`` (``ANGLE_TOLERANCE_DEG`` and
    ``SALIENCY_TOLERANCE`` unless given). The metadata records ``frame_lags``, ``model``
    (``model_name``, or the architecture's class name) and ``run`` when given. The file is
    written next to ``out_path`` under a temporary name and moved into place only after the
    parity check passes and, with ``max_bytes``, the size is within it: a RuntimeError
    (parity) or ValueError (size) leaves no file.
    """
    import onnx
    import onnxruntime as ort

    out_path = Path(out_path)
    angle_gate = ANGLE_TOLERANCE_DEG if angle_tolerance_deg is None else angle_tolerance_deg
    saliency_gate = SALIENCY_TOLERANCE if saliency_tolerance is None else saliency_tolerance
    check_inputs = np.asarray(check_inputs)
    lags = [int(lag) for lag in getattr(model, "frame_lags", ())]
    channels = 3 * (1 + len(lags))
    if check_inputs.dtype != np.uint8 or check_inputs.ndim != 4 or check_inputs.shape[1:] != (66, 200, channels):
        raise ValueError(f"check inputs must be uint8 (N, 66, 200, {channels}) frame stacks, "
                         f"got {check_inputs.dtype} {check_inputs.shape}")
    if len(check_inputs) == 0:
        raise ValueError("no check inputs: parity must be measured on at least one frame stack")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(f".{out_path.name}.{os.getpid()}-{secrets.token_hex(4)}.partial")
    wrapper = SteeringWithSaliency(model).eval()
    try:
        # the traced graph depends only on the input's shape, so a zero frame stack serves as
        # the example and the bytes depend on the weights alone
        example = torch.zeros((1, 66, 200, channels), dtype=torch.uint8)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", **ADAPTER_CHECK_WARNING)
            torch.onnx.export(
                wrapper,
                (example,),
                str(partial),
                input_names=[INPUT_NAME],
                output_names=list(OUTPUT_NAMES),
                dynamic_axes={
                    INPUT_NAME: {0: "batch"},
                    OUTPUT_NAMES[0]: {0: "batch"},
                    OUTPUT_NAMES[1]: {0: "batch"},
                },
                opset_version=OPSET,
                dynamo=False,
            )
        graph = onnx.load(str(partial))
        metadata = {"frame_lags": json.dumps(lags), "model": model_name or type(model).__name__.lower()}
        if run is not None:
            metadata["run"] = run
        onnx.helper.set_model_props(graph, metadata)
        onnx.save(graph, str(partial))

        session = ort.InferenceSession(str(partial), providers=["CPUExecutionProvider"])
        angle_diff, saliency_diff = _max_differences(wrapper, session, check_inputs, batch_size)
        if not angle_diff <= angle_gate:  # a NaN fails too
            raise RuntimeError(f"ONNX angle parity {angle_diff:.3g} deg exceeds {angle_gate:g} deg "
                               f"on {len(check_inputs)} frame stacks")
        if not saliency_diff <= saliency_gate:
            raise RuntimeError(f"ONNX saliency parity {saliency_diff:.3g} exceeds {saliency_gate:g} "
                               f"on {len(check_inputs)} frame stacks")
        size = partial.stat().st_size
        if max_bytes is not None and size > max_bytes:
            raise ValueError(f"the ONNX graph of {metadata['model']} is {size:,} bytes, "
                             f"over the {max_bytes:,}-byte limit")
        os.replace(partial, out_path)
    finally:
        with contextlib.suppress(OSError):
            partial.unlink(missing_ok=True)

    return {
        "path": str(out_path),
        "bytes": size,
        "sha256": _sha256(out_path),
        "opset": OPSET,
        "model": metadata["model"],
        "run": run,
        "frame_lags": lags,
        "check_frames": int(len(check_inputs)),
        "max_angle_diff_deg": angle_diff,
        "max_saliency_diff": saliency_diff,
        "angle_tolerance_deg": angle_gate,
        "saliency_tolerance": saliency_gate,
    }


def open_drive_cache(workspace: Workspace, drive: str):
    """A drive's dropout-repaired labels and its cache, checked against them: ``(labels, cache)``.

    Never builds a cache: a missing one, or one built from other labels or settings,
    is a MissingDataError naming ``sightline cache``, so nothing is computed from a
    stale cache.
    """
    from sightline.data.cache import open_cache
    from sightline.data.labels import load_labels, repair_dropouts

    cache_dir = workspace.cache_dir(drive)
    if not (cache_dir / "meta.json").is_file():
        raise MissingDataError(
            f"no cache for drive {drive!r} at {cache_dir}; run `sightline cache --drive {drive}` first"
        )
    labels = repair_dropouts(load_labels(workspace.drive(drive)))
    return labels, open_cache(labels, cache_dir)


def export_run(run_dir: Path, out_path: Path, workspace: Workspace, check_frames: int = 64) -> dict:
    """Export a trained run, parity-checked on frames of its test split.

    Writes the result beside the graph as ``<name>.export.json`` too.
    """
    from sightline.data.frames import stack_frames
    from sightline.reporting import load_run_model, load_split_indices, write_report

    run_dir, out_path = Path(run_dir), Path(out_path)
    model, history = load_run_model(run_dir)
    _, cache = open_drive_cache(workspace, history["config"]["notes"]["drive"])
    test_idx = np.sort(load_split_indices(run_dir, "test"))
    step = max(1, test_idx.size // check_frames)
    sample = stack_frames(cache.inputs, test_idx[::step][:check_frames], model.frame_lags)

    result = export_onnx(model, out_path, sample, model_name=history["config"]["model"], run=run_dir.name)
    write_report(out_path.with_suffix(".export.json"), result)
    return result
