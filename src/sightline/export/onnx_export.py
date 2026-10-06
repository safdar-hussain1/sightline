"""ONNX export with measured parity.

The exported graph takes raw uint8 RGB frames (N, 66, 200, 3) — exactly what
a browser canvas yields — and returns steering degrees plus the VisualBackProp
saliency map from one forward pass. Normalization lives inside the graph, so
there is no client-side preprocessing to drift.

Export is only considered successful when onnxruntime reproduces the PyTorch
outputs on real frames within tolerance; the measured maximum differences are
returned and published, not assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from sightline.paths import MissingDataError, Workspace
from sightline.saliency import SteeringWithSaliency

__all__ = ["export_onnx", "export_run"]

#: parity gate, degrees. fp32 graph replayed by a different runtime lands
#: far below this; the gate exists to catch structural export bugs.
ANGLE_TOLERANCE_DEG = 1e-3
SALIENCY_TOLERANCE = 1e-3


def export_onnx(
    model: torch.nn.Module,
    out_path: str | Path,
    check_inputs: np.ndarray,
) -> dict:
    """Export ``model`` (+saliency) to ONNX and verify against onnxruntime."""
    import onnxruntime as ort

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = SteeringWithSaliency(model).eval()
    example = torch.from_numpy(check_inputs[:1].copy())
    torch.onnx.export(
        wrapper,
        (example,),
        str(out_path),
        input_names=["frames"],
        output_names=["steering_deg", "saliency"],
        dynamic_axes={
            "frames": {0: "batch"},
            "steering_deg": {0: "batch"},
            "saliency": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    with torch.no_grad():
        torch_angle, torch_sal = wrapper(torch.from_numpy(check_inputs.copy()))

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_angle, ort_sal = sess.run(None, {"frames": check_inputs})

    angle_diff = float(np.abs(torch_angle.numpy() - ort_angle).max())
    sal_diff = float(np.abs(torch_sal.numpy() - ort_sal).max())
    if angle_diff > ANGLE_TOLERANCE_DEG:
        raise RuntimeError(f"ONNX angle parity {angle_diff} deg exceeds {ANGLE_TOLERANCE_DEG}")
    if sal_diff > SALIENCY_TOLERANCE:
        raise RuntimeError(f"ONNX saliency parity {sal_diff} exceeds {SALIENCY_TOLERANCE}")

    return {
        "path": str(out_path),
        "bytes": out_path.stat().st_size,
        "check_frames": int(check_inputs.shape[0]),
        "max_angle_diff_deg": angle_diff,
        "max_saliency_diff": sal_diff,
    }


def export_run(run_dir: Path, out_path: Path, workspace: Workspace, check_frames: int = 64) -> dict:
    """Export a trained run, parity-checked on real test-split frames."""
    from sightline.data.cache import DriveCache
    from sightline.reporting import load_run_model

    model, history = load_run_model(run_dir)
    drive_name = history["config"]["notes"]["drive"]
    cache_dir = workspace.cache_dir(drive_name)
    if not (cache_dir / "meta.json").is_file():
        raise MissingDataError(
            f"no cache for drive {drive_name!r} at {cache_dir}; "
            f"run `sightline cache --drive {drive_name}` first"
        )
    cache = DriveCache(cache_dir)
    test_idx = np.load(run_dir / "test_idx.npy")
    step = max(1, test_idx.size // check_frames)
    sample = np.ascontiguousarray(cache.inputs[np.sort(test_idx)[::step][:check_frames]])

    result = export_onnx(model, out_path, sample)
    result["run"] = run_dir.name
    (out_path.with_suffix(".export.json")).write_text(json.dumps(result, indent=2))
    return result
