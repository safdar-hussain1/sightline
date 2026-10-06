"""Shared input adapter: raw uint8 RGB frames in, normalized NCHW float out.

Normalization lives inside the model graph on purpose: the exported ONNX
then accepts the exact bytes a browser canvas produces (HWC uint8 RGB), and
there is no client-side preprocessing that could drift from Python.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["InputAdapter"]


class InputAdapter(nn.Module):
    """(N, 66, 200, 3) uint8/float RGB -> (N, 3, 66, 200) float32 in [-1, 1]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[3] != 3:
            raise ValueError(f"expected NHWC RGB input, got shape {tuple(x.shape)}")
        x = x.to(torch.float32)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x / 127.5 - 1.0
