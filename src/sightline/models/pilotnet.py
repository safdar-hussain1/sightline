"""PilotNet, ported faithfully.

The classic end-to-end steering architecture (Bojarski et al. 2016, as
popularized by the TensorFlow 1 community implementation): five VALID-padded
conv layers (24-36-48-64-64), four fully-connected layers (1164-100-50-10),
and a ``2*atan(z)`` output head in radians.

The atan head is kept deliberately: its output range is hard-capped at
(-180 deg, +180 deg), while real steering labels in the benchmark drives
reach +501.8 deg. That structural cap is one of the standard failure modes
Sightline measures, so the port must reproduce it exactly.

``load_legacy_npz`` fills this module with weights extracted from the
original TF1 checkpoint (tensors named Variable..Variable_19 in creation
order), converting HWIO conv kernels to OIHW and re-ordering the first FC
layer's input dimension from NHWC flatten order to NCHW flatten order.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sightline.models.adapter import InputAdapter

__all__ = ["PilotNet"]


class PilotNet(nn.Module):
    #: feature-map geometry after the conv stack on a 66x200 input
    FEAT_C, FEAT_H, FEAT_W = 64, 1, 18

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.adapter = InputAdapter()
        self.conv = nn.ModuleList(
            [
                nn.Conv2d(3, 24, 5, stride=2),
                nn.Conv2d(24, 36, 5, stride=2),
                nn.Conv2d(36, 48, 5, stride=2),
                nn.Conv2d(48, 64, 3, stride=1),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )
        self.fc = nn.ModuleList(
            [
                nn.Linear(self.FEAT_C * self.FEAT_H * self.FEAT_W, 1164),
                nn.Linear(1164, 100),
                nn.Linear(100, 50),
                nn.Linear(50, 10),
            ]
        )
        self.head = nn.Linear(10, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """uint8 NHWC RGB frames -> steering angle in degrees."""
        x = self.adapter(x)
        for conv in self.conv:
            x = torch.relu(conv(x))
        x = torch.flatten(x, 1)
        for fc in self.fc:
            x = self.drop(torch.relu(fc(x)))
        radians = 2.0 * torch.atan(self.head(x))
        return radians * (180.0 / math.pi)

    def load_legacy_npz(self, path: str | Path) -> None:
        """Load weights extracted from the original TF1 checkpoint.

        The original network consumed BGR frames scaled to [0, 1]; this
        module's adapter emits RGB scaled to [-1, 1]. Both differences are
        absorbed exactly into the first conv layer (it is linear and VALID
        padded): kernels are channel-swapped and halved, and the bias gains
        half of each kernel's tap sum, so ``conv1'(x_rgb/127.5 - 1)``
        equals ``conv1(x_bgr/255)`` for every pixel.
        """
        data = np.load(path)
        names = [f"Variable_{i}" if i else "Variable" for i in range(20)]
        missing = [n for n in names if n not in data]
        if missing:
            raise KeyError(f"legacy npz missing tensors: {missing}")

        tensors = [torch.from_numpy(np.asarray(data[n], dtype=np.float32)) for n in names]
        with torch.no_grad():
            for layer_idx, conv in enumerate(self.conv):
                w, b = tensors[2 * layer_idx], tensors[2 * layer_idx + 1]
                w = w.permute(3, 2, 0, 1).contiguous()  # HWIO -> OIHW
                if layer_idx == 0:
                    w = w.flip(1) / 2.0  # BGR<->RGB swap, then [0,1] -> [-1,1] scale
                    b = b + w.sum(dim=(1, 2, 3))
                conv.weight.copy_(w)
                conv.bias.copy_(b)
            fc_tensors = tensors[10:]
            for layer_idx, fc in enumerate([*self.fc, self.head]):
                w, b = fc_tensors[2 * layer_idx], fc_tensors[2 * layer_idx + 1]
                if layer_idx == 0:
                    # TF flattens the (H, W, C) feature map; torch flattens
                    # (C, H, W). Re-order the FC input dimension to match.
                    w = (
                        w.reshape(self.FEAT_H, self.FEAT_W, self.FEAT_C, -1)
                        .permute(2, 0, 1, 3)
                        .reshape(self.FEAT_C * self.FEAT_H * self.FEAT_W, -1)
                    )
                fc.weight.copy_(w.T)
                fc.bias.copy_(b)
