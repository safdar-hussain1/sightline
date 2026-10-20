"""SightlineNet: the compact steering network Sightline ships.

Same field of view and input budget as PilotNet, redesigned with current
practice:

* BatchNorm + SiLU after every conv (PilotNet is plain ReLU, no norm).
* A linear output head in degrees — unbounded, so the network can express
  the full label range (drives in the benchmark reach +501.8 deg; an atan
  head caps at 180).
* One moderate dropout, after the first fully connected layer of the
  regression head and its activation, instead of dropout stacked on four
  consecutive FC layers.
* ~0.3M parameters (PilotNet: ~1.6M), small enough to ship to the browser
  as fp32 ONNX.
* Optional history frames: ``in_frames`` RGB frames stacked along channels
  (current frame first), so the first conv sees ``3 * in_frames`` channels.
  Frame-to-frame image motion shows the car's own rotation, which a single
  frame cannot. Each extra frame adds 1,800 weights to the first conv
  (287,489 parameters with one frame, 289,289 with two).

The conv stack keeps VALID padding and the 66x200 input so results are
directly comparable with PilotNet, and the final feature map keeps its
spatial width (flatten, not global pooling): steering is a left/right
decision, and pooling away horizontal position discards the signal.
"""

from __future__ import annotations

import torch
from torch import nn

from sightline.models.adapter import InputAdapter

__all__ = ["SightlineNet"]


def _block(cin: int, cout: int, k: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=stride, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


class SightlineNet(nn.Module):
    FEAT_C, FEAT_H, FEAT_W = 64, 1, 18

    def __init__(self, in_frames: int = 1, dropout: float = 0.1):
        super().__init__()
        if in_frames < 1:
            raise ValueError(f"in_frames must be at least 1, got {in_frames}")
        self.in_frames = in_frames
        self.adapter = InputAdapter()
        self.features = nn.Sequential(
            _block(3 * in_frames, 24, 5, 2),
            _block(24, 36, 5, 2),
            _block(36, 48, 5, 2),
            _block(48, 64, 3, 1),
            _block(64, 64, 3, 1),
        )
        flat = self.FEAT_C * self.FEAT_H * self.FEAT_W
        self.regressor = nn.Sequential(
            nn.Linear(flat, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
        )
        #: fixed output scale: the head learns angle/SCALE, keeping the
        #: regression target near unit variance (train std is ~31 deg)
        self.OUTPUT_SCALE = 30.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """uint8 NHWC RGB frame stacks -> steering angle in degrees."""
        x = self.adapter(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.regressor(x) * self.OUTPUT_SCALE
