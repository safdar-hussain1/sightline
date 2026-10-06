"""VisualBackProp saliency — where the network looks when it steers.

Implements the VisualBackProp scheme (Bojarski et al. 2016): average each
conv layer's post-activation feature maps over channels, then walk from the
deepest layer back to the input, upsampling and pointwise-multiplying, so
only regions that stay active through the WHOLE stack survive. The result is
a 66x200 relevance map over the model's input; when the input stacks history
frames, the map covers the pixel grid they share.

The map is computed with plain tensor ops (no gradients, no hooks), so
``SteeringWithSaliency`` exports to ONNX as a single graph with two outputs
— the browser cockpit gets the model's sightline from the same forward pass
that produces the steering angle.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sightline.data.preprocess import INPUT_HEIGHT, INPUT_WIDTH
from sightline.models.pilotnet import PilotNet
from sightline.models.sightlinenet import SightlineNet

__all__ = ["conv_activations", "visual_backprop", "SteeringWithSaliency"]


def conv_activations(model: nn.Module, adapted: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the conv stack, returning (final features, per-layer activations)."""
    acts: list[torch.Tensor] = []
    x = adapted
    if isinstance(model, PilotNet):
        for conv in model.conv:
            x = torch.relu(conv(x))
            acts.append(x)
    elif isinstance(model, SightlineNet):
        for block in model.features:
            x = block(x)
            acts.append(x)
    else:
        raise TypeError(f"no activation route for {type(model).__name__}")
    return x, acts


def visual_backprop(acts: list[torch.Tensor]) -> torch.Tensor:
    """Backpropagate channel-mean relevance through the activation pyramid.

    Returns (N, INPUT_HEIGHT, INPUT_WIDTH) maps normalized to [0, 1] per
    sample.
    """
    if not acts:
        raise ValueError("no activations given")
    relevance = acts[-1].mean(dim=1, keepdim=True)
    for act in reversed(acts[:-1]):
        mean_map = act.mean(dim=1, keepdim=True)
        relevance = F.interpolate(
            relevance, size=mean_map.shape[2:], mode="bilinear", align_corners=False
        )
        relevance = relevance * mean_map
    relevance = F.interpolate(
        relevance, size=(INPUT_HEIGHT, INPUT_WIDTH), mode="bilinear", align_corners=False
    )
    flat = relevance.flatten(1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    normed = (flat - lo) / (hi - lo + 1e-12)
    return normed.reshape(-1, INPUT_HEIGHT, INPUT_WIDTH)


class SteeringWithSaliency(nn.Module):
    """Wraps a steering model into a (angle, saliency) two-output graph."""

    def __init__(self, model: PilotNet | SightlineNet):
        super().__init__()
        self.model = model

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adapted = self.model.adapter(frames)
        feat, acts = conv_activations(self.model, adapted)
        flat = torch.flatten(feat, 1)
        if isinstance(self.model, PilotNet):
            x = flat
            for fc in self.model.fc:
                x = self.model.drop(torch.relu(fc(x)))
            radians = 2.0 * torch.atan(self.model.head(x))
            angle = radians * (180.0 / torch.pi)
        else:
            angle = self.model.regressor(flat) * self.model.OUTPUT_SCALE
        return angle, visual_backprop(acts)
