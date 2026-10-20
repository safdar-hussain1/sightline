"""The model registry: each named model is an architecture plus its training recipe.

* ``pilotnet`` -- the classic single-frame PilotNet, with its ``2*atan``
  head that caps the output at +/-180 deg, trained with the standard
  augmentation recipe.
* ``sightlinenet`` -- SightlineNet on the current frame plus the history
  frames in ``SHIPPED_LAGS``, trained with the strong recipe.
* ``sightlinenet-1f`` -- the same network and recipe on the current frame
  alone: the single-frame baseline that isolates what history frames add.

Frame lags count frames back from the current one: a lag of 2 is the frame
two frames before it. A lag is a number of frames, not a duration, as the
time between frames varies along a drive; the site data records the measured
spread of that time for each lag of the shipped model (``timing`` in
``web/public/data/results.json``, written by ``sightline site-data``).
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from sightline.models.pilotnet import PilotNet
from sightline.models.sightlinenet import SightlineNet

__all__ = ["PilotNet", "SightlineNet", "ModelSpec", "MODEL_SPECS", "SHIPPED_LAGS", "build_model"]

#: history frames the shipped model sees besides the current one, chosen on
#: validation data (never on test data)
SHIPPED_LAGS: tuple[int, ...] = (2,)


@dataclass(frozen=True)
class ModelSpec:
    family: str
    frame_lags: tuple[int, ...]
    augment: str
    dropout: float


MODEL_SPECS: dict[str, ModelSpec] = {
    "pilotnet": ModelSpec("pilotnet", (), "standard", 0.2),
    "sightlinenet": ModelSpec("sightlinenet", SHIPPED_LAGS, "strong", 0.3),
    "sightlinenet-1f": ModelSpec("sightlinenet", (), "strong", 0.3),
}


def build_model(
    name: str,
    frame_lags: tuple[int, ...] | None = None,
    dropout: float | None = None,
) -> nn.Module:
    """Build a registered model; ``frame_lags``/``dropout`` override its spec.

    The returned module carries ``frame_lags``, which every consumer uses to
    stack the input frames the model expects.
    """
    if name not in MODEL_SPECS:
        raise ValueError(f"unknown model {name!r}; expected one of {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[name]
    lags = tuple(spec.frame_lags if frame_lags is None else frame_lags)
    drop = spec.dropout if dropout is None else dropout
    if spec.family == "pilotnet":
        if lags:
            raise ValueError("PilotNet is a single-frame model; it takes no history frames")
        model = PilotNet(dropout=drop)
    else:
        model = SightlineNet(in_frames=1 + len(lags), dropout=drop)
    model.frame_lags = lags
    return model
