from sightline.models.pilotnet import PilotNet
from sightline.models.sightlinenet import SightlineNet

__all__ = ["PilotNet", "SightlineNet", "build_model"]


def build_model(name: str) -> "PilotNet | SightlineNet":
    if name == "pilotnet":
        return PilotNet()
    if name == "sightlinenet":
        return SightlineNet()
    raise ValueError(f"unknown model {name!r}; expected pilotnet or sightlinenet")
