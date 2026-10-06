"""Where the data and the runs live, and the errors a missing or unusable one raises."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["Workspace", "MissingDataError", "UnusableRunError"]


class MissingDataError(FileNotFoundError):
    """A drive or run the command needs is not on disk."""


class UnusableRunError(ValueError):
    """A run is on disk but its record lacks what loading its model needs."""


@dataclass(frozen=True)
class Workspace:
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")

    def drive(self, name: str, verify_hint: bool = True) -> Path:
        """The drive's folder; ``verify_hint`` adds the pointer to ``sightline data verify``
        (off when that command is the one asking)."""
        path = Path(self.data_dir) / name
        if not (path / "data.txt").is_file():
            hint = " and check it with `sightline data verify`." if verify_hint else "."
            raise MissingDataError(
                f"drive {name!r} not found: expected {path / 'data.txt'}. "
                f"Download it (README, 'Get the data'){hint}"
            )
        return path

    def cache_dir(self, drive_name: str) -> Path:
        return Path(self.runs_dir) / "cache" / drive_name

    def run_dir(self, run_name: str) -> Path:
        path = Path(self.runs_dir) / run_name
        if not (path / "history.json").is_file():
            raise MissingDataError(f"run {run_name!r} not found: expected {path / 'history.json'}")
        return path
