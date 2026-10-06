"""Where the data and the runs live, and the one error a missing dataset raises."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["Workspace", "MissingDataError"]


class MissingDataError(FileNotFoundError):
    """A drive or run the command needs is not on disk."""


@dataclass(frozen=True)
class Workspace:
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")

    def drive(self, name: str) -> Path:
        path = Path(self.data_dir) / name
        if not (path / "data.txt").is_file():
            raise MissingDataError(
                f"drive {name!r} not found: expected {path / 'data.txt'}. "
                "Download it (README, 'Get the data') and check it with `sightline data verify`."
            )
        return path

    def cache_dir(self, drive_name: str) -> Path:
        return Path(self.runs_dir) / "cache" / drive_name

    def run_dir(self, run_name: str) -> Path:
        path = Path(self.runs_dir) / run_name
        if not (path / "history.json").is_file():
            raise MissingDataError(f"run {run_name!r} not found: expected {path / 'history.json'}")
        return path
