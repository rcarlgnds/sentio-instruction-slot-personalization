import json
import os
from dataclasses import dataclass
from pathlib import Path

from app.domain.models import RetrievalResult
from app.utils.logging import get_logger

log = get_logger(__name__)

PROVISIONAL_MIN = 0.40
PROVISIONAL_MAX = 0.70


@dataclass(frozen=True)
class CalibrationBounds:
    absolute_min: float
    absolute_max: float
    n_queries: int
    model_version: str
    date: str
    provisional: bool

    @classmethod
    def load(cls, path: str) -> "CalibrationBounds":
        p = Path(path)
        if not p.exists():
            log.warning(
                f"calibration.json not found at {path}. "
                f"Using PROVISIONAL bounds [{PROVISIONAL_MIN}, {PROVISIONAL_MAX}]. "
                "Run the scripts from system/, which holds the study's calibration.json."
            )
            return cls(
                absolute_min=PROVISIONAL_MIN,
                absolute_max=PROVISIONAL_MAX,
                n_queries=0,
                model_version="unknown",
                date="",
                provisional=True,
            )

        data = json.loads(p.read_text())
        return cls(
            absolute_min=data["absolute_min"],
            absolute_max=data["absolute_max"],
            n_queries=data.get("n_queries", 0),
            model_version=data.get("model_version", "unknown"),
            date=data.get("date", ""),
            provisional=data.get("provisional", False),
        )


class CalibrationProvider:
    def __init__(self, path: str, on_change=None):
        self.path = path
        self._on_change = on_change
        self._bounds: CalibrationBounds | None = None
        self._mtime: float | None = None

    def _current_mtime(self) -> float | None:
        try:
            return os.path.getmtime(self.path)
        except OSError:
            return None

    def get(self) -> CalibrationBounds:
        mtime = self._current_mtime()
        if self._bounds is None or mtime != self._mtime:
            self._bounds = CalibrationBounds.load(self.path)
            self._mtime = mtime
            log.info(
                f"[calibration] active bounds [{self._bounds.absolute_min}, "
                f"{self._bounds.absolute_max}] provisional={self._bounds.provisional}"
            )
            if self._on_change:
                self._on_change()
        return self._bounds


def normalize_sims(results: list[RetrievalResult], bounds: CalibrationBounds) -> bool:
    if not results:
        return False

    if max(r.sim_score for r in results) < bounds.absolute_min:
        for r in results:
            r.sim_norm = 0.0
        return False

    span = bounds.absolute_max - bounds.absolute_min
    if span <= 0:
        raise ValueError(f"Invalid calibration bounds: span={span}")

    for r in results:
        clipped = min(bounds.absolute_max, max(bounds.absolute_min, r.sim_score))
        r.sim_norm = (clipped - bounds.absolute_min) / span

    return True
