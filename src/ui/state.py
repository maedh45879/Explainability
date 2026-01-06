from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.types import InputSample, PredictionOutput


@dataclass
class AppState:
    sample: Optional[InputSample] = None
    prediction: Optional[PredictionOutput] = None
