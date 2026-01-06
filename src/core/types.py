from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Any

import numpy as np
import torch
from PIL import Image

InputType = Literal["audio", "image"]


@dataclass
class InputSample:
    input_type: InputType
    path: Optional[str]
    raw: Any  # np.ndarray or PIL.Image
    processed: torch.Tensor
    metadata: Dict[str, Any]


@dataclass
class PredictionOutput:
    label: str
    probs: Dict[str, float]
    top1: float


@dataclass
class ExplanationOutput:
    method_name: str
    artifacts: Dict[str, Any]
    renderable: Dict[str, Any]
