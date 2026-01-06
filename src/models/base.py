from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch

from ..core.types import InputSample, PredictionOutput


class BaseModelWrapper(ABC):
    name: str
    input_type: str
    labels: List[str]

    @abstractmethod
    def preprocess(self, file_path: str) -> InputSample:
        raise NotImplementedError

    @abstractmethod
    def predict(self, sample: InputSample) -> PredictionOutput:
        raise NotImplementedError

    def get_explain_target(self, sample: InputSample, pred: PredictionOutput) -> int:
        return self.labels.index(pred.label)

    def get_last_conv_layer(self):
        return None

    def tensor_from_numpy(self, arr: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    def model_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
