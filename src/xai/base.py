from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Set

from ..core.types import InputSample, ExplanationOutput
from ..models.base import BaseModelWrapper


class BaseXAIWrapper(ABC):
    name: str
    compatible_input_types: Set[str]
    compatible_model_names: Optional[Set[str]] = None

    @abstractmethod
    def explain(self, model: BaseModelWrapper, sample: InputSample, target_class: int) -> ExplanationOutput:
        raise NotImplementedError
