from __future__ import annotations

from typing import List

from .types import InputSample, PredictionOutput, ExplanationOutput
from ..models.base import BaseModelWrapper
from ..xai.base import BaseXAIWrapper


def run_prediction(model: BaseModelWrapper, sample: InputSample) -> PredictionOutput:
    return model.predict(sample)


def run_explanations(
    model: BaseModelWrapper,
    sample: InputSample,
    pred: PredictionOutput,
    xai_methods: List[BaseXAIWrapper],
) -> List[ExplanationOutput]:
    target = model.get_explain_target(sample, pred)
    outputs = []
    for method in xai_methods:
        outputs.append(method.explain(model, sample, target))
    return outputs
