from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from ..models.audio_models import AudioCNNWrapper, DeepfakeSavedModelWrapper
from ..models.image_models import build_alexnet, build_densenet
from ..models.base import BaseModelWrapper
from ..utils.logging import get_logger
from ..xai.gradcam_xai import GradCAMXAI
from ..xai.lime_xai import LimeXAI
from ..xai.shap_xai import ShapXAI
from ..xai.extra_xai import IntegratedGradientsXAI

LOGGER = get_logger(__name__)

@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_type: str
    factory: Callable[[], BaseModelWrapper]


_MODEL_SPECS: List[ModelSpec] = [
    ModelSpec("DeepfakeSavedModel", "audio", lambda: DeepfakeSavedModelWrapper()),
    ModelSpec("AudioCNN", "audio", AudioCNNWrapper),
    ModelSpec("AlexNet", "image", build_alexnet),
    ModelSpec("DenseNet", "image", build_densenet),
]
_MODEL_BY_NAME: Dict[str, ModelSpec] = {spec.name: spec for spec in _MODEL_SPECS}
_MODEL_CACHE: Dict[str, BaseModelWrapper] = {}


def _build_model(spec: ModelSpec) -> Optional[BaseModelWrapper]:
    try:
        return spec.factory()
    except Exception as exc:
        LOGGER.warning("Skipping model %s due to load error: %s", spec.name, exc)
        return None


def list_models() -> List[BaseModelWrapper]:
    models: List[BaseModelWrapper] = []
    for spec in _MODEL_SPECS:
        model = get_model_by_name(spec.name)
        if model is not None:
            models.append(model)
    return models


def list_xai():
    return [
        GradCAMXAI(),
        LimeXAI(),
        ShapXAI(),
        IntegratedGradientsXAI(),
    ]


def get_models_for_input_type(input_type: str):
    models: List[BaseModelWrapper] = []
    for spec in _MODEL_SPECS:
        if spec.input_type != input_type:
            continue
        model = get_model_by_name(spec.name)
        if model is not None:
            models.append(model)
    return models


def get_model_names_for_input_type(input_type: str) -> List[str]:
    return [spec.name for spec in _MODEL_SPECS if spec.input_type == input_type]


def get_model_by_name(model_name: str) -> Optional[BaseModelWrapper]:
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached
    spec = _MODEL_BY_NAME.get(model_name)
    if spec is None:
        return None
    model = _build_model(spec)
    if model is not None:
        _MODEL_CACHE[model_name] = model
    return model


def get_xai_for(input_type: str, model_name: str):
    candidates = []
    for xai in list_xai():
        if input_type not in xai.compatible_input_types:
            continue
        if xai.compatible_model_names and model_name not in xai.compatible_model_names:
            continue
        candidates.append(xai)
    return candidates
