from __future__ import annotations

from functools import lru_cache
from typing import List

from ..models.audio_models import AudioCNNWrapper
from ..models.image_models import build_alexnet, build_densenet
from ..xai.gradcam_xai import GradCAMXAI
from ..xai.lime_xai import LimeXAI
from ..xai.shap_xai import ShapXAI
from ..xai.extra_xai import IntegratedGradientsXAI


@lru_cache(maxsize=1)
def list_models():
    return [
        AudioCNNWrapper(),
        build_alexnet(),
        build_densenet(),
    ]


@lru_cache(maxsize=1)
def list_xai():
    return [
        GradCAMXAI(),
        LimeXAI(),
        ShapXAI(),
        IntegratedGradientsXAI(),
    ]


def get_models_for_input_type(input_type: str):
    return [m for m in list_models() if m.input_type == input_type]


def get_xai_for(input_type: str, model_name: str):
    candidates = []
    for xai in list_xai():
        if input_type not in xai.compatible_input_types:
            continue
        if xai.compatible_model_names and model_name not in xai.compatible_model_names:
            continue
        candidates.append(xai)
    return candidates
