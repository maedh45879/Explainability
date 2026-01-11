from __future__ import annotations

from typing import List
from types import MethodType

import numpy as np
import torch
import torchvision
import torch.nn.functional as F

from ..core.preprocess_image import IMAGENET_MEAN, IMAGENET_STD

from ..core.types import InputSample, PredictionOutput
from ..core.preprocess_image import load_image, image_to_tensor
from ..utils.logging import get_logger
from .base import BaseModelWrapper

LOGGER = get_logger(__name__)


class TorchVisionWrapper(BaseModelWrapper):
    input_type = "image"
    labels = ["benign", "malignant"]

    def __init__(self, name: str, model: torch.nn.Module, last_conv):
        self.name = name
        self.model = model
        self.model.eval()
        self._last_conv = last_conv

    def preprocess(self, file_path: str) -> InputSample:
        image = load_image(file_path)
        tensor = image_to_tensor(image)
        metadata = {
            "warning": "Using randomly initialized weights. Replace with trained weights for real predictions.",
            "original": image,
        }
        return InputSample(
            input_type="image",
            path=file_path,
            raw=image,
            processed=tensor,
            metadata=metadata,
        )

    def predict(self, sample: InputSample) -> PredictionOutput:
        logits = self.model_forward(sample.processed)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        probs_dict = {label: float(probs[idx]) for idx, label in enumerate(self.labels)}
        top_idx = int(torch.argmax(probs).item())
        return PredictionOutput(label=self.labels[top_idx], probs=probs_dict, top1=float(probs[top_idx]))

    def tensor_from_numpy(self, arr: np.ndarray) -> torch.Tensor:
        if arr.ndim == 3:
            arr = arr[None, :, :, :]
        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0)
        tensor = tensor.permute(0, 3, 1, 2)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor

    def model_forward(self, tensor: torch.Tensor, use_grad: bool = False) -> torch.Tensor:
        if use_grad:
            return self.model(tensor.float())
        with torch.no_grad():
            return self.model(tensor.float())

    def get_last_conv_layer(self):
        return self._last_conv


def build_alexnet() -> TorchVisionWrapper:
    model = torchvision.models.alexnet(weights=None)
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
    last_conv = model.features[10]
    return TorchVisionWrapper("AlexNet", model, last_conv)


def build_densenet() -> TorchVisionWrapper:
    model = torchvision.models.densenet121(weights=None)
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
    def _forward_no_inplace_relu(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    model.forward = MethodType(_forward_no_inplace_relu, model)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    last_conv = model.features[-1]
    return TorchVisionWrapper("DenseNet", model, last_conv)
