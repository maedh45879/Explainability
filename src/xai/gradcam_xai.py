from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..core.types import InputSample, ExplanationOutput
from ..models.base import BaseModelWrapper
from ..utils.viz import normalize_to_uint8
from .base import BaseXAIWrapper
from .render import build_image_artifacts


class GradCAMXAI(BaseXAIWrapper):
    name = "Grad-CAM"
    compatible_input_types = {"image"}

    def explain(self, model: BaseModelWrapper, sample: InputSample, target_class: int) -> ExplanationOutput:
        layer = model.get_last_conv_layer()
        if layer is None:
            return ExplanationOutput(self.name, {"error": "Grad-CAM not supported for this model."}, {})

        activations = []
        gradients = []

        def forward_hook(_, __, output):
            activations.append(output)

        def backward_hook(_, grad_in, grad_out):
            gradients.append(grad_out[0])

        handle_fwd = layer.register_forward_hook(forward_hook)
        handle_bwd = layer.register_full_backward_hook(backward_hook)

        tensor = sample.processed.clone().requires_grad_(True)
        tensor = tensor.to(dtype=torch.float32)
        try:
            logits = model.model_forward(tensor, use_grad=True)
        except TypeError:
            logits = model.model_forward(tensor)
        score = logits[:, target_class].sum()
        model.model.zero_grad()
        score.backward()

        handle_fwd.remove()
        handle_bwd.remove()

        act = activations[0].detach()
        grad = gradients[0].detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam_np = cam.cpu().numpy()
        cam_np = cam_np / (cam_np.max() + 1e-6)

        base = sample.metadata.get("original")
        artifacts = {"heatmap": cam_np}
        renderable = build_image_artifacts(base, cam_np)
        return ExplanationOutput(self.name, artifacts, renderable)
