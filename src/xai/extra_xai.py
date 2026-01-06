from __future__ import annotations

import numpy as np
import torch

from ..core.types import InputSample, ExplanationOutput
from ..models.base import BaseModelWrapper
from .base import BaseXAIWrapper
from .render import build_image_artifacts


class IntegratedGradientsXAI(BaseXAIWrapper):
    name = "Integrated Gradients"
    compatible_input_types = {"image"}

    def explain(self, model: BaseModelWrapper, sample: InputSample, target_class: int) -> ExplanationOutput:
        steps = 20
        x = sample.processed.to(dtype=torch.float32)
        baseline = torch.zeros_like(x)
        scaled = [baseline + (float(i) / steps) * (x - baseline) for i in range(1, steps + 1)]
        grads = []
        for tensor in scaled:
            tensor = tensor.clone().detach().requires_grad_(True)
            try:
                logits = model.model_forward(tensor, use_grad=True)
            except TypeError:
                logits = model.model_forward(tensor)
            score = logits[:, target_class].sum()
            model.model.zero_grad()
            score.backward()
            grads.append(tensor.grad.detach().clone())
        avg_grads = torch.stack(grads, dim=0).mean(dim=0)
        ig = (x - baseline) * avg_grads
        heatmap = ig.squeeze(0).abs().mean(dim=0).cpu().numpy()
        heatmap = heatmap / (heatmap.max() + 1e-6)

        base = sample.metadata.get("original")
        artifacts = {"heatmap": heatmap}
        renderable = build_image_artifacts(base, heatmap)
        return ExplanationOutput(self.name, artifacts, renderable)
