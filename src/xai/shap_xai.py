from __future__ import annotations

import numpy as np
import shap
from PIL import Image

from ..core.types import InputSample, ExplanationOutput
from ..models.base import BaseModelWrapper
from .base import BaseXAIWrapper
from .render import build_image_artifacts


class ShapXAI(BaseXAIWrapper):
    name = "SHAP"
    compatible_input_types = {"image", "audio"}

    def explain(self, model: BaseModelWrapper, sample: InputSample, target_class: int) -> ExplanationOutput:
        base_img = _sample_to_array(sample, model.input_type)

        def predict_fn(batch):
            if model.input_type == "audio" and batch.ndim == 4:
                batch = batch.mean(axis=3)
            batch_tensor = model.tensor_from_numpy(batch)
            logits = model.model_forward(batch_tensor)
            probs = logits.softmax(dim=1).detach().cpu().numpy()
            return probs

        background = np.zeros_like(base_img)[None, ...]
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(base_img[None, ...], nsamples=50)
        # shap_values is list[class][1,H,W,C]
        target_vals = shap_values[target_class][0]
        if target_vals.ndim == 3:
            heatmap = np.abs(target_vals).mean(axis=2)
        else:
            heatmap = np.abs(target_vals)
        heatmap = heatmap / (heatmap.max() + 1e-6)

        base_img_pil = _sample_to_image(sample)
        artifacts = {"heatmap": heatmap, "shap_values": shap_values}
        renderable = build_image_artifacts(base_img_pil, heatmap)
        return ExplanationOutput(self.name, artifacts, renderable)


def _sample_to_array(sample: InputSample, input_type: str) -> np.ndarray:
    if input_type == "image":
        return np.array(sample.metadata.get("original"))
    mel_img = sample.metadata.get("mel_image")
    if isinstance(mel_img, np.ndarray):
        return np.stack([mel_img] * 3, axis=2)
    return np.array(mel_img.convert("RGB"))


def _sample_to_image(sample: InputSample) -> Image.Image:
    if sample.input_type == "image":
        return sample.metadata.get("original")
    mel_img = sample.metadata.get("mel_image")
    if isinstance(mel_img, np.ndarray):
        mel_img = Image.fromarray(mel_img).convert("RGB")
    return mel_img
