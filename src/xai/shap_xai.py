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
        base_img = _sample_to_array(sample, model.input_type).astype(np.float32)
        base_shape = base_img.shape

        def predict_fn(batch_flat):
            batch_flat = np.asarray(batch_flat, dtype=np.float32)
            if batch_flat.ndim == 1:
                batch_flat = batch_flat[None, :]
            batch = batch_flat.reshape((-1,) + base_shape)
            if model.input_type == "audio" and batch.ndim == 4:
                batch = batch.mean(axis=3)
            batch_tensor = model.tensor_from_numpy(batch)
            logits = model.model_forward(batch_tensor)
            probs = logits.softmax(dim=1).detach().cpu().numpy()
            return probs

        X = base_img.reshape(-1)[None, :]
        background = np.zeros_like(X)
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X, nsamples=50)
        D = int(np.prod(base_shape))
        if isinstance(shap_values, list):
            safe_idx = min(target_class, len(shap_values) - 1)
            sv = np.asarray(shap_values[safe_idx])
        else:
            sv = np.asarray(shap_values)
        if sv.ndim >= 2:
            vals = sv[0].reshape(-1)
        else:
            vals = sv.reshape(-1)
        if vals.size != D:
            if vals.size % D == 0:
                C = vals.size // D
                c = target_class if target_class < C else 0
                vals = vals[c * D : (c + 1) * D]
            if vals.size != D:
                raise ValueError(
                    "SHAP output size does not match input size. "
                    f"got={vals.size}, expected={D}, shap_shape={sv.shape}."
                )
        target_vals = vals.astype(np.float32, copy=False).reshape(base_shape)
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
