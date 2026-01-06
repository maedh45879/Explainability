from __future__ import annotations

import numpy as np
from PIL import Image

from lime import lime_image

from ..core.types import InputSample, ExplanationOutput
from ..models.base import BaseModelWrapper
from .base import BaseXAIWrapper
from .render import build_image_artifacts


class LimeXAI(BaseXAIWrapper):
    name = "LIME"
    compatible_input_types = {"image", "audio"}

    def explain(self, model: BaseModelWrapper, sample: InputSample, target_class: int) -> ExplanationOutput:
        base_img = _sample_to_image(sample)
        explainer = lime_image.LimeImageExplainer()

        def predict_fn(images):
            processed = []
            for img in images:
                if model.input_type == "audio" and img.ndim == 3:
                    img = img.mean(axis=2)
                processed.append(img)
            batch = np.stack(processed, axis=0)
            batch_tensor = model.tensor_from_numpy(batch)
            logits = model.model_forward(batch_tensor)
            probs = logits.softmax(dim=1).detach().cpu().numpy()
            return probs

        explanation = explainer.explain_instance(
            np.array(base_img),
            predict_fn,
            labels=[target_class],
            num_samples=200,
        )
        image, mask = explanation.get_image_and_mask(
            target_class,
            positive_only=True,
            num_features=5,
            hide_rest=False,
        )
        heatmap = mask.astype(np.float32)
        artifacts = {"heatmap": heatmap, "lime_image": image}
        renderable = build_image_artifacts(base_img, heatmap)
        return ExplanationOutput(self.name, artifacts, renderable)


def _sample_to_image(sample: InputSample) -> Image.Image:
    if sample.input_type == "image":
        return sample.metadata.get("original")
    mel_img = sample.metadata.get("mel_image")
    if isinstance(mel_img, np.ndarray):
        mel_img = Image.fromarray(mel_img).convert("RGB")
    return mel_img.convert("RGB")
