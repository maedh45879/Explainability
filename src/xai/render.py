from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image

from ..utils.viz import overlay_heatmap


def build_image_artifacts(base: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Dict[str, Image.Image]:
    overlay = overlay_heatmap(base, heatmap, alpha=alpha)
    heat_img = Image.fromarray((heatmap * 255).clip(0, 255).astype(np.uint8))
    heat_img = heat_img.resize(base.size).convert("RGB")
    return {
        "original": base,
        "heatmap": heat_img,
        "overlay": overlay,
    }
