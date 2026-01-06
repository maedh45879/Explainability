from __future__ import annotations

from typing import Dict, List

from PIL import Image


def flatten_renderables(renderables: List[Dict[str, Image.Image]]) -> List[Image.Image]:
    images = []
    for render in renderables:
        for key in ("overlay", "heatmap", "original"):
            if key in render:
                images.append(render[key])
                break
    return images
