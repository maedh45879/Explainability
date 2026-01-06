from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr -= arr.min() if arr.size else 0.0
    denom = arr.max() if arr.size else 1.0
    denom = denom if denom > 0 else 1.0
    arr = arr / denom
    return (arr * 255).clip(0, 255).astype(np.uint8)


def apply_colormap(gray: np.ndarray) -> Image.Image:
    gray_img = Image.fromarray(normalize_to_uint8(gray))
    return gray_img.convert("RGB")


def overlay_heatmap(base: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    base_rgb = base.convert("RGB")
    heat_img = apply_colormap(heatmap).resize(base_rgb.size)
    return Image.blend(base_rgb, heat_img, alpha)


def crop_center(img: Image.Image, center_x: float, center_y: float, size_ratio: float) -> Image.Image:
    w, h = img.size
    size_ratio = max(0.1, min(1.0, size_ratio))
    crop_w = int(w * size_ratio)
    crop_h = int(h * size_ratio)
    cx = int(w * center_x)
    cy = int(h * center_y)
    left = max(0, cx - crop_w // 2)
    upper = max(0, cy - crop_h // 2)
    right = min(w, left + crop_w)
    lower = min(h, upper + crop_h)
    return img.crop((left, upper, right, lower))
