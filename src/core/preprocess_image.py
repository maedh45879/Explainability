from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image(path: str, size: int = 224) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((size, size))


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    tensor = tensor * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    tensor = (tensor * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(tensor)
