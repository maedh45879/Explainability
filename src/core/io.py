from __future__ import annotations

import os
from typing import Tuple

from .types import InputType
from ..utils.errors import UserFacingError


AUDIO_EXTS = {".wav"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def detect_input_type(file_path: str) -> InputType:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in IMAGE_EXTS:
        return "image"
    raise UserFacingError("Unsupported file type. Please upload a .wav audio or .png/.jpg image.")
