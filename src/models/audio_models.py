from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ..core.types import InputSample, PredictionOutput
from ..core.preprocess_audio import prepare_audio_tensor, mel_to_image
from ..utils.logging import get_logger
from ..utils.errors import UserFacingError
from .base import BaseModelWrapper

LOGGER = get_logger(__name__)
_DEFAULT_DEEPFAKE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "Deepfake-Audio-Detection-with-XAI",
    "Streamlit",
    "saved_model",
    "model",
)


class SimpleAudioCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class AudioCNNWrapper(BaseModelWrapper):
    name = "AudioCNN"
    input_type = "audio"
    labels = ["real", "fake"]

    def __init__(self):
        self.model = SimpleAudioCNN(num_classes=len(self.labels))
        self.model.eval()

    def preprocess(self, file_path: str) -> InputSample:
        audio_data = prepare_audio_tensor(file_path)
        mel = audio_data["mel"]
        mel = mel.unsqueeze(0)  # [1, 1, n_mels, time]
        # Resize to fixed size for the simple CNN
        mel = torch.nn.functional.interpolate(mel, size=(64, 64), mode="bilinear")
        metadata = {
            "sr": audio_data["sr"],
            "warning": "Using randomly initialized weights. Replace with trained weights for real predictions.",
            "mel_image": mel_to_image(audio_data["mel"]),
            "waveform": audio_data["waveform"].cpu().numpy(),
        }
        return InputSample(
            input_type="audio",
            path=file_path,
            raw=audio_data["waveform"].cpu().numpy(),
            processed=mel,
            metadata=metadata,
        )

    def predict(self, sample: InputSample) -> PredictionOutput:
        logits = self.model_forward(sample.processed)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        probs_dict = {label: float(probs[idx]) for idx, label in enumerate(self.labels)}
        top_idx = int(torch.argmax(probs).item())
        return PredictionOutput(label=self.labels[top_idx], probs=probs_dict, top1=float(probs[top_idx]))

    def tensor_from_numpy(self, arr: np.ndarray) -> torch.Tensor:
        if arr.ndim == 2:
            arr = arr[None, None, :, :]
        elif arr.ndim == 3:
            arr = arr[:, None, :, :]
        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0)
        tensor = torch.nn.functional.interpolate(tensor, size=(64, 64), mode="bilinear")
        mean = tensor.mean(dim=(2, 3), keepdim=True)
        std = tensor.std(dim=(2, 3), keepdim=True) + 1e-6
        return (tensor - mean) / std

    def model_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(tensor)


class DeepfakeSavedModelWrapper(BaseModelWrapper):
    name = "DeepfakeSavedModel"
    input_type = "audio"
    labels = ["real", "fake"]

    def __init__(self, model_path: Optional[str] = None):
        try:
            import tensorflow as tf
        except Exception as exc:
            raise RuntimeError("TensorFlow is required to load the Deepfake saved model.") from exc
        self._tf = tf
        path = model_path or _DEFAULT_DEEPFAKE_MODEL_PATH
        path = os.path.normpath(path)
        if not os.path.isdir(path):
            raise UserFacingError(
                "Deepfake saved model not found. Expected at "
                f"{path}. Verify the source repo path."
            )
        self.model = tf.keras.models.load_model(path)
        self.model.trainable = False
        self._model_path = path

    def preprocess(self, file_path: str) -> InputSample:
        audio_data = prepare_audio_tensor(file_path)
        mel = audio_data["mel"]
        mel_img = mel_to_image(mel)
        mel_pil = Image.fromarray(mel_img).resize((224, 224)).convert("RGB")
        arr = np.asarray(mel_pil).astype(np.float32) / 255.0
        tf_input = arr[None, ...]
        tensor = self.tensor_from_numpy(arr)
        metadata = {
            "sr": audio_data["sr"],
            "warning": f"Loaded Deepfake saved model from {self._model_path}.",
            "mel_image": mel_pil,
            "waveform": audio_data["waveform"].cpu().numpy(),
            "tf_input": tf_input,
        }
        return InputSample(
            input_type="audio",
            path=file_path,
            raw=audio_data["waveform"].cpu().numpy(),
            processed=tensor,
            metadata=metadata,
        )

    def predict(self, sample: InputSample) -> PredictionOutput:
        tf_input = sample.metadata.get("tf_input")
        if tf_input is None:
            tf_input = self._tensor_to_numpy(sample.processed)
        preds = self.model(tf_input, training=False).numpy()
        probs = preds.squeeze(0)
        probs_dict = {label: float(probs[idx]) for idx, label in enumerate(self.labels)}
        top_idx = int(np.argmax(probs))
        return PredictionOutput(label=self.labels[top_idx], probs=probs_dict, top1=float(probs[top_idx]))

    def tensor_from_numpy(self, arr: np.ndarray) -> torch.Tensor:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :, None]
        elif arr.ndim == 3:
            if arr.shape[-1] in (1, 3):
                arr = arr[None, :, :, :]
            else:
                arr = arr[:, :, :, None]
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return torch.from_numpy(arr)

    def model_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        arr = self._tensor_to_numpy(tensor)
        preds = self.model(arr, training=False).numpy()
        return torch.from_numpy(preds)

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 3:
            arr = arr[None, :, :, :]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr
