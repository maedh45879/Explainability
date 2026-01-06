from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.types import InputSample, PredictionOutput
from ..core.preprocess_audio import prepare_audio_tensor, mel_to_image
from ..utils.logging import get_logger
from .base import BaseModelWrapper

LOGGER = get_logger(__name__)


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
