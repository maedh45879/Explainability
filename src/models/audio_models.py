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
            import keras
        except Exception as exc:
            raise RuntimeError("TensorFlow is required to load the Deepfake saved model.") from exc
        self._tf = tf
        self._keras = keras
        path = model_path or _DEFAULT_DEEPFAKE_MODEL_PATH
        path = os.path.normpath(path)
        if not os.path.isdir(path):
            raise UserFacingError(
                "Deepfake saved model not found. Expected at "
                f"{path}. Verify the source repo path."
            )
        self._model_path = path
        self._saved_model = tf.saved_model.load(path)
        signatures = getattr(self._saved_model, "signatures", {}) or {}
        self._endpoint = self._select_endpoint(signatures)
        signature = signatures.get(self._endpoint)
        self._input_key, self._input_spec = self._get_input_spec(signature)
        self._layer = keras.layers.TFSMLayer(path, call_endpoint=self._endpoint)
        self.model = self._layer
        self._model_path = path

    def preprocess(self, file_path: str) -> InputSample:
        audio_data = prepare_audio_tensor(file_path)
        mel = audio_data["mel"]
        mel_img = mel_to_image(mel)
        mel_pil, tf_input, arr = self._prepare_tf_input(mel_img)
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
        preds = self._run_model(tf_input)
        probs = self._ensure_probs(preds)
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
        preds = self._run_model(arr)
        return torch.from_numpy(preds)

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 3:
            arr = arr[None, :, :, :]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr

    def _select_endpoint(self, signatures) -> str:
        if not signatures:
            raise UserFacingError(
                f"No callable signatures found in SavedModel at {self._model_path}."
            )
        if "serving_default" in signatures:
            return "serving_default"
        if len(signatures) == 1:
            return next(iter(signatures))
        for name in signatures:
            if "serving" in name.lower():
                return name
        return next(iter(signatures))

    def _get_input_spec(self, signature):
        if signature is None:
            return None, None
        try:
            args_spec, kwargs_spec = signature.structured_input_signature
        except Exception:
            return None, None
        if kwargs_spec:
            if len(kwargs_spec) != 1:
                raise UserFacingError(
                    f"SavedModel endpoint '{self._endpoint}' expects multiple inputs."
                )
            key = next(iter(kwargs_spec))
            return key, kwargs_spec[key]
        if args_spec:
            if len(args_spec) != 1:
                raise UserFacingError(
                    f"SavedModel endpoint '{self._endpoint}' expects multiple inputs."
                )
            return None, args_spec[0]
        return None, None

    def _prepare_tf_input(self, mel_img: np.ndarray) -> tuple[Image.Image, np.ndarray, np.ndarray]:
        expected_hwc = self._expected_hwc()
        mel_pil = Image.fromarray(mel_img)
        if expected_hwc is not None:
            height, width, channels = expected_hwc
            if height and width:
                mel_pil = mel_pil.resize((width, height))
            if channels == 1:
                mel_pil = mel_pil.convert("L")
            elif channels == 3:
                mel_pil = mel_pil.convert("RGB")
        else:
            mel_pil = mel_pil.resize((224, 224)).convert("RGB")
        arr = np.asarray(mel_pil).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[:, :, None]
        tf_input = arr[None, ...]
        self._validate_tf_input(tf_input)
        return mel_pil, tf_input, arr

    def _expected_hwc(self):
        if self._input_spec is None:
            return None
        shape = self._input_spec.shape
        if shape is None:
            return None
        shape = tuple(shape)
        if len(shape) == 4:
            _, height, width, channels = shape
            return height, width, channels
        if len(shape) == 3:
            height, width, channels = shape
            return height, width, channels
        return None

    def _validate_tf_input(self, tf_input: np.ndarray) -> None:
        if self._input_spec is None:
            return
        expected = tuple(self._input_spec.shape)
        actual = tf_input.shape
        if len(expected) == 3 and len(actual) == 4 and actual[0] == 1:
            actual = actual[1:]
        elif len(expected) != len(actual):
            raise UserFacingError(
                f"SavedModel expects rank {len(expected)} input but got rank {len(actual)}."
            )
        for idx, (exp_dim, act_dim) in enumerate(zip(expected, actual)):
            if exp_dim is None:
                continue
            if exp_dim != act_dim:
                raise UserFacingError(
                    f"SavedModel input shape mismatch at dim {idx}: expected {expected}, got {actual}."
                )

    def _run_model(self, tf_input: np.ndarray) -> np.ndarray:
        if self._input_key:
            outputs = self._layer({self._input_key: tf_input})
        else:
            outputs = self._layer(tf_input)
        if isinstance(outputs, dict):
            if len(outputs) == 1:
                outputs = next(iter(outputs.values()))
            else:
                for key in ("probabilities", "outputs", "output_0"):
                    if key in outputs:
                        outputs = outputs[key]
                        break
                if isinstance(outputs, dict):
                    outputs = next(iter(outputs.values()))
        if hasattr(outputs, "numpy"):
            return outputs.numpy()
        return np.asarray(outputs)

    def _ensure_probs(self, preds: np.ndarray) -> np.ndarray:
        probs = np.asarray(preds).squeeze()
        if probs.ndim == 0:
            raise UserFacingError("SavedModel output has unexpected scalar shape.")
        if probs.shape[-1] == 1:
            val = float(probs.ravel()[0])
            probs = np.array([1.0 - val, val], dtype=np.float32)
        if probs.shape[-1] != len(self.labels):
            raise UserFacingError(
                f"SavedModel output shape {probs.shape} does not match labels {self.labels}."
            )
        if (probs < 0).any() or (probs > 1).any() or abs(float(probs.sum()) - 1.0) > 1e-3:
            exp = np.exp(probs - np.max(probs))
            probs = exp / np.sum(exp)
        return probs
