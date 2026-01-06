from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

try:
    import torchaudio
except Exception:  # pragma: no cover - handled with friendly message
    torchaudio = None


DEFAULT_SR = 16000


def load_audio(path: str, target_sr: int = DEFAULT_SR) -> Tuple[torch.Tensor, int]:
    if torchaudio is None:
        raise RuntimeError("torchaudio is required for audio preprocessing.")
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    return waveform, sr


def audio_to_mel_spectrogram(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    if torchaudio is None:
        raise RuntimeError("torchaudio is required for audio preprocessing.")
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=64)(waveform)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
    return mel_db


def prepare_audio_tensor(path: str) -> Dict[str, torch.Tensor]:
    waveform, sr = load_audio(path)
    mel = audio_to_mel_spectrogram(waveform, sr)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return {
        "waveform": waveform,
        "sr": sr,
        "mel": mel,
    }


def mel_to_image(mel: torch.Tensor) -> np.ndarray:
    mel_np = mel.squeeze(0).cpu().numpy()
    mel_np = (mel_np - mel_np.min()) / (mel_np.max() - mel_np.min() + 1e-6)
    return (mel_np * 255).astype(np.uint8)
