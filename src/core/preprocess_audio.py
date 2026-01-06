from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

try:
    import torchaudio
except Exception:  # pragma: no cover - handled with friendly message
    torchaudio = None


DEFAULT_SR = 16000


def _load_with_soundfile(path: str) -> Tuple[torch.Tensor, int]:
    try:
        import soundfile as sf
    except Exception as exc:
        raise RuntimeError(
            "soundfile is required to load audio when torchaudio backend is unavailable."
        ) from exc
    data, sr = sf.read(path, always_2d=True)
    waveform = torch.from_numpy(data.T).to(dtype=torch.float32)
    return waveform, sr


def load_audio(path: str, target_sr: int = DEFAULT_SR) -> Tuple[torch.Tensor, int]:
    if torchaudio is None:
        waveform, sr = _load_with_soundfile(path)
    else:
        try:
            waveform, sr = torchaudio.load(path)
        except ImportError as exc:
            if "torchcodec" not in str(exc).lower():
                raise
            try:
                waveform, sr = torchaudio.load(path, backend="soundfile")
            except Exception:
                waveform, sr = _load_with_soundfile(path)
    waveform = waveform.to(dtype=torch.float32)
    if sr != target_sr:
        if torchaudio is None:
            raise RuntimeError("torchaudio is required to resample audio.")
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
