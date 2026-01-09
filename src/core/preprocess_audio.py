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
    try:
        data, sr = sf.read(path, always_2d=True)
    except Exception as exc:
        if "Format not recognised" not in str(exc):
            raise
        raise RuntimeError("soundfile_format_not_recognised") from exc
    data = _normalize_audio_array(data)
    waveform = torch.from_numpy(data.T).to(dtype=torch.float32)
    waveform = _to_mono_if_needed(waveform)
    return waveform, sr


def _normalize_audio_array(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32, copy=False)
    if data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    return data.astype(np.float32)


def _to_mono_if_needed(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.shape[0] > 1:
        return waveform.mean(dim=0, keepdim=True)
    return waveform


def _load_with_scipy(path: str) -> Tuple[torch.Tensor, int]:
    try:
        from scipy.io import wavfile
    except Exception as exc:
        raise RuntimeError(
            "scipy is required to load audio when soundfile cannot decode it."
        ) from exc
    sr, data = wavfile.read(path)
    if data.ndim == 1:
        data = data[:, None]
    data = _normalize_audio_array(data)
    waveform = torch.from_numpy(data.T).to(dtype=torch.float32)
    waveform = _to_mono_if_needed(waveform)
    return waveform, sr


def _raise_unsupported_audio() -> None:
    raise ValueError(
        "Unsupported audio encoding. Please export as WAV PCM (16-bit) or FLAC."
    )


def load_audio(path: str, target_sr: int = DEFAULT_SR) -> Tuple[torch.Tensor, int]:
    if torchaudio is None:
        try:
            waveform, sr = _load_with_soundfile(path)
        except RuntimeError as exc:
            if "soundfile_format_not_recognised" in str(exc):
                try:
                    waveform, sr = _load_with_scipy(path)
                except Exception:
                    _raise_unsupported_audio()
            else:
                raise
    else:
        try:
            waveform, sr = torchaudio.load(path, backend="soundfile")
        except Exception as exc:
            message = str(exc).lower()
            if "torchcodec" in message or "libtorchcodec" in message or "ffmpeg" in message:
                try:
                    waveform, sr = _load_with_soundfile(path)
                except RuntimeError as sf_exc:
                    if "soundfile_format_not_recognised" in str(sf_exc):
                        try:
                            waveform, sr = _load_with_scipy(path)
                        except Exception:
                            _raise_unsupported_audio()
                    else:
                        raise
            else:
                try:
                    waveform, sr = torchaudio.load(path)
                except Exception:
                    try:
                        waveform, sr = _load_with_soundfile(path)
                    except RuntimeError as sf_exc:
                        if "soundfile_format_not_recognised" in str(sf_exc):
                            try:
                                waveform, sr = _load_with_scipy(path)
                            except Exception:
                                _raise_unsupported_audio()
                        else:
                            raise
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
