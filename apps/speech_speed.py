"""Pitch-preserving desktop tempo adjustment with Rubber Band only."""
from __future__ import annotations

from typing import Callable
import math
import numpy as np


def require_rubberband():
    try:
        from pedalboard import time_stretch
    except ImportError as exc:
        raise ValueError("Rubber Band cần Pedalboard. Cài: uv sync --extra srt-quality (thêm --extra cuda nếu dùng GPU).") from exc
    return time_stretch


def rubberband_to_samples(audio, target_samples: int, sample_rate=48000,
                          check_stop: Callable = lambda: None) -> np.ndarray:
    if not isinstance(target_samples, (int, np.integer)) or isinstance(target_samples, bool) or target_samples < 1:
        raise ValueError("Khung audio phải có ít nhất một mẫu.")
    if not math.isfinite(float(sample_rate)) or sample_rate <= 0:
        raise ValueError("Tần số lấy mẫu phải lớn hơn zero.")
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim != 1 or not wav.size or not np.isfinite(wav).all():
        raise ValueError("Audio phải là waveform mono hữu hạn, không rỗng.")
    return _stretch_validated(wav, target_samples, sample_rate, check_stop)


def _stretch_validated(wav, target_samples, sample_rate, check_stop):
    check_stop()
    if len(wav) == target_samples:
        return wav.copy()
    if not np.any(wav):
        return np.zeros(target_samples, dtype=np.float32)
    time_stretch = require_rubberband()
    output = time_stretch(np.ascontiguousarray(wav), sample_rate,
                          stretch_factor=len(wav) / target_samples,
                          pitch_shift_in_semitones=0.0, high_quality=True,
                          retain_phase_continuity=True, preserve_formants=True)
    check_stop()
    output = np.asarray(output, dtype=np.float32).reshape(-1)
    if not output.size or not np.isfinite(output).all():
        raise ValueError("Rubber Band không trả về audio hợp lệ; hãy kiểm tra độ dài audio và tốc độ.")
    if len(output) < target_samples:
        return np.pad(output, (0, target_samples - len(output)))
    return output[:target_samples]


def stretch_rubberband(audio, speed: float, sample_rate=48000,
                       check_stop: Callable = lambda: None) -> np.ndarray:
    """Adjust ordinary desktop speech tempo with exact length and zero pitch shift."""
    if not math.isfinite(float(speed)) or not 0.5 <= speed <= 2:
        raise ValueError("Tốc độ phải nằm trong 0.5–2.0x.")
    if not math.isfinite(float(sample_rate)) or sample_rate <= 0:
        raise ValueError("Tần số lấy mẫu phải lớn hơn zero.")
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim != 1 or not np.isfinite(wav).all():
        raise ValueError("Audio phải là waveform mono hữu hạn.")
    check_stop()
    if speed == 1.0 or not wav.size:
        return wav.copy()
    return _stretch_validated(wav, max(1, round(len(wav) / speed)), sample_rate, check_stop)
