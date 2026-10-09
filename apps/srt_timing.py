"""Fit desktop speech to SRT windows without shifting later cues."""
from __future__ import annotations

from typing import Callable
import math
import numpy as np

from apps.srt_speech import Cue
from apps.speech_speed import rubberband_to_samples




def cue_sample_windows(cues: list[Cue], sample_rate: int = 48000) -> list[tuple[int, int]]:
    """Return starts/deadlines; overlapping cues end at the next start."""
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("Tần số lấy mẫu phải là số nguyên dương.")
    windows = []
    for index, cue in enumerate(cues):
        start = cue.start_ms * sample_rate // 1000
        deadline = cue.end_ms * sample_rate // 1000
        if start < 0 or deadline <= start:
            raise ValueError(f"Câu SRT {cue.index}: khung thời gian không hợp lệ.")
        if index + 1 < len(cues):
            next_start = cues[index + 1].start_ms * sample_rate // 1000
            if next_start <= start:
                raise ValueError(f"Câu SRT {cue.index} và {cues[index + 1].index}: mốc bắt đầu phải khác nhau và tăng dần.")
            deadline = min(deadline, next_start)
        windows.append((start, deadline))
    return windows


def fit_clip_to_samples(audio, target_samples: int, sample_rate=48000,
                        check_stop: Callable = lambda: None) -> tuple[np.ndarray, float]:
    if not isinstance(target_samples, (int, np.integer)) or isinstance(target_samples, bool) or target_samples < 1:
        raise ValueError("Khung audio phải có ít nhất một mẫu.")
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim != 1 or not np.isfinite(wav).all() or not math.isfinite(float(sample_rate)) or sample_rate <= 0:
        raise ValueError("Audio hoặc tần số lấy mẫu không hợp lệ.")
    check_stop()
    if len(wav) <= target_samples:
        fitted = wav.copy()
    else:
        fitted = rubberband_to_samples(wav, target_samples, sample_rate, check_stop)
    return fitted, max(1.0, len(wav) / target_samples)


def trim_srt_silence(audio) -> np.ndarray:
    """Remove quiet model padding so the detected voice begins at the cue start.

    A conservative -60 dB relative threshold keeps quiet consonants. This is
    waveform onset detection, not linguistic alignment or word recognition.
    """
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim != 1 or not wav.size or not np.isfinite(wav).all():
        raise ValueError("Câu SRT không chứa audio mono hữu hạn.")
    peak = float(np.abs(wav).max())
    threshold = max(1e-7, peak * 0.001)
    active = np.flatnonzero(np.abs(wav) > threshold)
    if not active.size:
        raise ValueError("Câu SRT chỉ có khoảng lặng; hãy tạo lại giọng đọc.")
    return wav[active[0]:active[-1] + 1].copy()


def prepare_srt_clip(audio, window_samples: int, min_speed=1.0, sample_rate=48000,
                      check_stop: Callable = lambda: None) -> tuple[np.ndarray, float]:
    """Trim padding, then apply max(initial speed, voice duration / window).

    One time-stretch avoids applying the initial and corrective speeds twice.
    Shorter clips stay at the initial speed, leaving silence in the timeline.
    """
    if not math.isfinite(float(min_speed)) or not 0.5 <= min_speed <= 2:
        raise ValueError("Tốc độ ban đầu / tối thiểu SRT phải nằm trong 0.5–2.0x.")
    if not isinstance(window_samples, (int, np.integer)) or isinstance(window_samples, bool) or window_samples < 1:
        raise ValueError("Khung audio phải có ít nhất một mẫu.")
    if not math.isfinite(float(sample_rate)) or sample_rate <= 0:
        raise ValueError("Tần số lấy mẫu phải lớn hơn zero.")
    check_stop()
    wav = trim_srt_silence(audio)
    speed = max(float(min_speed), len(wav) / window_samples)
    # Floor the base duration so sample rounding never makes the voice slower
    # than the requested minimum. A long clip has the exact window length.
    target = min(window_samples, max(1, math.floor(len(wav) / min_speed)))
    if target == len(wav):
        return wav, speed
    fitted = rubberband_to_samples(wav, target, sample_rate, check_stop)
    check_stop()
    # Move any quiet processing edge to the end without changing the requested
    # duration: the detected voice starts at sample zero and remains in-window.
    aligned = trim_srt_silence(fitted)
    if len(aligned) == len(fitted):
        return fitted, speed
    return np.pad(aligned, (0, len(fitted) - len(aligned))), speed


def lay_fitted_timeline(clips: list[np.ndarray], cues: list[Cue], sample_rate=48000,
                        check_stop: Callable = lambda: None) -> np.ndarray:
    if not cues or len(clips) != len(cues):
        raise ValueError("Số audio không khớp số câu SRT.")
    windows = cue_sample_windows(cues, sample_rate)
    track = np.zeros(max(cue.end_ms for cue in cues) * sample_rate // 1000, dtype=np.float32)
    for cue, clip, (start, deadline) in zip(cues, clips, windows):
        check_stop()
        wav = np.asarray(clip, dtype=np.float32)
        if wav.ndim != 1 or not wav.size or not np.isfinite(wav).all():
            raise ValueError(f"Câu SRT {cue.index}: audio không hợp lệ.")
        if len(wav) > deadline - start:
            raise ValueError(f"Câu SRT {cue.index}: audio chưa vừa khung thời gian.")
        track[start:start + len(wav)] = wav
    return track
