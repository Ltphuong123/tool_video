"""MoviePy video speed ramps with a shared, monotone source/output timeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import math
import os
import uuid

import numpy as np


@dataclass(frozen=True)
class SpeedTimeMap:
    """Map output seconds back to source seconds without cuts or jumps."""

    source_knots: np.ndarray
    output_knots: np.ndarray
    source_duration: float
    start: float
    end: float
    speed: float
    ramp_seconds: float

    @property
    def output_duration(self) -> float:
        return float(self.output_knots[-1])

    def source_time(self, output_t):
        return np.interp(output_t, self.output_knots, self.source_knots)

    def output_time(self, source_t):
        return np.interp(source_t, self.source_knots, self.output_knots)


def build_speed_time_map(duration: float, start: float, end: float,
                         speed: float = 1.5, ramp_seconds: float = 0.5) -> SpeedTimeMap:
    """Integrate dt = ds / speed(s); ramps use smoothstep in source seconds.

    Only the two ramps require dense knots. A several-hour source therefore
    uses the same small mapping as a short clip, and unchanged sections stay
    exact. ``ramp_seconds=0`` selects an immediate, constant speed change.
    """
    duration, start, end, speed, ramp_seconds = (
        float(value) for value in (duration, start, end, speed, ramp_seconds)
    )
    if not all(math.isfinite(value) for value in (duration, start, end, speed, ramp_seconds)):
        raise ValueError("Thời lượng, mốc thời gian và tốc độ phải là số hữu hạn.")
    if duration <= 0:
        raise ValueError("Video phải có thời lượng lớn hơn 0.")
    if start < 0 or end <= start or end > duration:
        raise ValueError("Cần 0 ≤ mốc bắt đầu < mốc kết thúc ≤ thời lượng video.")
    if not 0.25 <= speed <= 4:
        raise ValueError("Tốc độ video phải nằm trong 0.25–4.0x.")
    if ramp_seconds < 0 or ramp_seconds > (end - start) / 2:
        raise ValueError("Thời gian chuyển tốc độ phải từ 0 đến một nửa độ dài đoạn đã chọn.")
    if speed == 1:
        source = np.array([0, duration], dtype=np.float64)
        output = source.copy()
    elif ramp_seconds == 0:
        source = np.unique(np.array([0, start, end, duration], dtype=np.float64))
        output = np.array([
            value if value <= start else
            start + (min(value, end) - start) / speed + max(0, value - end)
            for value in source
        ], dtype=np.float64)
    else:
        fraction = np.linspace(0, 1, 1025)
        smoothstep = fraction * fraction * (3 - 2 * fraction)
        ramp_speed = 1 + (speed - 1) * smoothstep
        increments = np.diff(fraction) * ramp_seconds * (
            1 / ramp_speed[:-1] + 1 / ramp_speed[1:]
        ) / 2
        ramp_output = np.concatenate(([0], np.cumsum(increments)))
        up_source = start + ramp_seconds * fraction
        up_output = start + ramp_output
        plateau_end = end - ramp_seconds
        plateau_output = up_output[-1] + max(0, plateau_end - up_source[-1]) / speed
        down_source = plateau_end + ramp_seconds * fraction
        down_output = plateau_output + ramp_output[-1] - ramp_output[::-1]
        source = np.concatenate(([0], up_source, [plateau_end], down_source, [duration]))
        output = np.concatenate(([0], up_output, [plateau_output], down_output,
                                 [down_output[-1] + duration - end]))
        keep = np.concatenate(([True], np.diff(source) > 0))
        source, output = source[keep], output[keep]
    source.setflags(write=False)
    output.setflags(write=False)
    return SpeedTimeMap(source, output, duration, start, end, speed, ramp_seconds)


def require_moviepy():
    """Import only when opening a video, so speech tools stay independent."""
    try:
        from moviepy import VideoFileClip, AudioFileClip
        from proglog import ProgressBarLogger
    except ImportError as exc:
        raise ValueError("Chỉnh video cần MoviePy 2. Cài: uv sync --extra video --extra srt-quality "
                         "(thêm --extra cuda nếu dùng GPU).") from exc
    return VideoFileClip, AudioFileClip, ProgressBarLogger


def probe_video(path) -> dict:
    """Read basic video information without importing or loading a TTS model."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("Không tìm thấy file video nguồn.")
    VideoFileClip, _, _ = require_moviepy()
    clip = None
    try:
        clip = VideoFileClip(str(source))
        _guard_reader_cleanup(clip)
        if not math.isfinite(float(clip.duration)) or clip.duration <= 0:
            raise ValueError("Không đọc được thời lượng hợp lệ từ video nguồn.")
        if not math.isfinite(float(clip.fps)) or clip.fps <= 0:
            raise ValueError("Không đọc được FPS hợp lệ từ video nguồn.")
        return {"duration": float(clip.duration), "fps": float(clip.fps),
                "width": int(clip.w), "height": int(clip.h), "has_audio": clip.audio is not None}
    finally:
        if clip is not None:
            clip.close()


def render_speed_segment(source, destination, start: float, end: float,
                         speed: float = 1.5, ramp_seconds: float = 0.5,
                         keep_audio: bool = True, quality: int = 20,
                         preset: str = "fast", check_stop: Callable = lambda: None,
                         progress: Callable = lambda fraction: None) -> SpeedTimeMap:
    """Create a new MP4, keeping all frames outside the selected source range.

    Progress is a fraction in [0, 1]. The source remains open only while
    rendering. A failed or cancelled job removes its own temporary files.
    """
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if not source.is_file():
        raise ValueError("Không tìm thấy file video nguồn.")
    if source == destination or destination.exists():
        raise ValueError("Chọn file MP4 mới; không được ghi đè video nguồn hoặc file đã có.")
    if destination.suffix.lower() != ".mp4":
        raise ValueError("Video đầu ra phải có đuôi .mp4.")
    if not isinstance(quality, int) or isinstance(quality, bool) or not 0 <= quality <= 51:
        raise ValueError("Chất lượng CRF phải là số nguyên trong 0–51.")
    if preset not in ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"):
        raise ValueError("Preset mã hóa video không hợp lệ.")
    check_stop()
    VideoFileClip, AudioFileClip, ProgressBarLogger = require_moviepy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    partial = destination.parent / f".video_{token}.mp4"
    audio_file = destination.parent / f".video_{token}.wav"
    mux_audio = destination.parent / f".video_{token}.m4a"
    source_clip = output_clip = processed_audio = None

    class RenderLogger(ProgressBarLogger):
        def callback(self, **changes):
            check_stop()

        def bars_callback(self, bar, attr, value, old_value=None):
            check_stop()
            if attr == "index":
                total = self.bars[bar].get("total", 0)
                if total:
                    progress(0.25 + 0.74 * min(1, max(0, value / total)))

    try:
        progress(0)
        source_clip = VideoFileClip(str(source), audio=keep_audio, audio_fps=48000)
        _guard_reader_cleanup(source_clip)
        mapping = build_speed_time_map(source_clip.duration, start, end, speed, ramp_seconds)
        fps = float(source_clip.fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Không đọc được FPS hợp lệ từ video nguồn.")
        if mapping.output_duration * fps < 1:
            raise ValueError("Video đầu ra ngắn hơn một khung hình.")

        def checked_time(t):
            check_stop()
            return mapping.source_time(t)

        output_clip = source_clip.without_audio().time_transform(
            checked_time, apply_to=["mask"]
        ).with_duration(mapping.output_duration)
        if keep_audio and source_clip.audio is not None:
            if speed == 1:
                output_clip = output_clip.with_audio(source_clip.audio)
            else:
                _write_ramped_audio(source_clip.audio, audio_file, mapping,
                                    check_stop=check_stop,
                                    progress=lambda fraction: progress(0.18 * fraction))
                processed_audio = AudioFileClip(str(audio_file), fps=48000)
                _guard_reader_cleanup(processed_audio)
                output_clip = output_clip.with_audio(processed_audio)
        if output_clip.audio is not None:
            _encode_audio(output_clip.audio, mux_audio, check_stop=check_stop,
                          progress=lambda fraction: progress(0.18 + 0.07 * fraction))
        progress(0.25)
        # MoviePy 2.2.1 rounds the raw input FPS to two decimals. Restore the
        # source clock before encoding, otherwise long 23.976/29.97 videos
        # gradually drift away from their exact audio timeline.
        input_fps = float(f"{fps:.2f}")
        filters = (f"setpts=({input_fps:.12g}/{fps:.12g})*PTS,"
                   "pad=ceil(iw/2)*2:ceil(ih/2)*2")
        output_clip.write_videofile(
            str(partial), fps=fps, codec="libx264", preset=preset,
            threads=max(1, min(4, os.cpu_count() or 1)),
            audio=str(mux_audio) if output_clip.audio is not None else False,
            audio_codec="copy", logger=RenderLogger(),
            ffmpeg_params=["-crf", str(quality), "-pix_fmt", "yuv420p",
                           "-vf", filters, "-r", f"{fps:.12g}", "-movflags", "+faststart"],
        )
        check_stop()
        if not partial.is_file() or not partial.stat().st_size:
            raise ValueError("MoviePy không tạo được video đầu ra.")
        # On Windows rename refuses an existing destination, avoiding replacement.
        os.rename(partial, destination)
        progress(1)
        return mapping
    finally:
        # Copies share their readers. Close owning clips after rendering ends.
        try:
            if processed_audio is not None:
                processed_audio.close()
        finally:
            try:
                if source_clip is not None:
                    source_clip.close()
            finally:
                for path in (partial, audio_file, mux_audio):
                    path.unlink(missing_ok=True)


def _guard_reader_cleanup(clip):
    """MoviePy 2.2.1 skips pipe closure when a decoder has already exited."""
    for candidate in (clip, getattr(clip, "audio", None)):
        reader = getattr(candidate, "reader", None)
        if reader is None or getattr(reader, "_desktop_cleanup_guard", False):
            continue
        original_close = reader.close

        def close(*args, _reader=reader, _original=original_close, **kwargs):
            process = _reader.proc
            try:
                return _original(*args, **kwargs)
            finally:
                if process is not None:
                    for name in ("stdin", "stdout", "stderr"):
                        stream = getattr(process, name, None)
                        if stream is not None and not stream.closed:
                            stream.close()

        reader.close = close
        reader._desktop_cleanup_guard = True


def _encode_audio(audio, filename, check_stop, progress):
    """Use a context manager: MoviePy's audio convenience writer lacks finally."""
    from moviepy.audio.io.ffmpeg_audiowriter import FFMPEG_AudioWriter

    rate = 48000
    total = max(1, round(audio.duration * rate))
    consumed = 0
    with FFMPEG_AudioWriter(str(filename), rate, nbytes=4, nchannels=audio.nchannels,
                            codec="aac", bitrate="192k") as writer:
        for chunk in audio.iter_chunks(chunksize=24000, fps=rate, quantize=True,
                                       nbytes=4, logger=None):
            check_stop()
            writer.write_frames(chunk)
            consumed += len(chunk)
            progress(min(1, consumed / total))
    check_stop()


def _write_ramped_audio(audio, filename, mapping, check_stop=lambda: None,
                        progress=lambda fraction: None):
    """Stream pitch-preserving audio on the video's exact sample timeline.

    Rubber Band accepts a constant factor per call. Small overlapping windows
    approximate the ramp; a shared timeline places them without accumulating
    drift. Only a few seconds of audio are resident even for a long video.
    """
    import soundfile as sf
    from apps.speech_speed import require_rubberband

    time_stretch = require_rubberband()
    rate = 48000
    channels = int(audio.nchannels)
    if channels not in (1, 2):
        raise ValueError("Audio video cần là mono hoặc stereo.")
    total = max(1, round(mapping.output_duration * rate))
    pending_start = 0
    pending = np.empty((0, channels), dtype=np.float32)
    weights = np.empty(0, dtype=np.float32)

    def source_samples(first, last):
        check_stop()
        result = np.zeros((last - first, channels), dtype=np.float32)
        valid_last = min(last, max(0, math.ceil(float(audio.duration) * rate)))
        if valid_last > first:
            times = np.arange(first, valid_last, dtype=np.float64) / rate
            decoded = np.asarray(audio.get_frame(times), dtype=np.float32)
            if decoded.ndim == 1:
                decoded = decoded.reshape(-1, channels)
            if decoded.shape != (valid_last - first, channels) or not np.isfinite(decoded).all():
                raise ValueError("Audio nguồn không hợp lệ hoặc có mẫu không hữu hạn.")
            result[:len(decoded)] = decoded
        return result

    with sf.SoundFile(str(filename), "w", samplerate=rate, channels=channels,
                      format="WAV", subtype="FLOAT") as output:
        def flush(until):
            nonlocal pending_start, pending, weights
            count = min(max(0, until - pending_start), len(pending))
            if count:
                if np.any(weights[:count] <= 0):
                    raise ValueError("Đường thời gian audio có khoảng trống ngoài dự kiến.")
                output.write(pending[:count] / weights[:count, None])
                pending = pending[count:].copy()
                weights = weights[count:].copy()
                pending_start += count

        for begin, end in _audio_windows(mapping):
            check_stop()
            source_first = max(0, round((begin - 0.04) * rate))
            source_last = min(round(mapping.source_duration * rate), round((end + 0.04) * rate))
            out_first = max(0, round(float(mapping.output_time(source_first / rate)) * rate))
            out_last = min(total, round(float(mapping.output_time(source_last / rate)) * rate))
            if out_last <= out_first:
                continue
            flush(out_first)
            data = source_samples(source_first, source_last)
            expected = out_last - out_first
            # Unchanged regions pass through without a native tempo operation.
            unchanged_before = end <= mapping.start
            unchanged_after = begin >= mapping.end
            if unchanged_before or unchanged_after:
                stretched = data
                # Anchor unchanged samples on the adjacent untouched timeline;
                # only the small overlap may need rounding/padding.
                if len(stretched) < expected:
                    amount = expected - len(stretched)
                    padding = (amount, 0) if unchanged_after else (0, amount)
                    stretched = np.pad(stretched, (padding, (0, 0)))
                if len(stretched) > expected:
                    stretched = stretched[-expected:] if unchanged_after else stretched[:expected]
            elif not np.any(data):
                stretched = np.zeros((expected, channels), dtype=np.float32)
            else:
                stretched = np.asarray(time_stretch(
                    np.ascontiguousarray(data.T), rate,
                    stretch_factor=len(data) / expected, pitch_shift_in_semitones=0.0,
                    high_quality=True, retain_phase_continuity=True, preserve_formants=True,
                ), dtype=np.float32)
                check_stop()
                if stretched.ndim == 1 and channels == 1:
                    stretched = stretched[None, :]
                if (stretched.ndim != 2 or stretched.shape[0] != channels or
                        not stretched.shape[1] or not np.isfinite(stretched).all()):
                    raise ValueError("Rubber Band không trả về audio video hợp lệ.")
                stretched = stretched.T
                if len(stretched) < expected:
                    stretched = np.pad(stretched, ((0, expected - len(stretched)), (0, 0)))
                stretched = stretched[:expected]
            core_first = round(float(mapping.output_time(begin)) * rate)
            core_last = round(float(mapping.output_time(end)) * rate)
            fade_in = max(0, min(expected, core_first - out_first))
            fade_out = max(0, min(expected, out_last - core_last))
            window = np.ones(expected, dtype=np.float32)
            if fade_in and source_first > 0:
                window[:fade_in] = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade_in) / fade_in)
            if fade_out and out_last < total:
                window[-fade_out:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(fade_out) / fade_out)
            needed = out_last - pending_start - len(pending)
            if needed > 0:
                pending = np.pad(pending, ((0, needed), (0, 0)))
                weights = np.pad(weights, (0, needed))
            offset = out_first - pending_start
            pending[offset:offset + expected] += stretched * window[:, None]
            weights[offset:offset + expected] += window
            progress(min(1, end / mapping.source_duration))
        flush(total)
        if output.frames != total:
            raise ValueError("Thời lượng audio đầu ra không khớp đường thời gian video.")
    check_stop()


def _audio_windows(mapping):
    """Generate bounded source intervals, refining fast ramps for sync."""
    points = sorted(set((0.0, mapping.start, mapping.start + mapping.ramp_seconds,
                         mapping.end - mapping.ramp_seconds, mapping.end,
                         mapping.source_duration)))

    def refined(begin, end):
        # Measure time-map curvature including overlap. At most ~8 ms of
        # timing approximation per window; absolute positions prevent drift.
        ext_begin = max(0, begin - 0.04)
        ext_end = min(mapping.source_duration, end + 0.04)
        positions = np.linspace(ext_begin, ext_end, 9)
        actual = mapping.output_time(positions)
        linear = np.linspace(actual[0], actual[-1], len(positions))
        if np.max(np.abs(actual - linear)) > 0.008 and end - begin > 0.05:
            middle = (begin + end) / 2
            yield from refined(begin, middle)
            yield from refined(middle, end)
        else:
            yield begin, end

    for section_begin, section_end in zip(points[:-1], points[1:]):
        if section_end <= section_begin:
            continue
        changed = section_begin < mapping.end and section_end > mapping.start
        ramp = changed and mapping.ramp_seconds > 0 and (
            section_end <= mapping.start + mapping.ramp_seconds or
            section_begin >= mapping.end - mapping.ramp_seconds
        )
        maximum = 0.5 if ramp else (2.0 if changed else 4.0)
        chunks = max(1, math.ceil((section_end - section_begin) / maximum))
        for index in range(chunks):
            begin = section_begin + (section_end - section_begin) * index / chunks
            end = section_begin + (section_end - section_begin) * (index + 1) / chunks
            yield from refined(begin, end)
