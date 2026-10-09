"""Video retiming with a shared timeline, native FFmpeg and GPU encoding."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TYPE_CHECKING
import math
import os
import uuid

import numpy as np

if TYPE_CHECKING:
    from apps.video_markers import MarkerTimeMap, TimeMarker


@dataclass(frozen=True)
class SpeedSegment:
    """A speed change defined on the original video's source timeline."""

    start: float
    end: float
    speed: float = 1.5
    ramp_seconds: float = 0.5


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
    def segments(self) -> tuple[SpeedSegment, ...]:
        return (SpeedSegment(self.start, self.end, self.speed, self.ramp_seconds),)

    @property
    def output_duration(self) -> float:
        return float(self.output_knots[-1])

    def source_time(self, output_t):
        return np.interp(output_t, self.output_knots, self.source_knots)

    def output_time(self, source_t):
        return np.interp(source_t, self.source_knots, self.output_knots)


@dataclass(frozen=True)
class MultiSpeedTimeMap:
    """One continuous timeline for sorted, nonoverlapping source segments."""

    source_knots: np.ndarray
    output_knots: np.ndarray
    source_duration: float
    segments: tuple[SpeedSegment, ...]

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


def build_multi_speed_time_map(duration: float,
                               segments: Iterable[SpeedSegment]) -> MultiSpeedTimeMap:
    """Compose speed ramps while keeping gaps at their original 1x speed.

    Segment times always refer to the source video. Input order is immaterial;
    overlapping ranges are rejected, while adjacent ranges and an empty list
    are valid. Knot storage grows with the number of ramps, not video length.
    """
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video phải có thời lượng hữu hạn lớn hơn 0.")
    normalized = []
    for segment in segments:
        if not isinstance(segment, SpeedSegment):
            raise ValueError("Mỗi đoạn tốc độ phải là một SpeedSegment.")
        values = tuple(float(value) for value in
                       (segment.start, segment.end, segment.speed, segment.ramp_seconds))
        # Reuse single-segment validation before sorting (including NaN inputs).
        local = build_speed_time_map(duration, *values)
        normalized.append((SpeedSegment(*values), local))
    normalized.sort(key=lambda item: (item[0].start, item[0].end))
    for previous, following in zip(normalized, normalized[1:]):
        if following[0].start < previous[0].end:
            raise ValueError("Các đoạn thay đổi tốc độ không được chồng lấn.")

    source_parts = [np.array([0.0], dtype=np.float64)]
    output_parts = [np.array([0.0], dtype=np.float64)]
    source_cursor = output_cursor = 0.0
    for segment, local in normalized:
        if segment.start > source_cursor:
            output_cursor += segment.start - source_cursor
            source_parts.append(np.array([segment.start], dtype=np.float64))
            output_parts.append(np.array([output_cursor], dtype=np.float64))
        inner = local.source_knots[(local.source_knots > segment.start) &
                                   (local.source_knots < segment.end)]
        segment_source = np.concatenate((inner, [segment.end]))
        segment_output = (local.output_time(segment_source) -
                          float(local.output_time(segment.start)) + output_cursor)
        source_parts.append(segment_source)
        output_parts.append(segment_output)
        source_cursor = segment.end
        output_cursor = float(segment_output[-1])
    if source_cursor < duration:
        source_parts.append(np.array([duration], dtype=np.float64))
        output_parts.append(np.array([output_cursor + duration - source_cursor], dtype=np.float64))
    source = np.concatenate(source_parts)
    output = np.concatenate(output_parts)
    if not np.all(np.diff(source) > 0) or not np.all(np.diff(output) > 0):
        raise ValueError("Các mốc thời gian quá gần nhau để tạo đường thời gian hợp lệ.")
    source.setflags(write=False)
    output.setflags(write=False)
    return MultiSpeedTimeMap(source, output, duration,
                             tuple(segment for segment, _ in normalized))


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
                         keep_audio: bool = False, quality: int = 20,
                         preset: str = "fast", check_stop: Callable = lambda: None,
                         progress: Callable = lambda fraction: None) -> SpeedTimeMap:
    """Create a new MP4, keeping all frames outside the selected source range.

    Progress is a fraction in [0, 1]. The source remains open only while
    rendering. A failed or cancelled job removes its own temporary files.
    """
    return _render_speed_mapping(
        source, destination,
        lambda duration: build_speed_time_map(duration, start, end, speed, ramp_seconds),
        keep_audio, quality, preset, check_stop, progress,
    )


def render_speed_segments(source, destination, segments: Iterable[SpeedSegment],
                          keep_audio: bool = False, quality: int = 20,
                          preset: str = "fast", check_stop: Callable = lambda: None,
                          progress: Callable = lambda fraction: None) -> MultiSpeedTimeMap:
    """Render multiple source ranges using one shared video/audio timeline."""
    segments = tuple(segments)
    return _render_speed_mapping(
        source, destination,
        lambda duration: build_multi_speed_time_map(duration, segments),
        keep_audio, quality, preset, check_stop, progress,
    )


def render_marker_alignment(source, destination, old_markers: Iterable[TimeMarker],
                            new_markers: Iterable[TimeMarker],
                            keep_audio: bool = False, quality: int = 20,
                            preset: str = "fast", check_stop: Callable = lambda: None,
                            progress: Callable = lambda fraction: None) -> MarkerTimeMap:
    """Render the exact marker alignment with the shared video/audio pipeline."""
    from apps.video_markers import build_marker_time_map

    old_markers, new_markers = tuple(old_markers), tuple(new_markers)
    return _render_speed_mapping(
        source, destination,
        lambda duration: build_marker_time_map(duration, old_markers, new_markers),
        keep_audio, quality, preset, check_stop, progress,
    )


def _render_speed_mapping(source, destination, mapping_factory: Callable,
                           keep_audio: bool, quality: int, preset: str,
                           check_stop: Callable, progress: Callable):
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
    VideoFileClip, _, ProgressBarLogger = require_moviepy()
    from apps.video_fast_export import (
        can_render_ffmpeg, get_video_encoding_options, render_video_ffmpeg,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    partial = destination.parent / f".video_{token}.mp4"
    mux_audio = destination.parent / f".video_{token}.m4a"
    source_clip = output_clip = None
    video_start = 0.25 if keep_audio else 0.0
    video_span = 0.99 - video_start

    class RenderLogger(ProgressBarLogger):
        def callback(self, **changes):
            check_stop()

        def bars_callback(self, bar, attr, value, old_value=None):
            check_stop()
            if attr == "index":
                total = self.bars[bar].get("total", 0)
                if total:
                    progress(video_start + video_span * min(1, max(0, value / total)))

    try:
        progress(0)
        source_clip = VideoFileClip(str(source), audio=keep_audio, audio_fps=48000)
        _guard_reader_cleanup(source_clip)
        mapping = mapping_factory(source_clip.duration)
        fps = float(source_clip.fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Không đọc được FPS hợp lệ từ video nguồn.")
        if mapping.output_duration * fps < 1:
            raise ValueError("Video đầu ra ngắn hơn một khung hình.")

        def checked_time(t):
            check_stop()
            return mapping.source_time(t)

        has_audio = keep_audio and source_clip.audio is not None
        if keep_audio and source_clip.audio is not None:
            if all(segment.speed == 1 for segment in mapping.segments):
                _encode_audio(source_clip.audio, mux_audio, check_stop=check_stop,
                              progress=lambda fraction: progress(0.25 * fraction))
            else:
                from apps.video_export_audio import encode_mapped_audio

                encode_mapped_audio(source_clip.audio, mux_audio, mapping,
                                    check_stop=check_stop,
                                    progress=lambda fraction: progress(0.25 * fraction))
        progress(video_start)
        if can_render_ffmpeg(mapping):
            # All shared timelines, including smooth ramps, stay inside
            # FFmpeg, avoiding Python RGB copies and repeated seeking.
            render_video_ffmpeg(
                source, partial, mapping, fps, audio_path=mux_audio if has_audio else None,
                quality=quality, preset=preset, check_stop=check_stop,
                progress=lambda fraction: progress(video_start + video_span * fraction),
                size=source_clip.size,
            )
        else:
            output_clip = source_clip.without_audio().time_transform(
                checked_time, apply_to=["mask"]
            ).with_duration(mapping.output_duration)
            codec, codec_preset, codec_options = get_video_encoding_options(
                quality, preset, check_stop=check_stop, size=source_clip.size,
            )
            # MoviePy rounds raw input FPS to two decimals. Restore the clock
            # so fractional-FPS sources do not gradually drift from the audio.
            input_fps = float(f"{fps:.2f}")
            filters = (f"setpts=({input_fps:.12g}/{fps:.12g})*PTS,"
                       "pad=ceil(iw/2)*2:ceil(ih/2)*2")
            output_clip.write_videofile(
                str(partial), fps=fps, codec=codec, preset=codec_preset,
                threads=os.cpu_count() or 1,
                audio=str(mux_audio) if has_audio else False,
                audio_codec="copy", logger=RenderLogger(),
                ffmpeg_params=[*codec_options, "-pix_fmt", "yuv420p",
                               "-vf", filters, "-r", f"{fps:.12g}", "-movflags", "+faststart"],
            )
        check_stop()
        if not partial.is_file() or not partial.stat().st_size:
            raise ValueError("Không tạo được video đầu ra.")
        # On Windows rename refuses an existing destination, avoiding replacement.
        os.rename(partial, destination)
        progress(1)
        return mapping
    finally:
        # Copies share their readers. Close the owning source after rendering.
        try:
            if source_clip is not None:
                source_clip.close()
        finally:
            for path in (partial, mux_audio):
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
    """Write the same streamed, pitch-preserving samples used by preview playback."""
    import soundfile as sf

    with sf.SoundFile(str(filename), "w", samplerate=48000, channels=int(audio.nchannels),
                      format="WAV", subtype="FLOAT") as output:
        for chunk in _ramped_audio_chunks(audio, mapping, check_stop=check_stop, progress=progress):
            output.write(chunk)
        if output.frames != max(1, round(mapping.output_duration * 48000)):
            raise ValueError("Thời lượng audio đầu ra không khớp đường thời gian video.")
    check_stop()

def _ramped_audio_chunks(audio, mapping, check_stop=lambda: None,
                         progress=lambda fraction: None, start_output=0.0, workers=1):
    """Stream pitch-preserving audio on the video's exact sample timeline.

    Rubber Band accepts a constant factor per call. Small overlapping windows
    approximate the ramp; a shared timeline places them without accumulating
    drift. Only a few seconds of audio are resident even for a long video.
    """
    from apps.speech_speed import require_rubberband

    time_stretch = require_rubberband() if any(s.speed != 1 for s in mapping.segments) else None
    rate = 48000
    channels = int(audio.nchannels)
    if channels not in (1, 2):
        raise ValueError("Audio video cần là mono hoặc stereo.")
    total = max(1, round(mapping.output_duration * rate))
    if not math.isfinite(start_output) or not 0 <= start_output <= mapping.output_duration:
        raise ValueError("Mốc bắt đầu phát audio không hợp lệ.")
    pending_start = round(start_output * rate)
    seek_source = float(mapping.source_time(start_output))
    pending = np.empty((0, channels), dtype=np.float32)
    weights = np.empty(0, dtype=np.float32)

    def source_samples(first, last):
        check_stop()
        result = np.zeros((last - first, channels), dtype=np.float32)
        valid_first = max(0, first)
        valid_last = min(last, max(0, math.ceil(float(audio.duration) * rate)))
        if valid_last > valid_first:
            # MoviePy 2.2's recursive array reads accidentally pass a boolean
            # mask as timestamps when a request exceeds half its reader buffer.
            # Keep each native read within that limit; this also bounds seek I/O.
            reader = getattr(audio, "reader", None)
            block = max(1, int(getattr(reader, "buffersize", rate * 2)) // 2)
            pieces = []
            for begin in range(valid_first, valid_last, block):
                check_stop()
                times = np.arange(begin, min(begin + block, valid_last), dtype=np.float64) / rate
                pieces.append(np.asarray(audio.get_frame(times), dtype=np.float32))
            decoded = np.concatenate(pieces)
            if decoded.ndim == 1:
                decoded = decoded.reshape(-1, channels)
            if decoded.shape != (valid_last - valid_first, channels) or not np.isfinite(decoded).all():
                raise ValueError("Audio nguồn không hợp lệ hoặc có mẫu không hữu hạn.")
            offset = valid_first - first
            result[offset:offset + len(decoded)] = decoded
        return result

    def flush(until):
        nonlocal pending_start, pending, weights
        count = min(max(0, until - pending_start), len(pending))
        if count:
            if np.any(weights[:count] <= 0):
                raise ValueError("Đường thời gian audio có khoảng trống ngoài dự kiến.")
            chunk = pending[:count] / weights[:count, None]
            pending = pending[count:].copy()
            weights = weights[count:].copy()
            pending_start += count
            return chunk
        return None

    def prepare_windows():
        for begin, end in _audio_windows(mapping):
            check_stop()
            if end <= seek_source:
                continue
            begin = max(begin, seek_source)
            # Exact marker alignment may need speeds outside the manual 0.25–4x
            # range. Bound overlap in output seconds as well as source seconds,
            # so an extremely slow section cannot allocate minutes of samples.
            overlap_begin = max(0, begin - 0.04,
                                float(mapping.source_time(max(0, float(mapping.output_time(begin)) - 0.16))))
            overlap_end = min(mapping.source_duration, end + 0.04,
                              float(mapping.source_time(float(mapping.output_time(end)) + 0.16)))
            source_first = max(0, round(overlap_begin * rate))
            source_last = min(round(mapping.source_duration * rate), round(overlap_end * rate))
            core_first = round(float(mapping.output_time(begin)) * rate)
            core_last = round(float(mapping.output_time(end)) * rate)
            # Source sample rounding can itself span many output samples at very
            # slow speeds. Cap the final allocation after rounding, retaining the
            # complete core even when it is shorter than one source sample.
            overlap_samples = round(0.16 * rate)
            out_first = min(core_first, max(0, core_first - overlap_samples,
                                          round(float(mapping.output_time(source_first / rate)) * rate)))
            out_last = max(core_last, min(total, core_last + overlap_samples,
                                         round(float(mapping.output_time(source_last / rate)) * rate)))
            if out_last <= out_first:
                continue
            expected = out_last - out_first
            # Unchanged regions pass through without a native tempo operation.
            changed = any(segment.speed != 1 and begin < segment.end and end > segment.start
                          for segment in mapping.segments)
            if not changed:
                # Anchor samples on the core's exact 1x timeline. This also
                # handles gaps with a changed segment on each side; overlap
                # samples fade into the neighboring stretched windows.
                shift = round(begin * rate) - round(float(mapping.output_time(begin)) * rate)
                data = source_samples(out_first + shift, out_last + shift)
            else:
                data = source_samples(source_first, source_last)
            yield (begin, end, source_first, out_first, out_last, core_first, core_last, changed, data)

    def stretch_window(window):
        check_stop()
        _, _, _, out_first, out_last, _, _, changed, data = window
        expected = out_last - out_first
        if not changed:
            return window, data
        if changed and not np.any(data):
            stretched = np.zeros((expected, channels), dtype=np.float32)
        elif changed:
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
        return window, stretched

    from apps.video_audio_parallel import ordered_parallel_map

    results = ordered_parallel_map(
        stretch_window, prepare_windows(), check_stop=check_stop,
        workers=workers if time_stretch is not None else 1,
    )
    with closing(results):
        for descriptor, stretched in results:
            begin, end, source_first, out_first, out_last, core_first, core_last, _, _ = descriptor
            expected = out_last - out_first
            chunk = flush(out_first)
            if chunk is not None:
                yield chunk
            fade_in = max(0, min(expected, core_first - out_first))
            fade_out = max(0, min(expected, out_last - core_last))
            window = np.ones(expected, dtype=np.float32)
            if fade_in and source_first > 0:
                window[:fade_in] = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade_in) / fade_in)
            if fade_out and out_last < total:
                window[-fade_out:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(fade_out) / fade_out)
            drop = max(0, pending_start - out_first)
            if drop:
                stretched, window = stretched[drop:], window[drop:]
                out_first += drop
                expected -= drop
            if expected <= 0:
                continue
            needed = out_last - pending_start - len(pending)
            if needed > 0:
                pending = np.pad(pending, ((0, needed), (0, 0)))
                weights = np.pad(weights, (0, needed))
            offset = out_first - pending_start
            pending[offset:offset + expected] += stretched * window[:, None]
            weights[offset:offset + expected] += window
            progress(min(1, end / mapping.source_duration))
    chunk = flush(total)
    if chunk is not None:
        yield chunk
    if pending_start != total:
        raise ValueError("Thời lượng audio đầu ra không khớp đường thời gian video.")
    check_stop()


def _audio_windows(mapping):
    """Generate bounded source intervals, refining fast ramps for sync."""
    segments = tuple(segment for segment in mapping.segments if segment.speed != 1)
    points = {0.0, mapping.source_duration}
    for segment in segments:
        points.update((segment.start, segment.start + segment.ramp_seconds,
                       segment.end - segment.ramp_seconds, segment.end))
    points = sorted(points)

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
        active = next((segment for segment in segments
                       if section_begin < segment.end and section_end > segment.start), None)
        changed = active is not None
        ramp = changed and active.ramp_seconds > 0 and (
            section_end <= active.start + active.ramp_seconds or
            section_begin >= active.end - active.ramp_seconds
        )
        maximum = 0.5 if ramp else (2.0 if changed else 4.0)
        output_length = float(mapping.output_time(section_end) - mapping.output_time(section_begin))
        chunks = max(1, math.ceil((section_end - section_begin) / maximum),
                     math.ceil(output_length / 4.0))
        for index in range(chunks):
            begin = section_begin + (section_end - section_begin) * index / chunks
            end = section_begin + (section_end - section_begin) * (index + 1) / chunks
            yield from refined(begin, end)
