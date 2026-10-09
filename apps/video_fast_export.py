"""Native FFmpeg video retiming, including smooth speed ramps.

Maps are evaluated inside FFmpeg without RGB frame transfer through Python.
Dense ramp knots are simplified with a bounded error of 0.01 output frame.
"""
from __future__ import annotations

from collections import deque
from fractions import Fraction
import math
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time
from typing import Callable

import numpy as np


_PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow")
_NVENC_PRESETS = dict(zip(_PRESETS, ("p1", "p1", "p2", "p3", "p4", "p5", "p6")))
_ENCODER_CACHE: dict[tuple[str, int, int], tuple[bool, str]] = {}
_ENCODER_LOCK = threading.Lock()


class _FFmpegFailure(ValueError):
    """A native process failure, distinct from exceptions raised by callbacks."""


class _ProbeTimeout(TimeoutError):
    pass


def _ffmpeg_binary() -> str:
    # Use the same binary that MoviePy uses for its readers and audio writer.
    from moviepy.config import FFMPEG_BINARY

    return str(FFMPEG_BINARY)


def can_render_ffmpeg(mapping) -> bool:
    """Accept the monotone source/output maps shared by all desktop editors."""
    return (getattr(mapping, "segments", None) is not None and
            getattr(mapping, "source_knots", None) is not None and
            getattr(mapping, "output_knots", None) is not None)


def _mapping_knots(mapping) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(mapping.source_knots, dtype=np.float64)
    output = np.asarray(mapping.output_knots, dtype=np.float64)
    if (source.ndim != 1 or output.ndim != 1 or len(source) < 2 or
            len(source) != len(output) or not np.isfinite(source).all() or
            not np.isfinite(output).all() or not np.all(np.diff(source) > 0) or
            not np.all(np.diff(output) > 0) or source[0] != 0 or output[0] != 0 or
            not math.isclose(float(source[-1]), float(mapping.source_duration),
                             rel_tol=0, abs_tol=1e-8)):
        raise ValueError("Đường thời gian video không hợp lệ để xuất bằng FFmpeg.")
    return source, output


def _simplify_knots(source: np.ndarray, output: np.ndarray,
                    tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep endpoints and split wherever linear output-time error is too large.

    Error is tested at all original knots, bounding it across the entire
    piecewise-linear map. Iteration avoids recursion depth on long timelines.
    """
    keep = {0, len(source) - 1}
    pending = [(0, len(source) - 1)]
    while pending:
        begin, end = pending.pop()
        if end - begin <= 1:
            continue
        inner = source[begin + 1:end]
        slope = (output[end] - output[begin]) / (source[end] - source[begin])
        errors = np.abs(output[begin + 1:end] -
                        (output[begin] + (inner - source[begin]) * slope))
        index = int(np.argmax(errors))
        if errors[index] > tolerance:
            middle = begin + index + 1
            keep.add(middle)
            pending.extend(((begin, middle), (middle, end)))
    indices = sorted(keep)
    return source[indices], output[indices]


def _time_expression(source: np.ndarray, output: np.ndarray) -> str:
    """Build a balanced decision tree: each input frame visits O(log anchors)."""
    def number(value):
        return format(float(value), ".17g")

    def branch(begin, end):
        if end - begin == 1:
            slope = (output[begin + 1] - output[begin]) / (source[begin + 1] - source[begin])
            return (f"({number(output[begin])}+(ld(0)-{number(source[begin])})*"
                    f"{number(slope)})")
        middle = (begin + end) // 2
        return (f"if(lt(ld(0),{number(source[middle])}),"
                f"{branch(begin, middle)},{branch(middle, end)})")

    return f"st(0,(PTS-STARTPTS)*TB);({branch(0, len(source) - 1)})/TB"


def _video_filter(mapping, fps: float) -> str:
    source, output = _mapping_knots(mapping)
    source, output = _simplify_knots(source, output, 0.01 / fps)
    rate = Fraction(float(fps)).limit_denominator(1_000_000)
    rate_text = f"{rate.numerator}/{rate.denominator}"
    duration = format(float(output[-1]), ".17g")
    expression = _time_expression(source, output)
    # AVTB prevents source frame-rate time bases from rounding marker times.
    # Clone the last frame when a very slow final interval needs additional
    # samples, then trim precisely to the map's duration.
    return (f"settb=AVTB,setpts='{expression}',fps={rate_text}:start_time=0,"
            f"pad=ceil(iw/2)*2:ceil(ih/2)*2,"
            f"tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration}")


def _process_flags() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _run_ffmpeg(command: list[str], output_duration: float,
                check_stop: Callable, progress: Callable,
                timeout: float | None = None) -> str:
    """Drain both pipes and check cancellation even while no frames arrive.

    Progress keeps only the latest record, and diagnostics retain their last
    16 KiB. No unbounded queue or stderr buffer grows with video duration.
    """
    check_stop()
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   **_process_flags())
    except OSError as exc:
        raise _FFmpegFailure(f"Không khởi động được FFmpeg: {exc}") from exc
    updates: queue.Queue[float] = queue.Queue(maxsize=1)
    errors = deque(maxlen=16)

    def read_progress():
        try:
            for raw in iter(process.stdout.readline, b""):
                key, separator, value = raw.partition(b"=")
                if separator and key == b"out_time_us":
                    try:
                        seconds = max(0.0, int(value.strip()) / 1_000_000)
                    except ValueError:
                        continue
                    try:
                        updates.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        updates.put_nowait(seconds)
                    except queue.Full:
                        pass
        except (OSError, ValueError):
            pass  # Pipes may close during cancellation.

    def read_errors():
        try:
            while chunk := process.stderr.read(1024):
                errors.append(chunk)
        except (OSError, ValueError):
            pass

    readers = (threading.Thread(target=read_progress, daemon=True),
               threading.Thread(target=read_errors, daemon=True))
    for reader in readers:
        reader.start()
    started = time.monotonic()
    last_fraction = -1.0
    try:
        while True:
            check_stop()
            if timeout is not None and time.monotonic() - started > timeout:
                raise _ProbeTimeout("FFmpeg mất quá nhiều thời gian khi kiểm tra bộ mã hóa video.")
            try:
                seconds = updates.get(timeout=0.05)
            except queue.Empty:
                seconds = None
            if seconds is not None and output_duration > 0:
                fraction = min(0.995, max(0.0, seconds / output_duration))
                if fraction > last_fraction:
                    progress(fraction)
                    last_fraction = fraction
            if process.poll() is not None:
                break
        for reader in readers:
            reader.join(timeout=1)
        check_stop()
        detail = b"".join(errors).decode("utf-8", errors="replace").strip()
        if process.returncode:
            raise _FFmpegFailure(f"FFmpeg không xuất được video (mã {process.returncode}).\n{detail}")
        progress(1.0)
        return detail
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        for reader in readers:
            reader.join(timeout=1)
        for stream in (process.stdout, process.stderr):
            stream.close()


def _padded_size(size) -> tuple[int, int]:
    if size is None:
        return 320, 240
    try:
        width, height = (float(value) for value in size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Kích thước video không hợp lệ.") from exc
    if (not math.isfinite(width) or not math.isfinite(height) or
            width < 1 or height < 1 or width != int(width) or height != int(height)):
        raise ValueError("Kích thước video không hợp lệ.")
    return int(math.ceil(width / 2) * 2), int(math.ceil(height / 2) * 2)


def _probe_nvenc(check_stop: Callable, size=None) -> tuple[bool, str]:
    binary = _ffmpeg_binary()
    width, height = _padded_size(size)
    key = binary, width, height
    check_stop()
    # Exports normally run one at a time; the lock also prevents duplicate
    # device probes when this helper is used independently.
    with _ENCODER_LOCK:
        check_stop()
        if key not in _ENCODER_CACHE:
            command = [binary, "-hide_banner", "-loglevel", "error", "-nostdin",
                       "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r=30:d=0.05",
                       "-frames:v", "1", "-c:v", "h264_nvenc", "-preset", "p1",
                       "-pix_fmt", "yuv420p", "-f", "null", "-"]
            try:
                _run_ffmpeg(command, 0, check_stop, lambda fraction: None, timeout=8)
            except (_FFmpegFailure, _ProbeTimeout) as exc:
                check_stop()
                _ENCODER_CACHE[key] = (False, str(exc))
            else:
                _ENCODER_CACHE[key] = (True, "")
        return _ENCODER_CACHE[key]


def select_video_encoder(encoder: str = "auto",
                         check_stop: Callable = lambda: None, size=None) -> str:
    """Prefer NVENC only after a real frame can be encoded by the installed GPU."""
    if encoder not in ("auto", "libx264", "h264_nvenc"):
        raise ValueError("Bộ mã hóa video phải là auto, libx264 hoặc h264_nvenc.")
    check_stop()
    if encoder == "libx264":
        return encoder
    supported, detail = _probe_nvenc(check_stop, size)
    if supported:
        return "h264_nvenc"
    if encoder == "h264_nvenc":
        raise ValueError(f"Không dùng được bộ mã hóa NVIDIA NVENC.\n{detail}")
    return "libx264"


def get_video_encoding_options(quality: int, preset: str,
                               check_stop: Callable = lambda: None,
                               encoder: str = "auto", size=None) -> tuple[str, str, list[str]]:
    """Return codec, preset and arguments shared by native and MoviePy paths."""
    if not isinstance(quality, int) or not 0 <= quality <= 51:
        raise ValueError("Chất lượng video phải nằm trong 0–51.")
    if preset not in _PRESETS:
        raise ValueError("Preset mã hóa video không hợp lệ.")
    codec = select_video_encoder(encoder, check_stop, size)
    if codec == "h264_nvenc":
        if quality == 0:
            return codec, _NVENC_PRESETS[preset], ["-tune", "lossless", "-rc", "constqp", "-qp", "0"]
        return codec, _NVENC_PRESETS[preset], ["-rc", "vbr", "-cq", str(quality), "-b:v", "0"]
    return codec, preset, ["-crf", str(quality)]


def render_video_ffmpeg(source, destination, mapping, fps: float,
                        audio_path=None, quality: int = 20, preset: str = "fast",
                        check_stop: Callable = lambda: None,
                        progress: Callable = lambda fraction: None,
                        encoder: str = "auto", size=None) -> str:
    """Render constant intervals and smooth ramps natively; return the codec.

    ``audio_path`` is an already retimed AAC file and is copied without another
    encode. The destination must not exist and is removed on failure or cancel.
    Actual rendering errors are surfaced instead of silently retrying on CPU.
    """
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if not source.is_file():
        raise ValueError("Không tìm thấy file video nguồn.")
    if destination == source or destination.exists():
        raise ValueError("File video xuất đã tồn tại; hãy chọn một tên mới.")
    if destination.suffix.lower() != ".mp4":
        raise ValueError("Video xuất cần có đuôi .mp4.")
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS video phải là một số hữu hạn lớn hơn 0.")
    if not can_render_ffmpeg(mapping):
        raise ValueError("Unsupported video timeline mapping.")
    video_filter = _video_filter(mapping, fps)
    if audio_path is not None:
        audio_path = Path(audio_path).expanduser().resolve()
        if not audio_path.is_file():
            raise ValueError("Không tìm thấy âm thanh đã xử lý để ghép video.")
    if size is None:
        from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos

        try:
            size = ffmpeg_parse_infos(str(source))["video_size"]
        except (OSError, KeyError) as exc:
            raise ValueError(f"FFmpeg không đọc được kích thước video nguồn: {exc}") from exc
    codec, codec_preset, options = get_video_encoding_options(quality, preset, check_stop, encoder, size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation means failure cleanup can only delete our own file.
    with destination.open("xb"):
        pass
    completed = False
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".fffilter",
                                         prefix="video_", dir=destination.parent,
                                         delete=False) as script:
            script.write(video_filter)
            script_path = Path(script.name)
        try:
            command = [_ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-nostdin",
                       "-y", "-stats_period", "0.2", "-progress", "pipe:1", "-i", str(source)]
            if audio_path is not None:
                command.extend(["-i", str(audio_path)])
            command.extend(["-map", "0:v:0", "-filter_script:v", str(script_path),
                            "-c:v", codec, "-preset", codec_preset, *options,
                            "-pix_fmt", "yuv420p", "-fps_mode", "cfr"])
            if audio_path is not None:
                command.extend(["-map", "1:a:0", "-c:a", "copy"])
            else:
                command.append("-an")
            command.extend(["-t", format(float(mapping.output_duration), ".17g"),
                            "-movflags", "+faststart", str(destination)])
            progress(0.0)
            _run_ffmpeg(command, float(mapping.output_duration), check_stop, progress)
            check_stop()
            completed = True
        finally:
            script_path.unlink(missing_ok=True)
    finally:
        if not completed:
            destination.unlink(missing_ok=True)
    return codec
