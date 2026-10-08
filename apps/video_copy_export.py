"""Retiming compressed video packets, preserving every original video frame.

Unlike a fixed-frame-rate render, stream copying changes only presentation and
decoding timestamps. No pixels are decoded, resized or encoded, and slower
sections hold their original frames for longer. The resulting MP4 has variable
frame timing. Audio supplied here has already been retimed by the audio renderer.
"""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
from typing import Callable

import numpy as np

from apps.video_fast_export import _ffmpeg_binary, _mapping_knots, _run_ffmpeg


def _mapped_clock(source: np.ndarray, output: np.ndarray) -> str:
    """Balanced piecewise-linear evaluation at ``ld(0)``, including ramps.

    Extrapolating the first interval lets reordered B-frame decode timestamps
    precede zero without changing their relationship to presentation timestamps.
    """
    def number(value):
        return format(float(value), ".17g")

    def branch(begin, end):
        if end - begin == 1:
            slope = ((output[begin + 1] - output[begin]) /
                     (source[begin + 1] - source[begin]))
            return (f"({number(output[begin])}+(ld(0)-{number(source[begin])})*"
                    f"{number(slope)})")
        middle = (begin + end) // 2
        return (f"if(lt(ld(0),{number(source[middle])}),"
                f"{branch(begin, middle)},{branch(middle, end)})")

    return branch(0, len(source) - 1)


def _packet_filter(mapping) -> str:
    source, output = _mapping_knots(mapping)
    if not math.isclose(float(output[-1]), float(mapping.output_duration),
                        rel_tol=0, abs_tol=1e-8):
        raise ValueError("Thời lượng video đích không khớp đường thời gian.")
    expression = _mapped_clock(source, output)
    pts = f"st(0,PTS*TB);({expression})/TB_OUT"
    dts = f"st(0,DTS*TB);({expression})/TB_OUT"
    # Map both endpoints rather than multiplying duration by one local slope:
    # a frame's display interval can straddle a marker or a smooth speed ramp.
    duration = (f"st(0,(PTS+DURATION)*TB);st(1,{expression});"
                f"st(0,PTS*TB);max(1,(ld(1)-({expression}))/TB_OUT)")
    return (f"setts=pts='{pts}':dts='{dts}':duration='{duration}':"
            "time_base=1/1000000")


def render_video_copy(source, destination, mapping, audio_path=None,
                      check_stop: Callable = lambda: None,
                      progress: Callable = lambda fraction: None) -> str:
    """Copy video bitstreams while applying the shared map to packet times.

    This supports both marker alignment and sampled smooth speed ramps. The
    video codec, frame count, pixel format, dimensions, colors and frame payloads
    remain those of the source. MP4-incompatible source codecs raise an error;
    there is deliberately no fallback that would silently reencode the video.
    The destination must not exist and our reserved file is removed on failure.

    FFmpeg 7.1's ``-/bsf:v`` loads expressions from a file, so thousands of ramp
    knots do not exceed Windows' command-line size limit. PTS and DTS are mapped
    separately to preserve B-frame reordering. No output ``-t``, ``-r`` or video
    filter is used: those could discard frames or impose constant frame timing.
    """
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if not source.is_file():
        raise ValueError("Không tìm thấy file video nguồn.")
    if destination == source or destination.exists():
        raise ValueError("File video xuất đã tồn tại; hãy chọn một tên mới.")
    if destination.suffix.lower() != ".mp4":
        raise ValueError("Video xuất cần có đuôi .mp4.")
    packet_filter = _packet_filter(mapping)
    if audio_path is not None:
        audio_path = Path(audio_path).expanduser().resolve()
        if not audio_path.is_file():
            raise ValueError("Không tìm thấy âm thanh đã xử lý để ghép video.")
    check_stop()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb"):
        pass
    completed = False
    script_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".bsf",
                                         prefix="video_copy_", dir=destination.parent,
                                         delete=False) as script:
            script_path = Path(script.name)
            script.write(packet_filter)
        command = [_ffmpeg_binary(), "-hide_banner", "-loglevel", "warning", "-nostdin",
                   "-y", "-stats_period", "0.2", "-progress", "pipe:1", "-copyts",
                   "-start_at_zero", "-i", str(source)]
        if audio_path is not None:
            command.extend(["-i", str(audio_path)])
        command.extend(["-map", "0:v:0", "-map_metadata", "0", "-map_chapters", "-1",
                        "-c:v", "copy", "-/bsf:v", str(script_path)])
        if audio_path is not None:
            command.extend(["-map", "1:a:0", "-c:a", "copy"])
        else:
            command.append("-an")
        command.extend(["-avoid_negative_ts", "disabled", "-movflags", "+faststart",
                        str(destination)])
        progress(0.0)
        detail = _run_ffmpeg(command, float(mapping.output_duration), check_stop, progress)
        check_stop()
        # The muxer otherwise repairs colliding timestamps while claiming
        # success. Extremely compressed sub-microsecond maps must surface this
        # limitation instead of exporting timings different from the user map.
        if "non-monoton" in detail.lower() or "invalid dts" in detail.lower():
            raise ValueError("Các mốc quá sát nhau để giữ nguyên khung hình video. "
                             "Hãy tăng khoảng cách giữa các mốc đích.")
        completed = True
    finally:
        if script_path is not None:
            script_path.unlink(missing_ok=True)
        if not completed:
            destination.unlink(missing_ok=True)
    return "copy"
