"""Stream the preview's mapped audio straight to a checked AAC encoder."""
from __future__ import annotations

from contextlib import closing
import math
import os
from pathlib import Path
import subprocess
import tempfile
import uuid

import numpy as np


_RATE = 48000
_BLOCK_SAMPLES = 24000


def encode_mapped_audio(audio, filename, mapping, check_stop=lambda: None,
                        progress=lambda fraction: None) -> None:
    """Encode exact mapped samples without a WAV file or another audio decoder.

    The caller owns ``audio``. Only a few seconds of mapped float audio and a
    half-second PCM block are resident. An unsuccessful export leaves neither
    the destination nor an encoder process behind.
    """
    from moviepy.config import FFMPEG_BINARY
    from moviepy.tools import cross_platform_popen_params
    from apps.video_editor import _ramped_audio_chunks

    destination = Path(filename).expanduser().resolve()
    if destination.exists():
        raise ValueError("Chọn file audio mới; không được ghi đè file đã có.")
    duration = float(mapping.output_duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Thời lượng audio đầu ra không hợp lệ.")
    channels = int(audio.nchannels)
    if channels not in (1, 2):
        raise ValueError("Audio video cần là mono hoặc stereo.")
    check_stop()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.parent / f".audio_{uuid.uuid4().hex}.m4a"
    total = max(1, round(duration * _RATE))
    consumed = 0
    process = None
    # A file keeps error output from filling a pipe and blocking the encoder.
    with tempfile.TemporaryFile(mode="w+b") as errors:
        try:
            command = [
                FFMPEG_BINARY, "-hide_banner", "-loglevel", "error", "-n",
                "-f", "s32le", "-acodec", "pcm_s32le", "-ar", str(_RATE),
                "-ac", str(channels), "-i", "pipe:0", "-vn",
                "-c:a", "aac", "-b:a", "192k", "-ar", str(_RATE),
                "-f", "ipod", str(partial),
            ]
            process = subprocess.Popen(command, **cross_platform_popen_params({
                "stdin": subprocess.PIPE, "stdout": subprocess.DEVNULL,
                "stderr": errors,
            }))
            chunks = _ramped_audio_chunks(
                audio, mapping, check_stop=check_stop, workers=max(1, min(4, os.cpu_count() or 1)),
            )
            with closing(chunks):
                for chunk in chunks:
                    check_stop()
                    chunk = np.asarray(chunk)
                    if (chunk.ndim != 2 or chunk.shape[1] != channels or not len(chunk)
                            or not np.issubdtype(chunk.dtype, np.floating)
                            or not np.isfinite(chunk).all()):
                        raise ValueError("Audio đầu ra không hợp lệ hoặc có mẫu không hữu hạn.")
                    if consumed + len(chunk) > total:
                        raise ValueError("Thời lượng audio đầu ra không khớp đường thời gian video.")
                    for first in range(0, len(chunk), _BLOCK_SAMPLES):
                        check_stop()
                        block = chunk[first:first + _BLOCK_SAMPLES]
                        # Match MoviePy's 32-bit PCM quantization, avoiding the old
                        # temporary WAV reader's extra conversion to 16-bit PCM.
                        pcm = (np.clip(block, -0.99, 0.99) * (2 ** 31)).astype("<i4")
                        try:
                            process.stdin.write(pcm.tobytes())
                        except OSError as exc:
                            raise ValueError("FFmpeg không mã hóa được audio video.") from exc
                        consumed += len(block)
                        progress(consumed / total)
            if consumed != total:
                raise ValueError("Thời lượng audio đầu ra không khớp đường thời gian video.")
            check_stop()
            process.stdin.close()
            process.stdin = None
            while True:
                check_stop()
                try:
                    returncode = process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    pass
            if returncode:
                errors.seek(0, os.SEEK_END)
                errors.seek(max(0, errors.tell() - 4096))
                detail = errors.read().decode("utf-8", errors="replace").strip()
                raise ValueError(f"FFmpeg không mã hóa được audio video: {detail or returncode}")
            check_stop()
            if not partial.is_file() or not partial.stat().st_size:
                raise ValueError("FFmpeg không tạo được audio đầu ra.")
            if destination.exists():
                raise ValueError("Chọn file audio mới; không được ghi đè file đã có.")
            # Windows rename refuses an existing destination, including one
            # created while the encoder was running.
            os.rename(partial, destination)
        finally:
            if process is not None:
                _release_encoder(process)
            partial.unlink(missing_ok=True)


def _release_encoder(process) -> None:
    """Reap a failed/cancelled encoder before removing its temporary file."""
    try:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    finally:
        for name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, name, None)
            if stream is not None and not stream.closed:
                stream.close()
