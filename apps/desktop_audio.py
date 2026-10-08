"""Native Windows PCM playback, including continuous streamed chunks.

waveOut queues buffers without restarting the audio device between chunks.
Other platforms can use the optional sounddevice package.
"""
from __future__ import annotations

import ctypes
import os
import threading
import time
import uuid

import numpy as np
import soundfile as sf


class PlaybackStopped(RuntimeError):
    pass


class MicrophoneRecorder:
    """Windows microphone capture through MCI; called only on an explicit click."""
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("Ghi micro tích hợp hiện hỗ trợ Windows. Có thể nhập audio đã ghi từ file.")
        self._dll = ctypes.WinDLL("winmm")
        self._dll.mciSendStringW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_void_p]
        self._dll.mciGetErrorStringW.argtypes = [ctypes.c_uint32, ctypes.c_wchar_p, ctypes.c_uint32]
        self.alias = "vieneu_" + uuid.uuid4().hex[:12]
        self.opened = False

    def _command(self, text):
        code = self._dll.mciSendStringW(text, None, 0, None)
        if code:
            message = ctypes.create_unicode_buffer(512)
            self._dll.mciGetErrorStringW(code, message, len(message))
            raise RuntimeError(f"Microphone: {message.value or code}")

    def start(self):
        self._command(f"open new type waveaudio alias {self.alias}")
        self.opened = True
        try:
            self._command(f"set {self.alias} channels 1 bitspersample 16 samplespersec 48000")
            self._command(f"record {self.alias}")
        except Exception:
            self.close()
            raise

    def save(self, path):
        from pathlib import Path
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._command(f"stop {self.alias}")
            self._command(f'save {self.alias} "{path}"')
        finally:
            self.close()
        return path

    def close(self):
        if self.opened:
            try:
                self._command(f"close {self.alias}")
            finally:
                self.opened = False


class WavePlayer:
    def __init__(self):
        self.stopped = threading.Event()
        self._handle = None
        self._buffers = []
        self._stream = None
        self._rate = None
        self._clock_lock = threading.RLock()
        self._written_frames = 0
        self._finished_position = 0.0
        self._stream_origin = None

    @property
    def position_seconds(self):
        """Read the device playback clock, including pauses caused by underruns."""
        with self._clock_lock:
            if self._rate is None:
                return 0.0
            if self._handle is not None:
                # MMTIME: UINT wType followed by an eight-byte union.
                class MultimediaTime(ctypes.Structure):
                    _fields_ = [("kind", ctypes.c_uint32), ("value", ctypes.c_uint32),
                                ("extra", ctypes.c_uint32)]
                position = MultimediaTime(2, 0, 0)  # TIME_SAMPLES
                code = self._winmm.waveOutGetPosition(self._handle, ctypes.byref(position),
                                                     ctypes.sizeof(position))
                if code:
                    raise RuntimeError(f"Không đọc được vị trí phát âm thanh (mã {code}).")
                if position.kind == 2:
                    seconds = position.value / self._rate
                elif position.kind == 1:  # TIME_MS
                    seconds = position.value / 1000
                elif position.kind == 4:  # TIME_BYTES, mono int16
                    seconds = position.value / (self._rate * 2)
                else:
                    raise RuntimeError("Thiết bị không hỗ trợ đồng hồ phát âm thanh.")
                return min(self._written_frames / self._rate, seconds)
            if self._stream is not None and self._stream_origin is not None:
                seconds = max(0, self._stream.time - self._stream_origin - self._stream.latency)
                return min(self._written_frames / self._rate, seconds)
            return self._finished_position

    def stop(self):
        # Only signal here: the playback worker owns and releases native buffers.
        self.stopped.set()

    def _check(self):
        if self.stopped.is_set():
            raise PlaybackStopped("Đã dừng phát audio.")

    def _open(self, rate):
        self._rate = rate
        if os.name != "nt":
            try:
                import sounddevice
            except ImportError as exc:
                raise RuntimeError("Phát audio ngoài Windows cần cài sounddevice.") from exc
            self._stream = sounddevice.RawOutputStream(samplerate=rate, channels=1, dtype="int16")
            self._stream.start()
            self._stream_origin = self._stream.time
            return

        class WaveFormat(ctypes.Structure):
            _fields_ = [("tag", ctypes.c_uint16), ("channels", ctypes.c_uint16),
                        ("rate", ctypes.c_uint32), ("bytes_per_sec", ctypes.c_uint32),
                        ("align", ctypes.c_uint16), ("bits", ctypes.c_uint16),
                        ("extra", ctypes.c_uint16)]

        class WaveHeader(ctypes.Structure):
            _fields_ = [("data", ctypes.c_void_p), ("length", ctypes.c_uint32),
                        ("recorded", ctypes.c_uint32), ("user", ctypes.c_size_t),
                        ("flags", ctypes.c_uint32), ("loops", ctypes.c_uint32),
                        ("next", ctypes.c_void_p), ("reserved", ctypes.c_size_t)]

        self._header_cls = WaveHeader
        self._winmm = ctypes.WinDLL("winmm")
        self._winmm.waveOutOpen.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32,
                                          ctypes.POINTER(WaveFormat), ctypes.c_size_t,
                                          ctypes.c_size_t, ctypes.c_uint32]
        for name in ("waveOutPrepareHeader", "waveOutUnprepareHeader", "waveOutWrite"):
            getattr(self._winmm, name).argtypes = [ctypes.c_void_p, ctypes.POINTER(WaveHeader), ctypes.c_uint32]
        for name in ("waveOutReset", "waveOutClose"):
            getattr(self._winmm, name).argtypes = [ctypes.c_void_p]
        self._winmm.waveOutGetPosition.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        fmt = WaveFormat(1, 1, rate, rate * 2, 2, 16, 0)
        handle = ctypes.c_void_p()
        code = self._winmm.waveOutOpen(ctypes.byref(handle), 0xFFFFFFFF, ctypes.byref(fmt), 0, 0, 0)
        if code:
            raise RuntimeError(f"Không mở được thiết bị âm thanh Windows (mã {code}).")
        self._handle = handle

    def _reap(self):
        keep = []
        for header, buffer in self._buffers:
            if header.flags & 1:  # WHDR_DONE
                code = self._winmm.waveOutUnprepareHeader(self._handle, ctypes.byref(header), ctypes.sizeof(header))
                if code:
                    keep.append((header, buffer))
            else:
                keep.append((header, buffer))
        self._buffers = keep

    def write(self, audio, rate=48000):
        self._check()
        if self._rate is None:
            self._open(rate)
        if self._rate != rate:
            raise ValueError("Tần số audio thay đổi trong khi đang phát.")
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        # Limit native queue depth and buffer duration to bound stop latency.
        for start in range(0, len(wav), max(1, rate // 5)):
            self._check()
            data = (wav[start:start + rate // 5].clip(-1, 1) * 32767).astype("<i2").tobytes()
            if self._stream is not None:
                self._stream.write(data)
                self._written_frames += len(data) // 2
                continue
            while len(self._buffers) >= 6:
                self._check()
                self._reap()
                time.sleep(0.01)
            self._reap()
            buffer = ctypes.create_string_buffer(data, len(data))
            header = self._header_cls(data=ctypes.addressof(buffer), length=len(data))
            code = self._winmm.waveOutPrepareHeader(self._handle, ctypes.byref(header), ctypes.sizeof(header))
            if code:
                raise RuntimeError(f"Không chuẩn bị được audio (mã {code}).")
            self._buffers.append((header, buffer))
            code = self._winmm.waveOutWrite(self._handle, ctypes.byref(header), ctypes.sizeof(header))
            if code:
                raise RuntimeError(f"Không phát được audio (mã {code}).")
            self._written_frames += len(data) // 2

    def finish(self):
        while self._buffers:
            self._check()
            self._reap()
            time.sleep(0.01)
        if self._stream is not None:
            self._stream.stop()
        if self._rate:
            self._finished_position = self._written_frames / self._rate

    def close(self):
        with self._clock_lock:
            self._close_device()

    def _close_device(self):
        try:
            self._finished_position = max(self._finished_position, self.position_seconds)
        except RuntimeError:
            pass
        if self._stream is not None:
            self._stream.abort()
            self._stream.close()
            self._stream = None
        if self._handle is not None:
            self._winmm.waveOutReset(self._handle)
            for header, buffer in self._buffers:
                self._winmm.waveOutUnprepareHeader(self._handle, ctypes.byref(header), ctypes.sizeof(header))
            self._winmm.waveOutClose(self._handle)
            self._buffers.clear()
            self._handle = None

    def play_file(self, path):
        try:
            with sf.SoundFile(path) as file:
                while True:
                    self._check()
                    chunk = file.read(file.samplerate // 5, dtype="float32", always_2d=True)
                    if not chunk.size:
                        break
                    self.write(chunk.mean(axis=1), file.samplerate)
            self.finish()
        finally:
            self.close()
