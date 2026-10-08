"""Embedded video and audio preview with asynchronous MoviePy decoders.

Tk and ImageTk are only touched on the UI thread. The decoder owns its reader,
coalesces seeks and closes stale readers on that same worker thread.
"""
from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np


def _open_clip(path):
    # Keep MoviePy and Pillow optional until a video is actually opened.
    from apps.video_editor import _guard_reader_cleanup, require_moviepy

    VideoFileClip, _, _ = require_moviepy()
    clip = VideoFileClip(str(path), audio=False, target_resolution=(None, 540),
                         resize_algorithm="bilinear")
    _guard_reader_cleanup(clip)
    return clip


def _clip_info(clip):
    duration, fps = float(clip.duration), float(clip.fps)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Không đọc được thời lượng hợp lệ từ video nguồn.")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Không đọc được FPS hợp lệ từ video nguồn.")
    reader = getattr(clip, "reader", None)
    metadata = getattr(reader, "infos", {})
    width, height = metadata.get("video_size", (clip.w, clip.h))
    if abs(float(metadata.get("video_rotation", 0))) in (90, 270):
        width, height = height, width
    return {"duration": duration, "fps": fps, "width": int(width), "height": int(height),
            "has_audio": bool(metadata.get("audio_found", getattr(clip, "audio", None) is not None))}


def _close_clip(clip):
    if clip is not None:
        try:
            clip.close()
        except Exception:
            # The guarded MoviePy reader still closes its pipes if FFmpeg exits.
            pass


class _PreviewDecoder:
    """One reader owner; pending frame requests replace each other."""

    def __init__(self, opener=_open_clip):
        self._opener = opener
        self._condition = threading.Condition()
        self._generation = 0
        self._pending_open = None
        self._pending_frame = None
        self._events = deque()
        self._closed = False
        self.thread = threading.Thread(target=self._run, name="video-preview", daemon=True)
        self.thread.start()

    def open(self, generation, path):
        with self._condition:
            if self._closed:
                return
            self._generation = generation
            self._pending_open = (generation, path)
            self._pending_frame = None
            self._events.clear()
            self._condition.notify()

    def request(self, generation, revision, seconds):
        with self._condition:
            if not self._closed and generation == self._generation:
                self._pending_frame = (generation, revision, seconds)
                self._condition.notify()

    def drain(self):
        with self._condition:
            events = list(self._events)
            self._events.clear()
            return events

    def close(self):
        with self._condition:
            self._closed = True
            self._pending_open = self._pending_frame = None
            self._events.clear()
            self._condition.notify()

    def _current(self, generation):
        with self._condition:
            return not self._closed and generation == self._generation

    def _emit(self, event):
        with self._condition:
            if self._closed or event[1] != self._generation:
                return
            if event[0] == "frame":
                self._events = deque(item for item in self._events if item[0] != "frame")
            self._events.append(event)

    @staticmethod
    def _frame(clip, seconds, duration, fps):
        # A source timestamp at duration is outside the final frame's interval.
        seconds = min(max(0, seconds), max(0, duration - min(1 / fps, 0.001)))
        frame = np.asarray(clip.get_frame(seconds))
        if frame.ndim != 3 or frame.shape[2] not in (3, 4) or not frame.size:
            raise ValueError("Không đọc được khung hình video hợp lệ.")
        return np.array(frame, dtype=np.uint8, copy=True)

    def _run(self):
        clip = None
        active_generation = None
        info = None
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._pending_open is not None
                                             or self._pending_frame is not None)
                    if self._closed:
                        break
                    opening = self._pending_open
                    if opening is not None:
                        self._pending_open = None
                        request = None
                    else:
                        request = self._pending_frame
                        self._pending_frame = None
                if opening is not None:
                    generation, path = opening
                    _close_clip(clip)
                    clip, active_generation, info = None, None, None
                    if path is None:
                        continue
                    try:
                        clip = self._opener(path)
                        if not self._current(generation):
                            _close_clip(clip)
                            clip = None
                            continue
                        info = _clip_info(clip)
                        frame = self._frame(clip, 0, info["duration"], info["fps"])
                        active_generation = generation
                        self._emit(("loaded", generation, info, frame))
                    except Exception as exc:
                        _close_clip(clip)
                        clip, active_generation, info = None, None, None
                        self._emit(("error", generation, str(exc)))
                elif request is not None and clip is not None:
                    generation, revision, seconds = request
                    if generation != active_generation or not self._current(generation):
                        continue
                    try:
                        frame = self._frame(clip, seconds, info["duration"], info["fps"])
                        self._emit(("frame", generation, revision, seconds, frame))
                    except Exception as exc:
                        _close_clip(clip)
                        clip, active_generation, info = None, None, None
                        self._emit(("error", generation, str(exc)))
        finally:
            _close_clip(clip)


class VideoPreview(ttk.Frame):
    """A canvas preview; its public timestamps use original source seconds.

    Playback advances on the edited output clock and maps back to the source,
    including speed ramps. When sound is enabled, the audio device drives time.
    ``close`` is terminal; ``unload`` keeps the widget available for another file.
    """

    def __init__(self, parent, on_position=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.ready = False
        self.duration = self.fps = self.position = 0.0
        self.playing = False
        self.on_position = on_position
        self._decoder = None
        self._generation = self._revision = 0
        self._closed = False
        self._mapping = None
        self._audio = None
        self._path = None
        self.has_audio = False
        self.audio_enabled = True
        self._segments = ()
        self._anchor_time = self._anchor_output = 0.0
        self._next_frame_time = 0.0
        self._on_loaded = self._on_error = None
        self._image = self._photo = None
        self._resize_after = None
        self.canvas = tk.Canvas(self, background="#101722", highlightthickness=0,
                                width=480, height=270)
        self.canvas.pack(fill="both", expand=True)
        self._image_item = self.canvas.create_image(0, 0, anchor="center")
        self._message_item = self.canvas.create_text(0, 0, fill="#aebed0",
                                                    text="Chọn video để xem trước")
        self.canvas.bind("<Configure>", self._resized)
        self.bind("<Destroy>", self._destroyed, add="+")
        self._poll_after = self.after(20, self._tick)

    def open(self, path, on_loaded=None, on_error=None):
        if self._closed:
            raise RuntimeError("Khung xem trước đã đóng.")
        self.unload()
        self._on_loaded, self._on_error = on_loaded, on_error
        self._show_message("Đang mở video…")
        if self._decoder is None:
            self._decoder = _PreviewDecoder()
        self._path = Path(path).expanduser().resolve()
        self._decoder.open(self._generation, self._path)

    def unload(self):
        if self._closed:
            return
        self._stop_audio()
        self._path = None
        self.has_audio = False
        self.playing = self.ready = False
        self.duration = self.fps = self.position = 0.0
        self._generation += 1
        self._revision += 1
        self._mapping = None
        self._segments = ()
        self._on_loaded = self._on_error = None
        self._image = self._photo = None
        self.canvas.itemconfigure(self._image_item, image="")
        self._show_message("Chọn video để xem trước")
        if self._decoder is not None:
            self._decoder.open(self._generation, None)
        self._notify_position()

    def set_segments(self, segments):
        from apps.video_editor import build_multi_speed_time_map

        segments = tuple(segments)
        mapping = build_multi_speed_time_map(self.duration, segments) if self.ready else None
        if self.playing:
            self.position = self._clock_position()
        self._segments, self._mapping = segments, mapping
        if self.playing:
            self._anchor_output = float(mapping.output_time(self.position))
            self._anchor_time = time.monotonic()
            self._start_audio()
        self._notify_position()

    def seek(self, source_seconds):
        if self._closed or not self.ready:
            return
        seconds = float(source_seconds)
        if not math.isfinite(seconds):
            raise ValueError("Mốc xem trước phải là số hữu hạn.")
        self.position = min(self.duration, max(0.0, seconds))
        self._revision += 1
        self._anchor_output = float(self._mapping.output_time(self.position))
        self._anchor_time = time.monotonic()
        self._next_frame_time = self._anchor_time
        if self.playing:
            self._start_audio()
        self._request_frame()
        self._notify_position()

    def set_time_map(self, mapping):
        """Use an exact marker mapping without rebuilding it from speed controls."""
        if not self.ready or self._closed:
            raise ValueError("Hãy mở video trước khi căn các mốc thời gian.")
        if not math.isclose(mapping.source_duration, self.duration, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Đường thời gian không khớp thời lượng video đang mở.")
        if self.playing:
            self.position = self._clock_position()
        self._segments = tuple(mapping.segments)
        self._mapping = mapping
        if self.playing:
            self._anchor_output = float(mapping.output_time(self.position))
            self._anchor_time = time.monotonic()
            self._start_audio()
        self._notify_position()

    def play(self):
        if self._closed or not self.ready or self.playing:
            return
        if self.position >= self.duration:
            self.seek(0)
        self._anchor_output = float(self._mapping.output_time(self.position))
        self._anchor_time = time.monotonic()
        self._next_frame_time = self._anchor_time
        self.playing = True
        self._start_audio()
        self._notify_position()

    def pause(self):
        if self._closed or not self.playing:
            return
        self.position = self._clock_position()
        self.playing = False
        self._stop_audio()
        self._revision += 1
        self._request_frame()
        self._notify_position()

    def _clock_position(self):
        if self._audio is not None:
            return float(self._mapping.source_time(self._audio.output_position))
        elapsed = max(0.0, time.monotonic() - self._anchor_time)
        output = min(self._mapping.output_duration, self._anchor_output + elapsed)
        return float(self._mapping.source_time(output))

    def _stop_audio(self):
        if self._audio is not None:
            self._audio.stop()
            self._audio = None

    def _start_audio(self):
        self._stop_audio()
        if self.has_audio and self.audio_enabled and self.playing:
            from apps.video_preview_audio import PreviewAudio
            self._audio = PreviewAudio(self._path, self._mapping,
                                       float(self._mapping.output_time(self.position)))

    def set_audio_enabled(self, enabled):
        if self.playing:
            self.position = self._clock_position()
        self.audio_enabled = bool(enabled)
        if self.playing:
            self._anchor_output = float(self._mapping.output_time(self.position))
            self._anchor_time = time.monotonic()
            self._start_audio()
        self._notify_position()

    def _request_frame(self):
        if self._decoder is not None and self.ready:
            self._decoder.request(self._generation, self._revision, self.position)

    def _notify_position(self):
        if not self._closed and self.on_position is not None:
            self.on_position(self.position, self.playing)

    def _tick(self):
        self._poll_after = None
        if self._closed:
            return
        try:
            events = self._decoder.drain() if self._decoder is not None else ()
            for event in events:
                if event[1] != self._generation:
                    continue
                if event[0] == "loaded":
                    from apps.video_editor import build_multi_speed_time_map

                    info, frame = event[2:]
                    self.duration, self.fps = info["duration"], info["fps"]
                    self.has_audio = info["has_audio"]
                    try:
                        self._mapping = build_multi_speed_time_map(self.duration, self._segments)
                    except ValueError:
                        self._segments = ()
                        self._mapping = build_multi_speed_time_map(self.duration, ())
                    self.ready = True
                    self._show_frame(frame)
                    self._notify_position()
                    if self._on_loaded is not None:
                        self._on_loaded(dict(info))
                elif event[0] == "frame" and event[2] == self._revision:
                    self._show_frame(event[4])
                elif event[0] == "error":
                    self._stop_audio()
                    self.playing = self.ready = False
                    self._show_message("Không mở được video")
                    self._notify_position()
                    if self._on_error is not None:
                        self._on_error(event[2])
            if self.playing and self.ready:
                if self._audio is not None and self._audio.error is not None:
                    error = self._audio.error
                    self.pause()
                    if self._on_error is not None:
                        self._on_error("Không phát được âm thanh: " + error)
                    return
                now = time.monotonic()
                self.position = self._clock_position()
                if self.position >= self.duration:
                    self.position = self.duration
                    self.playing = False
                    self._stop_audio()
                if now >= self._next_frame_time or not self.playing:
                    self._request_frame()
                    self._next_frame_time = now + 1 / min(30.0, self.fps)
                    self._notify_position()
        finally:
            if not self._closed:
                self._poll_after = self.after(15, self._tick)

    def _show_message(self, message):
        self.canvas.itemconfigure(self._message_item, text=message)
        self.canvas.coords(self._message_item, self.canvas.winfo_width() / 2,
                           self.canvas.winfo_height() / 2)

    def _show_frame(self, frame):
        from PIL import Image

        self._image = Image.fromarray(frame)
        self.canvas.itemconfigure(self._message_item, text="")
        self._draw()

    def _resized(self, event):
        if self._closed:
            return
        self.canvas.coords(self._message_item, event.width / 2, event.height / 2)
        if self._resize_after is None:
            self._resize_after = self.after_idle(self._draw)

    def _draw(self):
        if self._resize_after is not None:
            try:
                self.after_cancel(self._resize_after)
            except tk.TclError:
                pass
        self._resize_after = None
        if self._closed or self._image is None:
            return
        from PIL import Image, ImageTk

        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 2 or height < 2:
            return
        scale = min(width / self._image.width, height / self._image.height)
        size = (max(1, round(self._image.width * scale)), max(1, round(self._image.height * scale)))
        picture = self._image.resize(size, Image.Resampling.BILINEAR)
        self._photo = ImageTk.PhotoImage(picture, master=self.canvas)
        self.canvas.coords(self._image_item, width / 2, height / 2)
        self.canvas.itemconfigure(self._image_item, image=self._photo)

    def close(self):
        if self._closed:
            return
        self._stop_audio()
        self._closed = True
        self.playing = self.ready = False
        for identifier in (self._poll_after, self._resize_after):
            if identifier is not None:
                try:
                    self.after_cancel(identifier)
                except tk.TclError:
                    pass
        self._poll_after = self._resize_after = None
        self._on_loaded = self._on_error = self.on_position = None
        self._image = self._photo = None
        if self._decoder is not None:
            # Never join a decoder on Tk's thread: FFmpeg may still be opening.
            self._decoder.close()

    def _destroyed(self, event):
        if event.widget is self:
            self.close()
