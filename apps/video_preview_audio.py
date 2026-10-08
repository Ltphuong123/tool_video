"""Bounded audio playback using the same retimed samples as video export."""
from __future__ import annotations

import threading

import numpy as np

from apps.desktop_audio import PlaybackStopped, WavePlayer


def _open_audio(path):
    from apps.video_editor import _guard_reader_cleanup, require_moviepy
    _, AudioFileClip, _ = require_moviepy()
    clip = AudioFileClip(str(path), fps=48000)
    _guard_reader_cleanup(clip)
    return clip


class PreviewAudio:
    """A single playback session; seeking starts a new, independently owned reader."""

    def __init__(self, path, mapping, start_output, player_factory=None, opener=None):
        self.path = path
        self.mapping = mapping
        self.start_output = float(start_output)
        self.error = None
        self.done = False
        self.stopped = threading.Event()
        self.player = (player_factory or WavePlayer)()
        self._opener = opener or _open_audio
        self._last_position = self.start_output
        self.thread = threading.Thread(target=self._run, name="video-audio", daemon=True)
        self.thread.start()

    @property
    def output_position(self):
        if self.done and self.error is None and not self.stopped.is_set():
            return self.mapping.output_duration
        try:
            self._last_position = min(self.mapping.output_duration,
                                      self.start_output + self.player.position_seconds)
        except Exception as exc:
            self.error = str(exc)
            self.stop()
        return self._last_position

    def stop(self):
        self.stopped.set()
        self.player.stop()

    def _check(self):
        if self.stopped.is_set():
            raise PlaybackStopped("Đã dừng âm thanh xem trước.")

    def _run(self):
        from apps.video_editor import _ramped_audio_chunks
        clip = None
        try:
            self._check()
            clip = self._opener(self.path)
            self._check()
            chunks = _ramped_audio_chunks(clip, self.mapping, check_stop=self._check,
                                          start_output=self.start_output)
            try:
                for chunk in chunks:
                    self._check()
                    self.player.write(np.asarray(chunk, dtype=np.float32).mean(axis=1), 48000)
                self.player.finish()
            finally:
                chunks.close()
        except PlaybackStopped:
            pass
        except Exception as exc:
            self.error = str(exc)
        finally:
            try:
                self.player.close()
            except Exception as exc:
                self.error = self.error or str(exc)
            finally:
                try:
                    if clip is not None:
                        clip.close()
                except Exception as exc:
                    self.error = self.error or str(exc)
                finally:
                    self.done = True
