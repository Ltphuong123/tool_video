"""Audio preview timing, pitch, seeking and ownership without a sound device."""
import importlib.util
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from apps.desktop_audio import WavePlayer
from apps.video_editor import SpeedSegment, _ramped_audio_chunks, build_multi_speed_time_map
from apps.video_preview import VideoPreview
from apps.video_preview_audio import PreviewAudio


class ToneAudio:
    nchannels = 2

    def __init__(self, duration=12):
        self.duration = duration
        self.calls = []
        self.closed = False
        self.close_thread = None

    def get_frame(self, times):
        self.calls.append((times[0], times[-1]))
        return np.column_stack((.2 * np.sin(2 * np.pi * 440 * times),
                                .1 * np.sin(2 * np.pi * 220 * times)))

    def close(self):
        self.closed = True
        self.close_thread = threading.get_ident()


class FakePlayer:
    def __init__(self):
        self.chunks = []
        self.closed = self.stopped = False
        self.position_seconds = 0

    def write(self, audio, rate):
        self.chunks.append(np.asarray(audio).copy())
        self.position_seconds += len(audio) / rate

    def stop(self):
        self.stopped = True

    def finish(self):
        pass

    def close(self):
        self.closed = True


class PreviewAudioTests(unittest.TestCase):
    def test_identity_seek_starts_near_seek_and_needs_no_rubberband(self):
        mapping = build_multi_speed_time_map(12, [])
        audio = ToneAudio()
        with patch("apps.speech_speed.require_rubberband", side_effect=AssertionError("unused")):
            chunks = list(_ramped_audio_chunks(audio, mapping, start_output=9.25))
        result = np.concatenate(chunks)
        self.assertEqual(result.shape, (round((12 - 9.25) * 48000), 2))
        self.assertGreaterEqual(audio.calls[0][0], 9.20)
        self.assertTrue(np.isfinite(result).all())

    @unittest.skipUnless(importlib.util.find_spec("pedalboard"), "optional Rubber Band not installed")
    def test_audio_after_seek_keeps_pitch_and_exact_multisegment_duration(self):
        mapping = build_multi_speed_time_map(12, [SpeedSegment(2, 4, 2, .25),
                                                SpeedSegment(7, 10, .5, .5)])
        start = float(mapping.output_time(3))
        result = np.concatenate(list(_ramped_audio_chunks(ToneAudio(), mapping, start_output=start)))
        self.assertEqual(len(result), round(mapping.output_duration * 48000) - round(start * 48000))
        self.assertTrue(np.isfinite(result).all())
        begin = round((float(mapping.output_time(8)) - start) * 48000)
        end = round((float(mapping.output_time(9)) - start) * 48000)
        signal = result[begin:end, 0]
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
        frequency = np.fft.rfftfreq(len(signal), 1 / 48000)[spectrum.argmax()]
        self.assertAlmostEqual(frequency, 440, delta=3)

    def test_worker_streams_only_remaining_audio_and_owns_reader_cleanup(self):
        mapping = build_multi_speed_time_map(3, [])
        audio, player = ToneAudio(3), FakePlayer()
        session = PreviewAudio("source.mp4", mapping, 1.25,
                               player_factory=lambda: player, opener=lambda path: audio)
        session.thread.join(3)
        self.assertFalse(session.thread.is_alive())
        self.assertIsNone(session.error)
        self.assertTrue(session.done and player.closed and audio.closed)
        self.assertEqual(sum(len(chunk) for chunk in player.chunks), round(1.75 * 48000))
        self.assertEqual(audio.close_thread, session.thread.ident)
        self.assertEqual(session.output_position, mapping.output_duration)

    def test_stop_during_reader_open_discards_audio_and_releases_reader(self):
        entered, release = threading.Event(), threading.Event()
        audio, player = ToneAudio(3), FakePlayer()
        def opener(path):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("Reader was not released")
            return audio
        session = PreviewAudio("source.mp4", build_multi_speed_time_map(3, []), 0,
                               player_factory=lambda: player, opener=opener)
        try:
            self.assertTrue(entered.wait(3))
            session.stop()
        finally:
            release.set()
            session.thread.join(3)
        self.assertEqual(player.chunks, [])
        self.assertTrue(player.closed and player.stopped and audio.closed and session.done)
        self.assertIsNone(session.error)

    def test_reader_error_is_reported_and_player_is_closed(self):
        player = FakePlayer()
        def opener(path):
            raise OSError("Missing audio decoder")
        session = PreviewAudio("source.mp4", build_multi_speed_time_map(3, []), 0,
                               player_factory=lambda: player, opener=opener)
        session.thread.join(3)
        self.assertIn("Missing audio decoder", session.error)
        self.assertTrue(player.closed and session.done)

    def test_picture_clock_uses_device_position_instead_of_wall_clock(self):
        preview = VideoPreview.__new__(VideoPreview)
        preview._mapping = build_multi_speed_time_map(12, [SpeedSegment(2, 6, 2, 0)])
        preview._audio = SimpleNamespace(output_position=3.25)
        with patch("apps.video_preview.time.monotonic", return_value=999999):
            self.assertEqual(preview._clock_position(), preview._mapping.source_time(3.25))

    def test_muting_and_unmuting_restart_at_the_current_edited_position(self):
        preview = VideoPreview.__new__(VideoPreview)
        preview._mapping = build_multi_speed_time_map(12, [SpeedSegment(2, 6, 2, 0)])
        old_session = SimpleNamespace(output_position=3.25, stop=Mock())
        preview._audio = old_session
        preview._closed = False
        preview.ready = preview.has_audio = preview.playing = preview.audio_enabled = True
        preview.on_position = None
        preview._path = "source.mp4"
        new_session = SimpleNamespace(output_position=3.25, stop=Mock())
        with patch("apps.video_preview.time.monotonic", return_value=100), \
             patch("apps.video_preview_audio.PreviewAudio", return_value=new_session) as create:
            preview.set_audio_enabled(False)
            old_session.stop.assert_called_once()
            self.assertIsNone(preview._audio)
            self.assertFalse(preview.audio_enabled)
            self.assertEqual(preview.position, preview._mapping.source_time(3.25))
            create.assert_not_called()
            preview.set_audio_enabled(True)
            create.assert_called_once_with("source.mp4", preview._mapping, 3.25)
            self.assertIs(preview._audio, new_session)

    def test_wave_player_clock_accepts_sample_millisecond_and_byte_formats(self):
        player = WavePlayer()
        player._handle = object()
        player._rate = 48000
        player._written_frames = 48000
        for kind, value in ((2, 24000), (1, 500), (4, 48000)):
            def read_position(handle, pointer, size, kind=kind, value=value):
                self.assertEqual(size, 12)
                pointer._obj.kind, pointer._obj.value = kind, value
                return 0
            player._winmm = SimpleNamespace(waveOutGetPosition=read_position)
            with self.subTest(kind=kind):
                self.assertEqual(player.position_seconds, .5)


if __name__ == "__main__":
    unittest.main()
