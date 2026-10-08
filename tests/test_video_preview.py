"""Preview reader ownership and timing without a display or real decoder."""
from pathlib import Path
import importlib.util
import tempfile
import threading
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from apps.video_editor import SpeedSegment, build_multi_speed_time_map
from apps.video_preview import VideoPreview, _PreviewDecoder


class FakeClip:
    duration = 12.0
    fps = 24.0
    w, h = 96, 54
    audio = None

    def __init__(self, block_at=None, entered=None, release=None):
        self.reader = SimpleNamespace(infos={"video_size": [1920, 1080], "audio_found": True})
        self.block_at = block_at
        self.entered, self.release = entered, release
        self.calls = []
        self.closed = False
        self.owner_threads = []

    def get_frame(self, seconds):
        self.calls.append(seconds)
        self.owner_threads.append(threading.get_ident())
        if seconds == self.block_at:
            self.entered.set()
            if not self.release.wait(3):
                raise TimeoutError("Test did not release decoder")
        return np.full((54, 96, 3), round(seconds), dtype=np.uint8)

    def close(self):
        self.closed = True
        self.owner_threads.append(threading.get_ident())


@unittest.skipUnless(importlib.util.find_spec("moviepy"), "optional MoviePy not installed")
class EmbeddedPreviewTests(unittest.TestCase):
    def test_real_video_displays_seeks_plays_the_edited_clock_and_releases_reader(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display is unavailable: {exc}")
        root.attributes("-alpha", 0.0)
        root.geometry("640x360")
        self.addCleanup(root.destroy)
        from moviepy import VideoClip

        preview = VideoPreview(root)
        preview.pack(fill="both", expand=True)
        self.addCleanup(preview.close)
        root.update()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "preview.mp4"
            clip = VideoClip(lambda t: np.full((36, 64, 3),
                                              (int(t * 120), 50, 150), dtype=np.uint8), duration=0.8)
            try:
                clip.with_fps(10).write_videofile(str(source), codec="libx264", audio=False,
                                                preset="ultrafast", threads=1, logger=None)
            finally:
                clip.close()
            errors, loaded = [], []
            preview.open(source, on_loaded=loaded.append, on_error=errors.append)
            def wait_until(predicate):
                deadline = time.monotonic() + 6
                while not predicate() and time.monotonic() < deadline:
                    root.update()
                    time.sleep(0.005)
                self.assertTrue(predicate(), f"Preview did not complete: {errors}")
            wait_until(lambda: preview.ready and preview._photo is not None)
            self.assertEqual((loaded[0]["width"], loaded[0]["height"]), (64, 36))
            previous_image = preview._image
            preview.seek(0.3)
            wait_until(lambda: preview._image is not previous_image)
            self.assertAlmostEqual(preview.position, 0.3)
            preview.set_segments([SpeedSegment(0, preview.duration, 2, 0)])
            self.assertAlmostEqual(preview._mapping.output_duration, preview.duration / 2)
            preview.seek(0)
            preview.play()
            wait_until(lambda: not preview.playing)
            self.assertAlmostEqual(preview.position, preview.duration)
            self.assertEqual(errors, [])
            decoder = preview._decoder
            preview.close()
            decoder.thread.join(3)
            self.assertFalse(decoder.thread.is_alive())
            source.rename(Path(folder) / "released.mp4")

    @unittest.skipUnless(importlib.util.find_spec("pedalboard"), "optional Rubber Band not installed")
    def test_real_video_audio_reader_drives_preview_and_seek_restarts_sound(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display is unavailable: {exc}")
        root.attributes("-alpha", 0.0)
        root.geometry("640x360")
        self.addCleanup(root.destroy)
        from moviepy import AudioClip, VideoClip
        from apps.video_preview_audio import PreviewAudio

        class ClockPlayer:
            def __init__(self):
                self.started = None
                self.frames = 0
                self.closed = self.stopped = False
                self.energy = 0.0
            @property
            def position_seconds(self):
                if self.started is None:
                    return 0.0
                return min(self.frames / 48000, time.monotonic() - self.started)
            def write(self, audio, rate):
                if self.started is None:
                    self.started = time.monotonic()
                self.frames += len(audio)
                self.energy += float(np.sum(np.asarray(audio) ** 2))
            def finish(self):
                while not self.stopped and self.position_seconds < self.frames / 48000:
                    time.sleep(.005)
            def stop(self):
                self.stopped = True
            def close(self):
                self.closed = True
        sessions = []
        def audio_session(*args):
            session = PreviewAudio(*args, player_factory=ClockPlayer)
            sessions.append(session)
            return session
        preview = VideoPreview(root)
        preview.pack(fill="both", expand=True)
        self.addCleanup(preview.close)
        root.update()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "sound.mp4"
            sound = AudioClip(lambda t: .1 * np.sin(2 * np.pi * 440 * np.asarray(t)),
                              duration=1.2, fps=48000)
            clip = VideoClip(lambda t: np.full((36, 64, 3), (40, 60, 120), dtype=np.uint8),
                             duration=1.2).with_fps(10).with_audio(sound)
            try:
                clip.write_videofile(str(source), codec="libx264", audio_codec="aac",
                                     preset="ultrafast", threads=1, logger=None)
            finally:
                clip.close()
                sound.close()
            errors = []
            def wait_until(predicate):
                deadline = time.monotonic() + 6
                while not predicate() and time.monotonic() < deadline:
                    root.update()
                    time.sleep(.005)
                self.assertTrue(predicate(), f"Audio preview did not complete: {errors}")
            try:
                with patch("apps.video_preview_audio.PreviewAudio", side_effect=audio_session):
                    preview.open(source, on_error=errors.append)
                    wait_until(lambda: preview.ready)
                    self.assertTrue(preview.has_audio)
                    preview.play()
                    wait_until(lambda: sessions and sessions[-1].player.frames > 0)
                    first = sessions[-1]
                    preview.seek(.5)
                    self.assertTrue(first.stopped.is_set())
                    self.assertEqual(len(sessions), 2)
                    self.assertAlmostEqual(sessions[-1].start_output, .5)
                    wait_until(lambda: preview.position > .55)
                    preview.pause()
                    self.assertTrue(sessions[-1].stopped.is_set())
                    frozen = preview.position
                    root.update()
                    self.assertAlmostEqual(preview.position, frozen)
                    preview.set_segments([SpeedSegment(0, preview.duration, 2, 0)])
                    preview.play()
                    wait_until(lambda: not preview.playing)
                    self.assertAlmostEqual(preview.position, preview.duration)
                    self.assertEqual(errors, [])
                    self.assertTrue(any(session.player.energy > .1 for session in sessions))
            finally:
                preview.close()
                for session in sessions:
                    session.stop()
                    session.thread.join(3)
                    self.assertFalse(session.thread.is_alive())
                    self.assertTrue(session.player.closed)
                if preview._decoder:
                    preview._decoder.thread.join(3)
            source.rename(Path(folder) / "released.mp4")


class PreviewDecoderTests(unittest.TestCase):
    def make_decoder(self, opener):
        decoder = _PreviewDecoder(opener)
        self.addCleanup(self.dispose, decoder)
        return decoder

    @staticmethod
    def dispose(decoder):
        decoder.close()
        decoder.thread.join(3)

    def wait_for_event(self, decoder, kind, generation=1, revision=None):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            for event in decoder.drain():
                if event[0] == kind and event[1] == generation:
                    if revision is None or event[2] == revision:
                        return event
            time.sleep(0.005)
        self.fail(f"No {kind} event for generation {generation}")

    def test_open_returns_original_metadata_and_reader_stays_on_worker(self):
        clip = FakeClip()
        decoder = self.make_decoder(lambda path: clip)
        decoder.open(1, Path("source.mp4"))
        event = self.wait_for_event(decoder, "loaded")
        self.assertEqual(event[2], {"duration": 12, "fps": 24, "width": 1920,
                                    "height": 1080, "has_audio": True})
        self.assertEqual(event[3].shape, (54, 96, 3))
        decoder.close()
        decoder.thread.join(3)
        self.assertFalse(decoder.thread.is_alive())
        self.assertTrue(clip.closed)
        self.assertEqual(set(clip.owner_threads), {decoder.thread.ident})

    def test_new_open_discards_and_closes_reader_that_finishes_late(self):
        entered, release = threading.Event(), threading.Event()
        first, second = FakeClip(), FakeClip()
        def opener(path):
            if path.name == "first.mp4":
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("Test did not release opening")
                return first
            return second
        decoder = self.make_decoder(opener)
        self.addCleanup(release.set)
        decoder.open(1, Path("first.mp4"))
        self.assertTrue(entered.wait(3))
        decoder.open(2, Path("second.mp4"))
        release.set()
        self.wait_for_event(decoder, "loaded", generation=2)
        self.assertTrue(first.closed)
        self.assertEqual(first.calls, [])
        self.assertEqual(second.calls, [0])

    def test_rapid_seeks_replace_pending_request_and_keep_latest_revision(self):
        entered, release = threading.Event(), threading.Event()
        clip = FakeClip(block_at=1, entered=entered, release=release)
        decoder = self.make_decoder(lambda path: clip)
        self.addCleanup(release.set)
        decoder.open(1, Path("source.mp4"))
        self.wait_for_event(decoder, "loaded")
        decoder.request(1, 1, 1)
        self.assertTrue(entered.wait(3))
        decoder.request(1, 2, 2)
        decoder.request(1, 3, 3)
        release.set()
        event = self.wait_for_event(decoder, "frame", revision=3)
        self.assertEqual(clip.calls, [0, 1, 3])
        self.assertEqual(event[3], 3)
        np.testing.assert_array_equal(event[4], 3)

    def test_unload_closes_reader_and_next_open_remains_available(self):
        first, second = FakeClip(), FakeClip()
        decoder = self.make_decoder(lambda path: first if path.name == "first.mp4" else second)
        decoder.open(1, Path("first.mp4"))
        self.wait_for_event(decoder, "loaded")
        decoder.open(2, None)
        decoder.open(3, Path("second.mp4"))
        self.wait_for_event(decoder, "loaded", generation=3)
        self.assertTrue(first.closed)
        decoder.request(1, 1, 4)
        self.assertEqual(first.calls, [0])

    def test_close_during_open_is_nonblocking_and_discards_loaded_callback(self):
        entered, release = threading.Event(), threading.Event()
        clip = FakeClip()
        def opener(path):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("Test did not release opening")
            return clip
        decoder = self.make_decoder(opener)
        self.addCleanup(release.set)
        decoder.open(1, Path("source.mp4"))
        self.assertTrue(entered.wait(3))
        decoder.close()
        self.assertTrue(decoder.thread.is_alive())
        release.set()
        decoder.thread.join(3)
        self.assertFalse(decoder.thread.is_alive())
        self.assertTrue(clip.closed)
        self.assertEqual(decoder.drain(), [])

    def test_invalid_clip_is_closed_and_reports_open_error(self):
        clip = FakeClip()
        clip.duration = float("nan")
        decoder = self.make_decoder(lambda path: clip)
        decoder.open(1, Path("source.mp4"))
        event = self.wait_for_event(decoder, "error")
        self.assertIn("thời lượng", event[2])
        self.assertTrue(clip.closed)
        self.assertEqual(clip.calls, [])

    def test_final_seek_reads_inside_last_frame_interval(self):
        clip = FakeClip()
        decoder = self.make_decoder(lambda path: clip)
        decoder.open(1, Path("source.mp4"))
        self.wait_for_event(decoder, "loaded")
        decoder.request(1, 1, clip.duration)
        self.wait_for_event(decoder, "frame", revision=1)
        self.assertLess(clip.calls[-1], clip.duration)
        self.assertGreater(clip.calls[-1], clip.duration - 1 / clip.fps)


class PreviewClockTests(unittest.TestCase):
    def test_marker_mapping_keeps_exact_anchor_at_speed_below_manual_range(self):
        from apps.video_markers import TimeMarker, build_marker_time_map
        mapping = build_marker_time_map(20, [TimeMarker(1, 1), TimeMarker(2, 10000)],
                                        [TimeMarker(1, 1000), TimeMarker(2, 5000)])
        preview = VideoPreview.__new__(VideoPreview)
        preview.ready = True
        preview._closed = preview.playing = False
        preview.duration = 20
        preview.on_position = Mock()
        preview.position = 0
        preview.set_time_map(mapping)
        self.assertIs(preview._mapping, mapping)
        self.assertEqual(preview._mapping.output_time(.001), 1)
        self.assertEqual(preview._mapping.source_time(5), 10)
        self.assertEqual(preview._segments[0].speed, .001)

    def test_mapping_for_another_video_fails_before_replacing_current_mapping(self):
        from apps.video_markers import TimeMarker, build_marker_time_map
        preview = VideoPreview.__new__(VideoPreview)
        preview.ready = True
        preview._closed = False
        preview.duration = 20
        preview._mapping = None
        mapping = build_marker_time_map(21, [TimeMarker(1, 1000)], [TimeMarker(1, 1000)])
        with self.assertRaises(ValueError):
            preview.set_time_map(mapping)
        self.assertIsNone(preview._mapping)

    def test_marker_mapping_restarts_sound_at_the_same_source_position(self):
        from apps.video_markers import TimeMarker, build_marker_time_map
        preview = VideoPreview.__new__(VideoPreview)
        preview.ready = preview.playing = preview.has_audio = preview.audio_enabled = True
        preview._closed = False
        preview.duration = 20
        preview.on_position = None
        preview._path = "video.mp4"
        preview._mapping = build_multi_speed_time_map(20, [])
        previous = SimpleNamespace(output_position=2, stop=Mock())
        preview._audio = previous
        mapping = build_marker_time_map(20, [TimeMarker(1, 1000), TimeMarker(2, 10000)],
                                        [TimeMarker(1, 2000), TimeMarker(2, 8000)])
        new_session = SimpleNamespace(output_position=float(mapping.output_time(2)), stop=Mock())
        with patch("apps.video_preview_audio.PreviewAudio", return_value=new_session) as create:
            preview.set_time_map(mapping)
        self.assertEqual(preview.position, 2)
        previous.stop.assert_called_once()
        create.assert_called_once_with("video.mp4", mapping, float(mapping.output_time(2)))
        self.assertIs(preview._audio, new_session)

    def test_playback_clock_uses_same_multisegment_ramps_as_export(self):
        mapping = build_multi_speed_time_map(30, [SpeedSegment(4, 12, 2, 1),
                                                SpeedSegment(18, 24, .5, .5)])
        # Test the pure clock calculation without constructing a Tk widget.
        preview = VideoPreview.__new__(VideoPreview)
        preview._mapping = mapping
        preview._audio = None
        preview._anchor_time = 100
        preview._anchor_output = float(mapping.output_time(3))
        for elapsed in (0, .5, 2, 5, 10, 20, 40):
            with self.subTest(elapsed=elapsed), patch("apps.video_preview.time.monotonic",
                                                    return_value=100 + elapsed):
                expected = mapping.source_time(min(mapping.output_duration,
                                                   preview._anchor_output + elapsed))
                self.assertAlmostEqual(preview._clock_position(), expected)


if __name__ == "__main__":
    unittest.main()
