"""Speed-ramp timing and MoviePy integration without a TTS model or devices."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from apps.video_editor import (
    SpeedSegment, _audio_windows, _guard_reader_cleanup, _write_ramped_audio,
    build_multi_speed_time_map, build_speed_time_map, probe_video,
    render_speed_segment, render_speed_segments, require_moviepy,
)


class SpeedTimeMapTests(unittest.TestCase):
    def test_multiple_ranges_sort_and_preserve_every_gap_at_one_x(self):
        mapping = build_multi_speed_time_map(
            10, [SpeedSegment(6, 8, 0.5, 0), SpeedSegment(1, 3, 2, 0)],
        )
        self.assertEqual(mapping.segments, (SpeedSegment(1, 3, 2, 0), SpeedSegment(6, 8, 0.5, 0)))
        self.assertEqual(mapping.output_duration, 11)
        source = [0, 1, 2, 3, 4, 6, 7, 8, 10]
        output = [0, 1, 1.5, 2, 3, 5, 7, 9, 11]
        np.testing.assert_array_equal(mapping.output_time(source), output)
        np.testing.assert_array_equal(mapping.source_time(output), source)

    def test_multiple_ramps_compose_their_durations_and_remain_reversible(self):
        segments = (SpeedSegment(2, 5, 4, 0.6), SpeedSegment(7, 9, 0.25, 0.4),
                    SpeedSegment(9, 11, 1.6, 0.2))
        mapping = build_multi_speed_time_map(15, segments)
        expected = 15 + sum(build_speed_time_map(15, segment.start, segment.end,
                                                segment.speed, segment.ramp_seconds).output_duration - 15
                            for segment in segments)
        self.assertAlmostEqual(mapping.output_duration, expected, places=12)
        self.assertTrue(np.all(np.diff(mapping.source_knots) > 0))
        self.assertTrue(np.all(np.diff(mapping.output_knots) > 0))
        times = np.linspace(0, 15, 4001)
        np.testing.assert_allclose(mapping.source_time(mapping.output_time(times)), times, atol=1e-12)
        for begin, end in ((0, 2), (5, 7), (11, 15)):
            self.assertAlmostEqual(mapping.output_time(end) - mapping.output_time(begin), end - begin)
        with self.assertRaises(ValueError):
            mapping.output_knots[0] = 1

    def test_empty_and_one_x_multiple_ranges_leave_timeline_unchanged(self):
        for segments in ([], [SpeedSegment(1, 3, 1, 0.5), SpeedSegment(5, 8, 1, 1)]):
            with self.subTest(segments=segments):
                mapping = build_multi_speed_time_map(10, segments)
                np.testing.assert_array_equal(mapping.output_time(np.arange(11)), np.arange(11))
                self.assertEqual(mapping.output_duration, 10)

    def test_multiple_ranges_reject_overlap_and_invalid_values(self):
        invalid = ([SpeedSegment(1, 4), SpeedSegment(3, 5)],
                   [SpeedSegment(1, 4), SpeedSegment(1, 4)],
                   [SpeedSegment(-1, 2)], [SpeedSegment(1, 11)],
                   [SpeedSegment(2, 2)], [SpeedSegment(math.nan, 4)],
                   [SpeedSegment(1, math.inf)], [SpeedSegment(1, 3, math.nan)],
                   [SpeedSegment(1, 3, 4.1)], [SpeedSegment(1, 3, 1.5, math.inf)],
                   [SpeedSegment(1, 3, 1.5, 1.1)], [(1, 3, 1.5, 0.5)])
        for segments in invalid:
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                build_multi_speed_time_map(10, segments)
        for duration in (0, -1, math.nan, math.inf):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                build_multi_speed_time_map(duration, [])

    def test_multi_audio_windows_cover_adjacent_ranges_and_gaps_without_holes(self):
        mapping = build_multi_speed_time_map(
            300, [SpeedSegment(20, 80, 4, 0.5), SpeedSegment(80, 110, 0.25, 0.3),
                  SpeedSegment(150, 230, 1.5, 0.4)],
        )
        windows = list(_audio_windows(mapping))
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[-1][1], 300)
        self.assertLessEqual(max(end - begin for begin, end in windows), 4)
        for current, following in zip(windows[:-1], windows[1:]):
            self.assertEqual(current[1], following[0])
        self.assertLess(mapping.source_knots.nbytes + mapping.output_knots.nbytes, 100000)

    def test_constant_speed_changes_only_selected_range(self):
        mapping = build_speed_time_map(120, 10, 30, 2, 0)
        self.assertEqual(mapping.output_duration, 110)
        np.testing.assert_allclose(mapping.source_time([0, 9, 10, 15, 20, 25, 110]),
                                   [0, 9, 10, 20, 30, 35, 120])
        slower = build_speed_time_map(120, 10, 30, 0.5, 0)
        self.assertEqual(slower.output_duration, 140)
        self.assertEqual(slower.output_time(50), 70)

    def test_ramp_is_monotone_and_reversible_at_every_source_time(self):
        for speed in (0.25, 0.8, 1.5, 4):
            for ramp in (0, 0.5, 5):
                with self.subTest(speed=speed, ramp=ramp):
                    mapping = build_speed_time_map(30, 5, 15, speed, ramp)
                    self.assertTrue(np.all(np.diff(mapping.source_knots) > 0))
                    self.assertTrue(np.all(np.diff(mapping.output_knots) > 0))
                    source_times = np.linspace(0, 30, 2001)
                    np.testing.assert_allclose(mapping.source_time(mapping.output_time(source_times)),
                                               source_times, atol=1e-12)
                    self.assertEqual(mapping.source_time(mapping.output_duration), 30)

    def test_ramp_speed_joins_one_x_and_plateau_continuously(self):
        mapping = build_speed_time_map(20, 4, 12, 3, 1)
        for source_t, expected in ((4, 1), (5, 3), (8, 3), (11, 3), (12, 1)):
            output_t = mapping.output_time(source_t)
            epsilon = 0.0001
            derivative = (mapping.source_time(output_t + epsilon) -
                          mapping.source_time(output_t - epsilon)) / (2 * epsilon)
            self.assertAlmostEqual(derivative, expected, delta=0.00002)
        output_start = mapping.output_time(4)
        output_end = mapping.output_time(12)
        self.assertEqual(output_start, 4)
        self.assertAlmostEqual(mapping.source_time(output_end + 2), 14)

    def test_one_x_has_exact_identity_and_no_dense_knots(self):
        mapping = build_speed_time_map(3600, 30, 3000, 1, 100)
        values = np.arange(3601)
        np.testing.assert_array_equal(mapping.source_time(values), values)
        self.assertEqual(mapping.output_duration, 3600)
        self.assertEqual(len(mapping.source_knots), 2)

    def test_long_video_map_memory_does_not_depend_on_duration(self):
        mapping = build_speed_time_map(24 * 3600, 60, 20 * 3600, 1.5, 1)
        self.assertLessEqual(len(mapping.source_knots), 2054)
        self.assertLess(mapping.source_knots.nbytes + mapping.output_knots.nbytes, 40000)
        with self.assertRaises(ValueError):
            mapping.source_knots[0] = 10

    def test_outside_selected_range_stays_one_x(self):
        mapping = build_speed_time_map(100, 10, 80, 1.7, 2)
        np.testing.assert_array_equal(mapping.source_time(np.arange(10)), np.arange(10))
        after = mapping.output_time(np.arange(81, 101))
        np.testing.assert_allclose(np.diff(after), 1)

    def test_invalid_range_speed_and_ramp_fail_before_processing(self):
        invalid = ((0, 0, 1, 1.5, 0), (10, -1, 2, 1.5, 0),
                   (10, 2, 2, 1.5, 0), (10, 2, 11, 1.5, 0),
                   (10, 2, 4, 0, 0), (10, 2, 4, 4.1, 0),
                   (10, 2, 4, 1.5, -0.1), (10, 2, 4, 1.5, 1.1),
                   (math.inf, 2, 4, 1.5, 0), (10, 2, 4, math.nan, 0))
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                build_speed_time_map(*arguments)

    def test_audio_windows_cover_source_and_remain_bounded(self):
        for speed in (0.25, 4):
            mapping = build_speed_time_map(300, 100, 200, speed, 0.5)
            windows = list(_audio_windows(mapping))
            self.assertEqual(windows[0][0], 0)
            self.assertEqual(windows[-1][1], 300)
            self.assertLessEqual(max(end - begin for begin, end in windows), 4)
            for current, following in zip(windows[:-1], windows[1:]):
                self.assertEqual(current[1], following[0])

    def test_missing_moviepy_has_actionable_optional_install_message(self):
        with patch.dict("sys.modules", {"moviepy": None}), self.assertRaisesRegex(ValueError, "--extra video"):
            require_moviepy()

    def test_output_cannot_replace_source_or_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.mp4"
            source.write_bytes(b"original")
            destination = Path(folder) / "existing.mp4"
            destination.write_bytes(b"keep")
            for output in (source, destination):
                with self.assertRaises(ValueError):
                    render_speed_segment(source, output, 1, 2)
            self.assertEqual(source.read_bytes(), b"original")
            self.assertEqual(destination.read_bytes(), b"keep")

    def test_invalid_encoder_settings_fail_without_loading_moviepy(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.mp4"
            source.write_bytes(b"original")
            with patch("apps.video_editor.require_moviepy") as loader:
                for options in ({"quality": 52}, {"preset": "invalid"}):
                    with self.assertRaises(ValueError):
                        render_speed_segment(source, Path(folder) / "out.mp4", 1, 2, **options)
                loader.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("pedalboard"), "optional pedalboard not installed")
class VideoAudioTests(unittest.TestCase):
    def test_multi_rubberband_audio_preserves_pitch_and_untouched_middle_gap(self):
        class SineAudio:
            nchannels = 2
            duration = 6

            def get_frame(self, times):
                return np.column_stack((0.2 * np.sin(2 * np.pi * 220 * times),
                                        0.1 * np.sin(2 * np.pi * 440 * times)))

        mapping = build_multi_speed_time_map(
            6, [SpeedSegment(1, 2.5, 2, 0.25), SpeedSegment(4, 5.5, 0.5, 0.25)],
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "multi.wav"
            _write_ramped_audio(SineAudio(), path, mapping)
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
            self.assertEqual(audio.shape, (round(mapping.output_duration * rate), 2))
            # The middle gap is copied at 1x using the new absolute offset.
            begin = round(float(mapping.output_time(3)) * rate)
            count = rate // 2
            expected = SineAudio().get_frame(np.arange(3 * rate, 3 * rate + count) / rate).astype(np.float32)
            np.testing.assert_allclose(audio[begin:begin + count], expected, atol=1e-7)
            for source_begin, source_end in ((1.5, 2), (4.5, 5)):
                first = round(float(mapping.output_time(source_begin)) * rate)
                last = round(float(mapping.output_time(source_end)) * rate)
                for channel, expected_pitch in ((0, 220), (1, 440)):
                    section = audio[first:last, channel]
                    spectrum = np.abs(np.fft.rfft(section * np.hanning(len(section))))
                    pitch = np.fft.rfftfreq(len(section), 1 / rate)[spectrum.argmax()]
                    self.assertAlmostEqual(pitch, expected_pitch, delta=3)

    def test_transients_in_multiple_ramps_and_gaps_follow_video_time(self):
        class PulseAudio:
            nchannels = 1
            duration = 8

            def __init__(self, center):
                self.center = center

            def get_frame(self, times):
                envelope = np.exp(-((times - self.center) / 0.014) ** 2)
                return (0.3 * envelope * np.sin(2 * np.pi * 400 * times))[:, None]

        mapping = build_multi_speed_time_map(
            8, [SpeedSegment(1, 3, 0.5, 0.4), SpeedSegment(5, 7, 2, 0.4)],
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pulses.wav"
            for center in (0.8, 1.15, 2.2, 3.5, 4.4, 5.15, 6.1, 7.5):
                with self.subTest(center=center):
                    _write_ramped_audio(PulseAudio(center), path, mapping)
                    audio, rate = sf.read(path, dtype="float32")
                    self.assertEqual(len(audio), round(mapping.output_duration * rate))
                    actual = np.argmax(np.abs(audio)) / rate
                    self.assertAlmostEqual(actual, mapping.output_time(center), delta=1 / 30)

    def test_rubberband_audio_is_stereo_pitch_preserving_and_exact_length(self):
        class SineAudio:
            nchannels = 2
            duration = 6

            def get_frame(self, times):
                return np.column_stack((0.2 * np.sin(2 * np.pi * 220 * times),
                                        0.1 * np.sin(2 * np.pi * 440 * times)))

        rate = 48000
        mapping = build_speed_time_map(6, 2, 4, 1.5, 0.4)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audio.wav"
            _write_ramped_audio(SineAudio(), path, mapping)
            audio, actual_rate = sf.read(path, dtype="float32", always_2d=True)
            self.assertEqual(actual_rate, rate)
            self.assertEqual(audio.shape, (round(mapping.output_duration * rate), 2))
            expected = SineAudio().get_frame(np.arange(rate) / rate).astype(np.float32)
            np.testing.assert_array_equal(audio[:rate], expected)
            begin = round(mapping.output_time(2.6) * rate)
            end = round(mapping.output_time(3.4) * rate)
            for channel, expected_pitch in ((0, 220), (1, 440)):
                section = audio[begin:end, channel]
                spectrum = np.abs(np.fft.rfft(section * np.hanning(len(section))))
                pitch = np.fft.rfftfreq(len(section), 1 / rate)[spectrum.argmax()]
                self.assertAlmostEqual(pitch, expected_pitch, delta=3)

    def test_transient_audio_stays_on_shared_video_timeline(self):
        class PulseAudio:
            nchannels = 1
            duration = 7

            def __init__(self, center):
                self.center = center

            def get_frame(self, times):
                envelope = np.exp(-((times - self.center) / 0.014) ** 2)
                return (0.3 * envelope * np.sin(2 * np.pi * 400 * times))[:, None]

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audio.wav"
            for speed in (0.25, 1.5, 4):
                mapping = build_speed_time_map(7, 2, 5, speed, 0.5)
                for center in (2.15, 3.4, 4.9):
                    with self.subTest(speed=speed, center=center):
                        _write_ramped_audio(PulseAudio(center), path, mapping)
                        audio, rate = sf.read(path, dtype="float32")
                        actual = np.argmax(np.abs(audio)) / rate
                        self.assertAlmostEqual(actual, mapping.output_time(center), delta=1 / 30)

    def test_short_source_audio_is_padded_to_video_duration(self):
        class ShortAudio:
            nchannels = 1
            duration = 0.5

            def get_frame(self, times):
                return np.full((len(times), 1), 0.1)

        mapping = build_speed_time_map(3, 1, 2, 1.5, 0.25)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audio.wav"
            _write_ramped_audio(ShortAudio(), path, mapping)
            audio, rate = sf.read(path, dtype="float32")
            self.assertEqual(len(audio), round(mapping.output_duration * rate))
            self.assertFalse(np.any(audio[rate:]))


@unittest.skipUnless(importlib.util.find_spec("moviepy"), "optional MoviePy not installed")
class MoviePyRenderTests(unittest.TestCase):
    def setUp(self):
        from moviepy import AudioClip, VideoClip
        self.folder = tempfile.TemporaryDirectory()
        self.directory = Path(self.folder.name)
        self.source = self.directory / "source.mp4"
        self.source_video = VideoClip(
            lambda t: np.full((48, 64, 3), (round(t * 80), 30, 150), dtype=np.uint8),
            duration=2,
        ).with_fps(12)
        audio = AudioClip(lambda t: np.column_stack((
            0.2 * np.sin(2 * np.pi * 220 * np.asarray(t)),
            0.1 * np.sin(2 * np.pi * 440 * np.asarray(t)),
        )).squeeze(), duration=2, fps=48000)
        self.source_video = self.source_video.with_audio(audio)
        self.source_video.write_videofile(str(self.source), codec="libx264", audio_codec="aac",
                                          preset="ultrafast", threads=1, logger=None,
                                          temp_audiofile=str(self.directory / "source.m4a"))
        self.source_video.close()

    def tearDown(self):
        self.folder.cleanup()

    def test_real_multi_render_keeps_audio_fps_and_frames_in_middle_gap(self):
        from moviepy import VideoFileClip
        output = self.directory / "multiple.mp4"
        segments = [SpeedSegment(0.2, 0.7, 2, 0.1), SpeedSegment(1.2, 1.7, 0.5, 0.1)]
        mapping = render_speed_segments(self.source, output, segments, preset="ultrafast")
        information = probe_video(output)
        self.assertEqual(information["fps"], 12)
        self.assertTrue(information["has_audio"])
        self.assertAlmostEqual(information["duration"], mapping.output_duration, delta=1 / 12 + 0.03)
        with VideoFileClip(str(self.source), audio=False) as source, VideoFileClip(str(output)) as result:
            _guard_reader_cleanup(source)
            _guard_reader_cleanup(result)
            for source_t in (0.1, 0.9, 1.1, 1.8):
                expected = source.get_frame(source_t)
                actual = result.get_frame(float(mapping.output_time(source_t)))
                self.assertLess(np.abs(expected.astype(float) - actual.astype(float)).mean(), 9)
            self.assertAlmostEqual(result.audio.duration, mapping.output_duration, delta=1 / 12 + 0.05)
        self.assertFalse(list(self.directory.glob(".video_*")))

    def test_multi_cancel_removes_temporary_files_and_releases_source(self):
        class Cancelled(RuntimeError):
            pass

        stopped = False

        def progress(fraction):
            nonlocal stopped
            stopped = fraction > 0.3

        def check_stop():
            if stopped:
                raise Cancelled("multi render cancelled")

        output = self.directory / "multi_cancelled.mp4"
        with self.assertRaises(Cancelled):
            render_speed_segments(
                self.source, output,
                [SpeedSegment(0.2, 0.7, 2, 0.1), SpeedSegment(1.2, 1.7, 0.5, 0.1)],
                preset="ultrafast", check_stop=check_stop, progress=progress,
            )
        self.assertFalse(output.exists())
        self.assertFalse(list(self.directory.glob(".video_*")))
        self.source.rename(self.directory / "released.mp4")

    def test_multi_one_x_requires_no_rubberband(self):
        output = self.directory / "multiple_identity.mp4"
        with patch("apps.speech_speed.require_rubberband", side_effect=AssertionError("unused")):
            mapping = render_speed_segments(
                self.source, output, [SpeedSegment(0.2, 0.7, 1, 0.1), SpeedSegment(1.2, 1.7, 1, 0.1)],
                preset="ultrafast",
            )
        self.assertEqual(mapping.output_duration, 2)
        self.assertTrue(probe_video(output)["has_audio"])

    def test_real_render_keeps_source_fps_audio_and_unchanged_frames(self):
        from moviepy import VideoFileClip
        output = self.directory / "ramped.mp4"
        progress = []
        mapping = render_speed_segment(self.source, output, 0.5, 1.5, 1.5, 0.25,
                                       preset="ultrafast", progress=progress.append)
        information = probe_video(output)
        self.assertEqual(information["fps"], 12)
        self.assertEqual((information["width"], information["height"]), (64, 48))
        self.assertTrue(information["has_audio"])
        self.assertAlmostEqual(information["duration"], mapping.output_duration, delta=1 / 12 + 0.03)
        self.assertEqual(progress[0], 0)
        self.assertEqual(progress[-1], 1)
        with VideoFileClip(str(self.source), audio=False) as source, VideoFileClip(str(output), audio=False) as result:
            _guard_reader_cleanup(source)
            _guard_reader_cleanup(result)
            for output_t in (0.25, mapping.output_time(1.75)):
                expected = source.get_frame(mapping.source_time(output_t))
                actual = result.get_frame(output_t)
                self.assertLess(np.abs(expected.astype(float) - actual.astype(float)).mean(), 9)
        self.assertFalse(list(self.directory.glob(".video_*")))

    def test_original_speed_requires_no_rubberband(self):
        output = self.directory / "same_speed.mp4"
        with patch("apps.speech_speed.require_rubberband", side_effect=AssertionError("unused")):
            mapping = render_speed_segment(self.source, output, 0.5, 1.5, 1, 0.25, preset="ultrafast")
        self.assertEqual(mapping.output_duration, 2)
        self.assertTrue(probe_video(output)["has_audio"])

    def test_cancel_during_writer_removes_video_audio_and_releases_source(self):
        class Cancelled(RuntimeError):
            pass

        stopped = False

        def progress(fraction):
            nonlocal stopped
            if fraction > 0.3:
                stopped = True

        def check_stop():
            if stopped:
                raise Cancelled("cancel test")

        output = self.directory / "cancelled.mp4"
        with self.assertRaises(Cancelled):
            render_speed_segment(self.source, output, 0.5, 1.5, 1.5, 0.25,
                                 preset="ultrafast", check_stop=check_stop, progress=progress)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.directory.glob(".video_*")))
        # The FFmpeg decoder is closed: Windows permits moving the source.
        moved = self.directory / "closed.mp4"
        self.source.rename(moved)
        self.assertTrue(moved.exists())

    def test_validation_after_probe_closes_source_and_leaves_no_partial_file(self):
        output = self.directory / "invalid.mp4"
        with self.assertRaises(ValueError):
            render_speed_segment(self.source, output, 0.5, 3, 1.5, 0.25)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.directory.glob(".video_*")))
        self.source.rename(self.directory / "closed.mp4")

    def test_cancel_during_audio_encoder_closes_writer_before_cleanup(self):
        class Cancelled(RuntimeError):
            pass

        stopped = False

        def progress(fraction):
            nonlocal stopped
            if 0.18 < fraction < 0.25:
                stopped = True

        def check_stop():
            if stopped:
                raise Cancelled("cancel audio encoder")

        output = self.directory / "cancelled_audio.mp4"
        with self.assertRaises(Cancelled):
            render_speed_segment(self.source, output, 0.5, 1.5, 1.5, 0.25,
                                 preset="ultrafast", check_stop=check_stop, progress=progress)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.directory.glob(".video_*")))

    def test_odd_source_dimensions_are_padded_without_cropping_content(self):
        from moviepy import ColorClip, VideoFileClip
        odd = self.directory / "odd.mp4"
        clip = ColorClip((65, 49), color=(20, 140, 40), duration=1).with_fps(12)
        clip.write_videofile(str(odd), codec="libx264", preset="ultrafast", logger=None,
                             ffmpeg_params=["-pix_fmt", "yuv444p"], threads=1)
        clip.close()
        output = self.directory / "padded.mp4"
        render_speed_segment(odd, output, 0.2, 0.8, 1.5, 0.2, preset="ultrafast")
        information = probe_video(output)
        self.assertEqual((information["width"], information["height"]), (66, 50))
        self.assertFalse(information["has_audio"])
        with VideoFileClip(str(output), audio=False) as result:
            _guard_reader_cleanup(result)
            frame = result.get_frame(0.1)
            np.testing.assert_allclose(frame[20, 30], [20, 140, 40], atol=4)

    def test_fractional_fps_restores_source_clock_instead_of_rounding_input_clock(self):
        from moviepy import ColorClip, VideoClip
        fractional = self.directory / "fractional.mp4"
        clip = ColorClip((64, 48), color=(20, 140, 40), duration=1).with_fps(24000 / 1001)
        clip.write_videofile(str(fractional), codec="libx264", preset="ultrafast", logger=None,
                             ffmpeg_params=["-r", "24000/1001"], threads=1)
        clip.close()
        original_fps = probe_video(fractional)["fps"]
        output = self.directory / "fractional_output.mp4"
        original_write = VideoClip.write_videofile
        seen = {}

        def write(instance, *args, **kwargs):
            seen.update(kwargs)
            return original_write(instance, *args, **kwargs)

        with patch.object(VideoClip, "write_videofile", write):
            render_speed_segment(fractional, output, 0.2, 0.8, 1.5, 0.2, preset="ultrafast")
        parameters = seen["ffmpeg_params"]
        actual_clock = parameters[parameters.index("-r") + 1]
        self.assertAlmostEqual(float(actual_clock), original_fps, places=8)
        filters = parameters[parameters.index("-vf") + 1]
        self.assertIn(f"setpts=({float(f'{original_fps:.2f}'):.12g}/{original_fps:.12g})*PTS", filters)
        self.assertAlmostEqual(probe_video(output)["fps"], original_fps, places=3)


if __name__ == "__main__":
    unittest.main()
