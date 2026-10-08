"""Numbered anchors, exact alignment, and the shared video/audio renderer."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from apps.video_editor import (
    _audio_windows, _guard_reader_cleanup, _ramped_audio_chunks, _write_ramped_audio,
    probe_video, render_marker_alignment,
)
from apps.video_markers import (
    TimeMarker, build_marker_time_map, parse_marker_text, read_marker_file,
    validate_marker_pairs,
)


OLD_TEXT = "1\n00:00:00,060\n\n2\n00:00:03,130\n\n3\n00:00:06,290\n\n4\n00:00:08,950\n"
NEW_TEXT = "1\n00:00:00,133\n\n2\n00:00:02,995\n\n3\n00:00:05,666\n\n4\n00:00:07,666\n"


class MarkerFileTests(unittest.TestCase):
    def test_example_preserves_every_millisecond_and_id(self):
        self.assertEqual(parse_marker_text(OLD_TEXT), (
            TimeMarker(1, 60), TimeMarker(2, 3130), TimeMarker(3, 6290), TimeMarker(4, 8950),
        ))

    def test_crlf_bom_whitespace_and_nonconsecutive_ids_are_valid(self):
        text = "\ufeff \r\n 3\r\n 00:00:00,001 \r\n\r\n 8 \r\n100:59:59,999\r\n"
        self.assertEqual(parse_marker_text(text), (TimeMarker(3, 1), TimeMarker(8, 363599999)))

    def test_read_utf8_bom_file_and_reject_missing_or_invalid_encoding(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mốc cũ.txt"
            path.write_text(OLD_TEXT, encoding="utf-8-sig")
            self.assertEqual(read_marker_file(path), parse_marker_text(OLD_TEXT))
            path.write_bytes(b"\xff\xfeinvalid")
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                read_marker_file(path)
            with self.assertRaisesRegex(ValueError, "Không tìm thấy"):
                read_marker_file(Path(folder) / "missing.txt")

    def test_invalid_structure_timestamps_ids_and_order_are_rejected(self):
        invalid = (
            "", "\n\n", "1", "0\n00:00:00,001", "-1\n00:00:00,001",
            "foo\n00:00:00,001", "1\n00:00:60,000", "1\n00:60:00,000",
            "1\n00:00:00.001", "1\n0:00:00,001", "1\n00:00:00,01",
            "1\n00:00:00,001 --> 00:00:02,000", "1\n-00:00:00,001",
            "1\n00:00:00,001\n1\n00:00:00,002",
            "1\n00:00:00,002\n2\n00:00:00,001",
            "1\n00:00:00,001\n2\n00:00:00,001",
        )
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_marker_text(text)

    def test_format_error_reports_original_line_number(self):
        with self.assertRaisesRegex(ValueError, "Dòng 4"):
            parse_marker_text("\n1\n\n00:99:00,000")


class MarkerTimeMapTests(unittest.TestCase):
    def test_user_example_lands_exactly_on_all_anchors_and_keeps_tail(self):
        old, new = parse_marker_text(OLD_TEXT), parse_marker_text(NEW_TEXT)
        mapping = build_marker_time_map(12, old, new)
        old_seconds = [marker.time_ms / 1000 for marker in old]
        new_seconds = [marker.time_ms / 1000 for marker in new]
        np.testing.assert_array_equal(mapping.output_time(old_seconds), new_seconds)
        np.testing.assert_array_equal(mapping.source_time(new_seconds), old_seconds)
        self.assertEqual(round(mapping.output_duration * 1000), 10716)
        self.assertEqual(mapping.source_duration, 12)
        self.assertEqual(mapping.segments[-1].speed, 1)
        self.assertEqual(mapping.tail_mode, "keep")
        self.assertEqual(mapping.old_markers, old)
        self.assertEqual(mapping.new_markers, new)
        self.assertAlmostEqual(mapping.output_time(10.95), 9.666, places=12)
        self.assertTrue(all(segment.ramp_seconds == 0 for segment in mapping.segments))
        self.assertFalse(mapping.source_knots.flags.writeable)
        self.assertFalse(mapping.output_knots.flags.writeable)

    def test_first_interval_speed_is_derived_from_shared_zero_origin(self):
        mapping = build_marker_time_map(2, [TimeMarker(10, 200)], [TimeMarker(10, 1000)])
        np.testing.assert_array_equal(mapping.source_knots, [0, 0.2, 2])
        np.testing.assert_array_equal(mapping.output_knots, [0, 1, 2.8])
        self.assertEqual(mapping.segments[0].speed, 0.2)

    def test_zero_origin_and_last_anchor_at_video_end_need_no_duplicate_knots(self):
        mapping = build_marker_time_map(
            2, [TimeMarker(1, 0), TimeMarker(2, 2000)],
            [TimeMarker(1, 0), TimeMarker(2, 1000)],
        )
        np.testing.assert_array_equal(mapping.source_knots, [0, 2])
        np.testing.assert_array_equal(mapping.output_knots, [0, 1])
        self.assertEqual(len(mapping.segments), 1)
        self.assertEqual(mapping.segments[0].speed, 2)

    def test_single_zero_anchor_keeps_whole_video_at_one_x(self):
        markers = [TimeMarker(7, 0)]
        mapping = build_marker_time_map(3, markers, markers)
        self.assertEqual(mapping.output_duration, 3)
        self.assertEqual(mapping.segments[0].speed, 1)

    def test_extreme_required_speeds_are_not_clamped(self):
        mapping = build_marker_time_map(
            4, [TimeMarker(1, 1000), TimeMarker(2, 2000)],
            [TimeMarker(1, 10000), TimeMarker(2, 10001)],
        )
        self.assertEqual(mapping.segments[0].speed, 0.1)
        self.assertEqual(mapping.segments[1].speed, 1000)
        self.assertEqual(mapping.output_time(2), 10.001)

    def test_equal_millisecond_intervals_have_exact_one_x_after_shift(self):
        mapping = build_marker_time_map(
            2, [TimeMarker(1, 3), TimeMarker(2, 53)],
            [TimeMarker(1, 1003), TimeMarker(2, 1053)],
        )
        self.assertEqual(mapping.segments[1].speed, 1)
        self.assertEqual(mapping.segments[-1].speed, 1)

    def test_large_mapping_is_compact_and_reversible(self):
        mapping = build_marker_time_map(
            7200, [TimeMarker(1, 10000), TimeMarker(2, 3600000)],
            [TimeMarker(1, 8000), TimeMarker(2, 4000000)],
        )
        self.assertEqual(len(mapping.source_knots), 4)
        positions = np.linspace(0, 7200, 3001)
        np.testing.assert_allclose(mapping.source_time(mapping.output_time(positions)),
                                   positions, atol=1e-12)

    def test_matching_validator_needs_no_source_duration(self):
        old, new = parse_marker_text(OLD_TEXT), parse_marker_text(NEW_TEXT)
        self.assertEqual(validate_marker_pairs(iter(old), iter(new)), (old, new))

    def test_pair_validation_rejects_missing_duplicate_reordered_and_zero_mismatch(self):
        good = [TimeMarker(1, 100), TimeMarker(2, 200)]
        pairs = (
            ([], good), (good, []), (good, [TimeMarker(1, 100)]),
            (good, [TimeMarker(1, 100), TimeMarker(3, 200)]),
            (good, [TimeMarker(1, 100), TimeMarker(1, 200)]),
            (good, [TimeMarker(2, 100), TimeMarker(1, 200)]),
            (good, [TimeMarker(1, 200), TimeMarker(2, 100)]),
            ([TimeMarker(1, 0)], [TimeMarker(1, 1)]),
            ([TimeMarker(1, 1)], [TimeMarker(1, 0)]),
            ([TimeMarker(0, 0)], [TimeMarker(0, 0)]),
            ([TimeMarker(True, 0)], [TimeMarker(True, 0)]),
            ([TimeMarker(1, -1)], [TimeMarker(1, -1)]),
            ([TimeMarker(1, 1.5)], [TimeMarker(1, 1.5)]),
            ([TimeMarker(1, True)], [TimeMarker(1, True)]),
            ([(1, 100)], [(1, 100)]),
        )
        for old, new in pairs:
            with self.subTest(old=old, new=new), self.assertRaises(ValueError):
                validate_marker_pairs(old, new)

    def test_invalid_duration_out_of_bounds_and_tail_mode_are_rejected(self):
        old, new = [TimeMarker(1, 1000)], [TimeMarker(1, 2000)]
        for duration in (0, -1, math.nan, math.inf, "invalid", None, 0.999):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                build_marker_time_map(duration, old, new)
        with self.assertRaises(ValueError):
            build_marker_time_map(2, old, new, tail_mode="cut")

    def test_infinite_derived_speed_or_output_duration_is_rejected(self):
        huge_time = int(1e308) * 1000
        for old, new in (([TimeMarker(1, huge_time)], [TimeMarker(1, 1)]),
                         ([TimeMarker(1, 1000)], [TimeMarker(1, huge_time)])):
            with self.subTest(old=old, new=new), self.assertRaises(ValueError):
                build_marker_time_map(1e308, old, new)

    def test_slow_audio_windows_bound_output_memory(self):
        mapping = build_marker_time_map(3, [TimeMarker(1, 2000)], [TimeMarker(1, 200000)])
        windows = list(_audio_windows(mapping))
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[-1][1], 3)
        for begin, end in windows:
            self.assertLessEqual(mapping.output_time(end) - mapping.output_time(begin), 4 + 1e-10)

    def test_identity_audio_needs_no_stretch_engine_and_exact_sample_count(self):
        class ConstantAudio:
            nchannels = 1
            duration = 2

            def get_frame(self, times):
                return np.full((len(times), 1), 0.1, dtype=np.float32)

        markers = [TimeMarker(1, 60), TimeMarker(2, 1200)]
        mapping = build_marker_time_map(2, markers, markers)
        with patch("apps.speech_speed.require_rubberband", side_effect=AssertionError("unused")):
            chunks = list(_ramped_audio_chunks(ConstantAudio(), mapping))
        self.assertEqual(sum(len(chunk) for chunk in chunks), 96000)
        np.testing.assert_allclose(np.concatenate(chunks), 0.1, atol=1e-7)


@unittest.skipUnless(importlib.util.find_spec("pedalboard"), "optional pedalboard not installed")
class MarkerAudioTests(unittest.TestCase):
    def test_changed_intervals_keep_pitch_and_unchanged_tail_is_copied(self):
        class SineAudio:
            nchannels = 2
            duration = 6

            def get_frame(self, times):
                return np.column_stack((0.2 * np.sin(2 * np.pi * 220 * times),
                                        0.1 * np.sin(2 * np.pi * 440 * times)))

        mapping = build_marker_time_map(
            6, [TimeMarker(1, 2000), TimeMarker(2, 4000)],
            [TimeMarker(1, 4000), TimeMarker(2, 5000)],
        )
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "audio.wav"
            _write_ramped_audio(SineAudio(), output, mapping)
            audio, rate = sf.read(output, always_2d=True, dtype="float32")
        self.assertEqual(audio.shape, (round(mapping.output_duration * rate), 2))
        for source_start, source_end in ((0.6, 1.4), (2.6, 3.4)):
            first = round(float(mapping.output_time(source_start)) * rate)
            last = round(float(mapping.output_time(source_end)) * rate)
            for channel, expected_pitch in ((0, 220), (1, 440)):
                section = audio[first:last, channel]
                spectrum = np.abs(np.fft.rfft(section * np.hanning(len(section))))
                frequency = np.fft.rfftfreq(len(section), 1 / rate)[spectrum.argmax()]
                self.assertAlmostEqual(frequency, expected_pitch, delta=3)
        source_first = round(4.6 * rate)
        output_first = round(float(mapping.output_time(4.6)) * rate)
        expected = SineAudio().get_frame(np.arange(source_first, source_first + rate // 4) / rate)
        np.testing.assert_allclose(audio[output_first:output_first + rate // 4], expected, atol=1e-7)

    def test_audio_transients_follow_piecewise_marker_timeline(self):
        class PulseAudio:
            nchannels = 1
            duration = 3

            def __init__(self, center):
                self.center = center

            def get_frame(self, times):
                envelope = np.exp(-((times - self.center) / 0.012) ** 2)
                return (0.2 * envelope * np.sin(2 * np.pi * 440 * times))[:, None]

        mapping = build_marker_time_map(
            3, [TimeMarker(1, 300), TimeMarker(2, 1200), TimeMarker(3, 2200)],
            [TimeMarker(1, 150), TimeMarker(2, 1600), TimeMarker(3, 2400)],
        )
        for center in (0.15, 0.3, 0.8, 1.2, 1.7, 2.2, 2.7):
            with self.subTest(center=center):
                chunks = list(_ramped_audio_chunks(PulseAudio(center), mapping))
                audio = np.concatenate(chunks)
                self.assertEqual(len(audio), round(mapping.output_duration * 48000))
                actual = np.argmax(np.abs(audio[:, 0])) / 48000
                self.assertAlmostEqual(actual, mapping.output_time(center), delta=1 / 30)

    def test_extremely_slow_silent_audio_keeps_each_chunk_small(self):
        class SilentAudio:
            nchannels = 1
            duration = 1

            def get_frame(self, times):
                return np.zeros((len(times), 1), dtype=np.float32)

        # The second case has less than one source sample per output window.
        for duration, source_ms in ((1, 1000), (0.001, 1)):
            with self.subTest(duration=duration):
                mapping = build_marker_time_map(duration, [TimeMarker(1, source_ms)],
                                                [TimeMarker(1, 200000)])
                total = 0
                for chunk in _ramped_audio_chunks(SilentAudio(), mapping):
                    total += len(chunk)
                    self.assertLessEqual(len(chunk), math.ceil(4.4 * 48000))
                self.assertEqual(total, round(mapping.output_duration * 48000))


@unittest.skipUnless(importlib.util.find_spec("moviepy") and importlib.util.find_spec("pedalboard"),
                     "optional MoviePy or pedalboard not installed")
class MarkerRenderTests(unittest.TestCase):
    def setUp(self):
        from moviepy import AudioClip, VideoClip

        self.folder = tempfile.TemporaryDirectory()
        self.directory = Path(self.folder.name)
        self.source = self.directory / "source.mp4"
        clip = VideoClip(lambda t: np.full((48, 64, 3),
                                          (round(t * 80), 30, 150), dtype=np.uint8),
                         duration=2).with_fps(24)
        audio = AudioClip(lambda t: np.column_stack((
            0.2 * np.sin(2 * np.pi * 220 * np.asarray(t)),
            0.1 * np.sin(2 * np.pi * 440 * np.asarray(t)),
        )).squeeze(), duration=2, fps=48000)
        clip = clip.with_audio(audio)
        clip.write_videofile(str(self.source), codec="libx264", audio_codec="aac",
                             preset="ultrafast", threads=1, logger=None,
                             temp_audiofile=str(self.directory / "source.m4a"))
        clip.close()
        self.old = [TimeMarker(1, 200), TimeMarker(2, 800), TimeMarker(3, 1400)]
        self.new = [TimeMarker(1, 133), TimeMarker(2, 666), TimeMarker(3, 1666)]

    def tearDown(self):
        self.folder.cleanup()

    def test_real_marker_render_aligns_frames_and_keeps_audio_fps_dimensions(self):
        from moviepy import VideoFileClip

        output = self.directory / "aligned.mp4"
        progress = []
        mapping = render_marker_alignment(self.source, output, self.old, self.new,
                                          preset="ultrafast", progress=progress.append)
        information = probe_video(output)
        self.assertEqual(information["fps"], 24)
        self.assertTrue(information["has_audio"])
        self.assertEqual((information["width"], information["height"]), (64, 48))
        self.assertAlmostEqual(information["duration"], mapping.output_duration, delta=1 / 24 + 0.03)
        self.assertEqual(progress[0], 0)
        self.assertEqual(progress[-1], 1)
        with VideoFileClip(str(self.source), audio=False) as source, VideoFileClip(str(output)) as result:
            _guard_reader_cleanup(source)
            _guard_reader_cleanup(result)
            for old, new in zip(self.old, self.new):
                expected = source.get_frame(old.time_ms / 1000)
                actual = result.get_frame(new.time_ms / 1000)
                self.assertLess(np.abs(expected.astype(float) - actual.astype(float)).mean(), 9)
            self.assertAlmostEqual(result.audio.duration, mapping.output_duration, delta=1 / 24 + 0.05)
        self.assertFalse(list(self.directory.glob(".video_*")))

    def test_identity_render_needs_no_stretch_engine(self):
        output = self.directory / "identity.mp4"
        with patch("apps.speech_speed.require_rubberband", side_effect=AssertionError("unused")):
            mapping = render_marker_alignment(self.source, output, self.old, self.old,
                                              preset="ultrafast")
        self.assertEqual(mapping.output_duration, 2)
        self.assertTrue(probe_video(output)["has_audio"])

    def test_cancellation_cleans_outputs_and_releases_video(self):
        class Cancelled(RuntimeError):
            pass

        stopped = False

        def progress(fraction):
            nonlocal stopped
            stopped = fraction > 0.3

        def check_stop():
            if stopped:
                raise Cancelled("marker render cancelled")

        output = self.directory / "cancelled.mp4"
        with self.assertRaises(Cancelled):
            render_marker_alignment(self.source, output, self.old, self.new,
                                    preset="ultrafast", check_stop=check_stop, progress=progress)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.directory.glob(".video_*")))
        self.source.rename(self.directory / "released.mp4")

    def test_invalid_source_anchor_closes_source_without_partial_output(self):
        output = self.directory / "invalid.mp4"
        with self.assertRaisesRegex(ValueError, "vượt thời lượng"):
            render_marker_alignment(self.source, output, [TimeMarker(1, 2500)],
                                    [TimeMarker(1, 2000)], preset="ultrafast")
        self.assertFalse(output.exists())
        self.assertFalse(list(self.directory.glob(".video_*")))
        self.source.rename(self.directory / "released.mp4")


if __name__ == "__main__":
    unittest.main()
