"""Native retiming, GPU selection and cancellable FFmpeg process integration."""
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from apps.video_editor import SpeedSegment, _guard_reader_cleanup, build_multi_speed_time_map
from apps.video_fast_export import (
    _ENCODER_CACHE, _FFmpegFailure, _ffmpeg_binary, _simplify_knots, _time_expression, _video_filter,
    can_render_ffmpeg, get_video_encoding_options, render_video_ffmpeg,
    select_video_encoder,
)
from apps.video_markers import TimeMarker, build_marker_time_map


class NativeEncoderTests(unittest.TestCase):
    def setUp(self):
        _ENCODER_CACHE.clear()

    def tearDown(self):
        _ENCODER_CACHE.clear()

    def test_native_path_accepts_constant_ranges_and_smooth_ramps(self):
        self.assertTrue(can_render_ffmpeg(build_multi_speed_time_map(10, [])))
        self.assertTrue(can_render_ffmpeg(build_multi_speed_time_map(
            10, [SpeedSegment(2, 8, 1, 1)],
        )))
        self.assertTrue(can_render_ffmpeg(build_multi_speed_time_map(
            10, [SpeedSegment(2, 8, 2, 0)],
        )))
        self.assertTrue(can_render_ffmpeg(build_multi_speed_time_map(
            10, [SpeedSegment(2, 8, 2, 1)],
        )))

    def test_ramp_simplification_bounds_error_across_long_source_timeline(self):
        mapping = build_multi_speed_time_map(7200, [
            SpeedSegment(100, 130, 4, 5), SpeedSegment(3600, 3650, 0.25, 10),
        ])
        for fps in (12, 30, 60):
            source, output = _simplify_knots(mapping.source_knots, mapping.output_knots, 0.01 / fps)
            errors = np.abs(np.interp(mapping.source_knots, source, output) - mapping.output_knots)
            self.assertLessEqual(errors.max(), 0.01 / fps + 1e-10)
            self.assertLess(len(source), len(mapping.source_knots) / 4)
            self.assertEqual(source[0], 0)
            self.assertEqual(output[-1], mapping.output_duration)

    def test_auto_uses_actual_encoder_probe_and_caches_failure(self):
        with patch("apps.video_fast_export._run_ffmpeg", side_effect=_FFmpegFailure("driver missing")) as run:
            self.assertEqual(select_video_encoder(), "libx264")
            self.assertEqual(select_video_encoder(), "libx264")
            self.assertEqual(run.call_count, 1)
            command = run.call_args.args[0]
            self.assertIn("h264_nvenc", command)
            self.assertIn("-frames:v", command)
            with self.assertRaisesRegex(ValueError, "driver missing"):
                select_video_encoder("h264_nvenc")

    def test_gpu_preset_quality_and_cpu_options_are_explicit(self):
        with patch("apps.video_fast_export._run_ffmpeg", return_value=""):
            self.assertEqual(get_video_encoding_options(20, "veryfast"),
                             ("h264_nvenc", "p2", ["-rc", "vbr", "-cq", "20", "-b:v", "0"]))
            self.assertIn("lossless", get_video_encoding_options(0, "fast")[2])
        self.assertEqual(get_video_encoding_options(20, "veryfast", encoder="libx264"),
                         ("libx264", "veryfast", ["-crf", "20"]))

    def test_cancellation_in_probe_is_propagated_and_never_cached(self):
        for exception in (RuntimeError("stopped"), ValueError("stopped"), OSError("stopped")):
            with self.subTest(exception=type(exception)), patch(
                "apps.video_fast_export._run_ffmpeg", side_effect=exception,
            ), self.assertRaises(type(exception)):
                select_video_encoder()
            self.assertEqual(_ENCODER_CACHE, {})

    def test_explicit_cpu_skips_device_probe(self):
        with patch("apps.video_fast_export._probe_nvenc") as probe:
            self.assertEqual(select_video_encoder("libx264"), "libx264")
            probe.assert_not_called()

    def test_probe_uses_actual_padded_resolution_and_caches_each_size_separately(self):
        with patch("apps.video_fast_export._run_ffmpeg", side_effect=[_FFmpegFailure("too small"), ""]) as run:
            self.assertEqual(select_video_encoder(size=(63, 47)), "libx264")
            self.assertEqual(select_video_encoder(size=(1920, 1080)), "h264_nvenc")
            self.assertEqual(select_video_encoder(size=(63, 47)), "libx264")
            self.assertEqual(run.call_count, 2)
            self.assertIn("color=c=black:s=64x48:r=30:d=0.05", run.call_args_list[0].args[0])
            self.assertIn("color=c=black:s=1920x1080:r=30:d=0.05", run.call_args_list[1].args[0])

    def test_expression_decision_depth_grows_logarithmically(self):
        source = np.arange(1025, dtype=float)
        output = np.cumsum(np.r_[0, np.tile([0.5, 1.5], 512)])
        expression = _time_expression(source, output)
        depth = maximum = 0
        for character in expression:
            if character == "(":
                depth += 1
                maximum = max(maximum, depth)
            elif character == ")":
                depth -= 1
        self.assertLess(maximum, 18)
        self.assertEqual(depth, 0)
        self.assertEqual(expression.count("if("), 1023)


class NativeRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from moviepy import VideoFileClip

        cls.VideoFileClip = VideoFileClip
        cls.folder = tempfile.TemporaryDirectory()
        cls.base = Path(cls.folder.name)
        cls.colors = cls.base / "video nguồn.mp4"
        cls._ffmpeg([
            "-f", "lavfi", "-i", "color=c=red:s=320x240:r=20:d=1",
            "-f", "lavfi", "-i", "color=c=lime:s=320x240:r=20:d=1",
            "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=20:d=1",
            "-f", "lavfi", "-i", "color=c=yellow:s=320x240:r=20:d=1",
            "-filter_complex", "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[v]",
            "-map", "[v]", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
            "-pix_fmt", "yuv420p", str(cls.colors),
        ])
        cls.audio = cls.base / "audio đúng.m4a"
        cls._ffmpeg(["-f", "lavfi", "-i", "sine=frequency=660:sample_rate=48000:duration=4",
                     "-c:a", "aac", "-b:a", "192k", str(cls.audio)])
        cls.fractional = cls.base / "fractional.mp4"
        cls._ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=96x64:r=24000/1001:d=1",
                     "-c:v", "libx264", "-preset", "ultrafast", str(cls.fractional)])
        cls.odd = cls.base / "odd.mp4"
        cls._ffmpeg(["-f", "lavfi", "-i", "testsrc=s=83x51:r=10:d=1",
                     "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv444p", str(cls.odd)])

    @classmethod
    def _ffmpeg(cls, args):
        result = subprocess.run([_ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y", *args],
                                capture_output=True, timeout=20)
        if result.returncode:
            raise AssertionError(result.stderr.decode("utf-8", errors="replace"))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def setUp(self):
        self.output = self.base / f"output_{self._testMethodName}.mp4"
        self.addCleanup(self.output.unlink, missing_ok=True)

    def _open(self, path):
        clip = self.VideoFileClip(str(path))
        _guard_reader_cleanup(clip)
        self.addCleanup(clip.close)
        return clip

    def test_multiple_anchor_intervals_and_tail_keep_correct_video_audio(self):
        mapping = build_marker_time_map(
            4, [TimeMarker(1, 1000), TimeMarker(2, 2000), TimeMarker(3, 3000)],
            [TimeMarker(1, 500), TimeMarker(2, 2500), TimeMarker(3, 3000)],
        )
        updates = []
        codec = render_video_ffmpeg(self.colors, self.output, mapping, 20, self.audio,
                                   preset="veryfast", encoder="libx264", progress=updates.append)
        self.assertEqual(codec, "libx264")
        clip = self._open(self.output)
        self.assertAlmostEqual(clip.duration, 4, delta=1 / 20)
        for time, expected in ((0.2, [255, 0, 0]), (1.0, [0, 255, 0]),
                               (2.75, [0, 0, 255]), (3.5, [255, 255, 0])):
            with self.subTest(time=time):
                np.testing.assert_allclose(clip.get_frame(time)[20, 20], expected, atol=6)
        self.assertIsNotNone(clip.audio)
        audio = np.asarray(clip.audio.get_frame(np.arange(0.3, 0.8, 1 / 48000)))
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        peak = np.fft.rfftfreq(len(audio), 1 / 48000)[np.abs(np.fft.rfft(audio)).argmax()]
        self.assertAlmostEqual(peak, 660, delta=3)
        self.assertEqual(updates[0], 0)
        self.assertEqual(updates[-1], 1)
        self.assertEqual(updates, sorted(updates))
        self.assertEqual(list(self.base.glob("*.fffilter")), [])

    def test_native_smooth_ramps_keep_colors_duration_and_silent_output(self):
        mapping = build_multi_speed_time_map(4, [
            SpeedSegment(0.2, 1.8, 2, 0.4), SpeedSegment(2.2, 3.8, 0.5, 0.4),
        ])
        render_video_ffmpeg(self.colors, self.output, mapping, 20, encoder="libx264")
        clip = self._open(self.output)
        self.assertIsNone(clip.audio)
        self.assertAlmostEqual(clip.duration, mapping.output_duration, delta=1 / 20)
        for source_t, expected in ((0.5, [255, 0, 0]), (1.5, [0, 255, 0]),
                                   (2.5, [0, 0, 255]), (3.5, [255, 255, 0])):
            np.testing.assert_allclose(clip.get_frame(float(mapping.output_time(source_t)))[20, 20],
                                       expected, atol=6)

    def test_fractional_frame_rate_and_video_only_export(self):
        source = self._open(self.fractional)
        mapping = build_multi_speed_time_map(source.duration, [])
        render_video_ffmpeg(self.fractional, self.output, mapping, source.fps, encoder="libx264")
        output = self._open(self.output)
        self.assertAlmostEqual(output.fps, 24000 / 1001, delta=1e-9)
        self.assertAlmostEqual(output.duration, mapping.output_duration, delta=1 / source.fps)
        self.assertIsNone(output.audio)

    def test_odd_dimensions_are_padded_for_h264(self):
        mapping = build_multi_speed_time_map(1, [])
        render_video_ffmpeg(self.odd, self.output, mapping, 10, encoder="libx264")
        output = self._open(self.output)
        self.assertEqual(output.size, [84, 52])

    def test_large_anchor_file_is_passed_as_filter_script(self):
        # An argv expression of this size exceeds the Windows command-line limit.
        source = np.linspace(0, 4, 1025)
        output = np.cumsum(np.r_[0, np.tile([0.002, 0.0058125], 512)])
        mapping = SimpleNamespace(source_knots=source, output_knots=output,
                                  source_duration=4, output_duration=float(output[-1]), segments=())
        self.assertGreater(len(_video_filter(mapping, 20)), 32768)
        render_video_ffmpeg(self.colors, self.output, mapping, 20, encoder="libx264")
        result = self._open(self.output)
        self.assertAlmostEqual(result.duration, mapping.output_duration, delta=1 / 20)

    def test_auto_gpu_path_encodes_real_output_when_available(self):
        mapping = build_multi_speed_time_map(4, [SpeedSegment(0, 4, 2, 0)])
        codec = render_video_ffmpeg(self.colors, self.output, mapping, 20)
        self.assertIn(codec, ("h264_nvenc", "libx264"))
        result = self._open(self.output)
        self.assertAlmostEqual(result.duration, 2, delta=1 / 20)

    def test_cancel_terminates_native_process_and_removes_partial_and_script(self):
        processes = []
        real_popen = subprocess.Popen
        checks = 0

        def track(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        def stop():
            nonlocal checks
            checks += 1
            if checks >= 4:
                raise RuntimeError("cancelled")

        mapping = build_multi_speed_time_map(4, [])
        with patch("apps.video_fast_export.subprocess.Popen", side_effect=track), self.assertRaisesRegex(
            RuntimeError, "cancelled",
        ):
            render_video_ffmpeg(self.colors, self.output, mapping, 20,
                               encoder="libx264", check_stop=stop)
        self.assertTrue(processes)
        self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.base.glob("*.fffilter")), [])

    def test_actual_native_failure_surfaces_diagnostics_without_cpu_retry(self):
        invalid = self.base / "invalid.mp4"
        invalid.write_bytes(b"not a video")
        self.addCleanup(invalid.unlink, missing_ok=True)
        mapping = build_multi_speed_time_map(4, [])
        with patch("apps.video_fast_export.get_video_encoding_options",
                   return_value=("libx264", "fast", ["-crf", "20"])) as options:
            with self.assertRaisesRegex(ValueError, "FFmpeg"):
                render_video_ffmpeg(invalid, self.output, mapping, 20, size=(320, 240))
            self.assertEqual(options.call_count, 1)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.base.glob("*.fffilter")), [])

    def test_existing_source_destination_is_preserved(self):
        original = self.colors.read_bytes()
        mapping = build_multi_speed_time_map(4, [])
        with self.assertRaises(ValueError):
            render_video_ffmpeg(self.colors, self.colors, mapping, 20, encoder="libx264")
        self.assertEqual(self.colors.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
