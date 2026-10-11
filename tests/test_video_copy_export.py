"""Lossless packet retiming: picture hashes, timestamps, size and cancellation."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos

from apps.video_copy_export import render_video_copy
from apps.video_editor import SpeedSegment, build_multi_speed_time_map, render_speed_segments
from apps.video_fast_export import _ffmpeg_binary
from apps.video_markers import TimeMarker, build_marker_time_map


class VideoCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.folder.name)
        cls.source = cls.directory / "source with B frames.mp4"
        cls.run_ffmpeg("-f", "lavfi", "-i", "testsrc2=s=320x240:r=24:d=3",
                       "-c:v", "libx264", "-preset", "medium", "-bf", "3",
                       "-g", "24", "-an", str(cls.source))
        cls.packets = cls.hashes(cls.source, compressed=True)
        cls.pictures = cls.hashes(cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    @classmethod
    def run_ffmpeg(cls, *args):
        result = subprocess.run([_ffmpeg_binary(), "-hide_banner", "-loglevel", "error",
                                 "-nostdin", "-y", *args], capture_output=True, check=True)
        return result.stdout.decode("utf-8")

    @classmethod
    def hashes(cls, path, compressed=False):
        options = ["-c:v", "copy"] if compressed else ["-fps_mode", "passthrough",
                                                        "-enc_time_base", "1:1000000"]
        text = cls.run_ffmpeg("-i", str(path), "-map", "0:v:0", *options,
                              "-f", "framehash", "-hash", "sha256", "-")
        match = re.search(r"#tb 0: (\d+)/(\d+)", text)
        base = int(match[1]) / int(match[2])
        records = []
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            columns = [part.strip() for part in line.split(",")]
            records.append((int(columns[1]) * base, int(columns[2]) * base,
                            int(columns[3]) * base, columns[-1]))
        return records

    def setUp(self):
        self.output = self.directory / f"{self._testMethodName}.mp4"
        self.addCleanup(self.output.unlink, missing_ok=True)

    def assert_preserved(self, mapping):
        packets = self.hashes(self.output, compressed=True)
        self.assertEqual([row[-1] for row in packets], [row[-1] for row in self.packets])
        self.assertEqual([row[-1] for row in self.hashes(self.output)],
                         [row[-1] for row in self.pictures])
        for before, after in zip(self.packets, packets):
            self.assertAlmostEqual(after[1], float(mapping.output_time(before[1])), delta=0.000003)
        self.assertTrue(all(b[0] > a[0] for a, b in zip(packets, packets[1:])))
        info = ffmpeg_parse_infos(str(self.output))
        self.assertAlmostEqual(info["duration"], mapping.output_duration, delta=0.011)
        self.assertEqual(info["video_size"], [320, 240])
        self.assertEqual(info["video_codec_name"], "h264")
        self.assertFalse(info["audio_found"])
        self.assertLess(self.output.stat().st_size, self.source.stat().st_size * 1.1)
        self.assertEqual(list(self.directory.glob("*.bsf")), [])

    def test_marker_alignment_preserves_all_packets_pixels_and_b_frame_tail_duration(self):
        mapping = build_marker_time_map(3, [TimeMarker(1, 1000), TimeMarker(2, 2000)],
                                       [TimeMarker(1, 500), TimeMarker(2, 2500)])
        render_video_copy(self.source, self.output, mapping)
        self.assert_preserved(mapping)

    def test_smooth_ramps_preserve_all_packets_pixels_and_duration(self):
        mapping = build_multi_speed_time_map(3, [SpeedSegment(0.3, 2.7, 0.5, 0.4)])
        render_video_copy(self.source, self.output, mapping)
        self.assert_preserved(mapping)

    def test_identity_preserves_payload_and_duration(self):
        mapping = build_multi_speed_time_map(3, [])
        render_video_copy(self.source, self.output, mapping)
        self.assert_preserved(mapping)

    def test_cancel_reaps_process_and_removes_owned_output_and_filter(self):
        mapping = build_multi_speed_time_map(3, [])
        stopped = False
        processes = []
        popen = subprocess.Popen

        def track(*args, **kwargs):
            process = popen(*args, **kwargs)
            processes.append(process)
            return process

        def progress(value):
            nonlocal stopped
            stopped = value > 0

        def stop():
            if stopped:
                raise RuntimeError("cancelled")

        with patch("apps.video_fast_export.subprocess.Popen", side_effect=track):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                render_video_copy(self.source, self.output, mapping,
                                  check_stop=stop, progress=progress)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.directory.glob("*.bsf")), [])
        self.assertTrue(all(process.poll() is not None for process in processes))

    def test_existing_output_is_never_overwritten(self):
        self.output.write_bytes(b"existing")
        with self.assertRaises(ValueError):
            render_video_copy(self.source, self.output, build_multi_speed_time_map(3, []))
        self.assertEqual(self.output.read_bytes(), b"existing")

    def test_editor_copy_mode_never_opens_rgb_decoder_or_encoder(self):
        with patch("moviepy.VideoFileClip", side_effect=AssertionError("RGB decoder unused")), \
             patch("apps.video_fast_export.get_video_encoding_options", side_effect=AssertionError("encoder unused")):
            mapping = render_speed_segments(
                self.source, self.output, [SpeedSegment(0.3, 2.7, 0.5, 0.4)],
                export_mode="copy", quality=None, preset=None,
            )
        self.assert_preserved(mapping)

    def test_ten_bit_hevc_pictures_codec_and_color_metadata_are_preserved(self):
        source = self.directory / "ten bit.mp4"
        self.addCleanup(source.unlink, missing_ok=True)
        self.run_ffmpeg("-f", "lavfi", "-i", "testsrc2=s=160x120:r=24:d=2",
                       "-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p10le",
                       "-x265-params", "pools=1:frame-threads=1:log-level=error:colorprim=9:transfer=16:colormatrix=9",
                       "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                       "-colorspace", "bt2020nc", str(source))
        mapping = build_multi_speed_time_map(2, [SpeedSegment(0, 2, 0.5, 0)])
        render_video_copy(source, self.output, mapping)
        self.assertEqual([row[-1] for row in self.hashes(source, compressed=True)],
                         [row[-1] for row in self.hashes(self.output, compressed=True)])
        self.assertEqual([row[-1] for row in self.hashes(source)],
                         [row[-1] for row in self.hashes(self.output)])
        info = ffmpeg_parse_infos(str(self.output))
        self.assertEqual(info["video_codec_name"], "hevc")
        self.assertIn("Main 10", info["video_profile"])
        self.assertAlmostEqual(info["duration"], 4, delta=0.011)
        # Probe stream descriptors without decoding: verify ten-bit transfer
        # and BT.2020 flags survived alongside the exact packet payloads.
        result = subprocess.run([_ffmpeg_binary(), "-hide_banner", "-i", str(self.output)],
                                capture_output=True)
        description = result.stderr.decode("utf-8", errors="replace")
        self.assertIn("yuv420p10le", description)
        self.assertIn("bt2020", description)
        self.assertIn("smpte2084", description)

    def test_fractional_frame_rate_packet_times_remain_exact(self):
        source = self.directory / "fractional.mp4"
        self.addCleanup(source.unlink, missing_ok=True)
        self.run_ffmpeg("-f", "lavfi", "-i", "testsrc2=s=160x120:r=30000/1001:d=3",
                       "-c:v", "libx264", "-bf", "3", str(source))
        duration = ffmpeg_parse_infos(str(source))["duration"]
        mapping = build_marker_time_map(duration, [TimeMarker(1, 1000), TimeMarker(2, 2000)],
                                       [TimeMarker(1, 900), TimeMarker(2, 1600)])
        render_video_copy(source, self.output, mapping)
        original, output = self.hashes(source, compressed=True), self.hashes(self.output, compressed=True)
        self.assertEqual([row[-1] for row in original], [row[-1] for row in output])
        for before, after in zip(original, output):
            self.assertAlmostEqual(after[1], float(mapping.output_time(before[1])), delta=0.000003)
        self.assertAlmostEqual(ffmpeg_parse_infos(str(self.output))["duration"],
                               mapping.output_duration, delta=0.011)

    def test_muxing_optional_audio_keeps_copied_video_and_matching_duration(self):
        audio = self.directory / "retimed sound.m4a"
        self.addCleanup(audio.unlink, missing_ok=True)
        mapping = build_multi_speed_time_map(3, [SpeedSegment(0, 3, 0.5, 0)])
        self.run_ffmpeg("-f", "lavfi", "-i", "sine=r=48000:d=6", "-c:a", "aac", str(audio))
        render_video_copy(self.source, self.output, mapping, audio_path=audio)
        self.assertEqual([row[-1] for row in self.hashes(self.output, compressed=True)],
                         [row[-1] for row in self.packets])
        info = ffmpeg_parse_infos(str(self.output))
        self.assertTrue(info["audio_found"])
        self.assertAlmostEqual(info["duration"], 6, delta=0.025)

    def test_nonzero_source_start_is_normalized_without_changing_frames(self):
        source = self.directory / "offset.mp4"
        self.addCleanup(source.unlink, missing_ok=True)
        self.run_ffmpeg("-i", str(self.source), "-c:v", "copy", "-output_ts_offset", "5", str(source))
        mapping = build_multi_speed_time_map(3, [SpeedSegment(0, 3, 1.5, 0)])
        render_video_copy(source, self.output, mapping)
        self.assert_preserved(mapping)

    def test_odd_dimensions_are_preserved_without_padding_or_chroma_conversion(self):
        source = self.directory / "odd.mp4"
        self.addCleanup(source.unlink, missing_ok=True)
        self.run_ffmpeg("-f", "lavfi", "-i", "testsrc=s=65x49:r=24:d=2",
                       "-c:v", "libx264", "-pix_fmt", "yuv444p", str(source))
        mapping = build_multi_speed_time_map(2, [SpeedSegment(0, 2, 1.5, 0)])
        render_video_copy(source, self.output, mapping)
        self.assertEqual(ffmpeg_parse_infos(str(self.output))["video_size"], [65, 49])
        self.assertEqual([row[-1] for row in self.hashes(source)],
                         [row[-1] for row in self.hashes(self.output)])

    def test_failed_source_leaves_no_partial_and_never_retries_an_encoder(self):
        source = self.directory / "bad.mp4"
        source.write_bytes(b"not a video")
        self.addCleanup(source.unlink, missing_ok=True)
        with patch("apps.video_fast_export.get_video_encoding_options",
                   side_effect=AssertionError("encoder unused")):
            with self.assertRaisesRegex(ValueError, "FFmpeg"):
                render_video_copy(source, self.output, build_multi_speed_time_map(3, []))
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.directory.glob("*.bsf")), [])

    def test_tool_all_three_operations_can_copy_without_loading_a_tts_model(self):
        from apps.v3turbo_tool import TurboTool

        tool = TurboTool(self.directory)
        segments = [SpeedSegment(0.3, 2.7, 0.5, 0.4)]
        operations = (
            lambda: tool.edit_video(self.source, 0.3, 2.7, 0.5, 0.4, export_mode="copy"),
            lambda: tool.edit_video_segments(self.source, segments, export_mode="copy"),
            lambda: tool.edit_video_markers(self.source, [TimeMarker(1, 1000)],
                                           [TimeMarker(1, 2000)], export_mode="copy"),
        )
        for operation in operations:
            output, note = operation()
            self.addCleanup(output.unlink, missing_ok=True)
            self.assertEqual([row[-1] for row in self.hashes(output, compressed=True)],
                             [row[-1] for row in self.packets])
            self.assertIsNone(tool.tts)
            self.assertIn("Giữ nguyên hình", note)

    def test_invalid_mode_is_rejected_before_rendering(self):
        with self.assertRaises(ValueError):
            render_speed_segments(self.source, self.output, [], export_mode="bad")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
