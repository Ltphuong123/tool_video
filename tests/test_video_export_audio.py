"""Direct AAC encoding preserves mapped samples and releases its subprocess."""
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from apps.video_editor import build_multi_speed_time_map
from apps.video_export_audio import encode_mapped_audio


class Cancelled(RuntimeError):
    pass


class ConstantAudio:
    nchannels = 2
    duration = 1

    def get_frame(self, times):
        return np.column_stack((np.full(len(times), 0.125), np.full(len(times), -0.25)))


class CapturedInput(io.BytesIO):
    def __init__(self):
        super().__init__()
        self.blocks = []

    def write(self, data):
        self.blocks.append(data)
        return super().write(data)


class FakeProcess:
    def __init__(self, command, *, returncode=0, **kwargs):
        self.stdin = CapturedInput()
        self.input = self.stdin
        self.stdout = self.stderr = None
        self.returncode = None
        self.final_returncode = returncode
        self.terminated = self.killed = False
        Path(command[-1]).write_bytes(b"aac")

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self.final_returncode
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -1

    def kill(self):
        self.killed = True
        self.returncode = -1


@unittest.skipUnless(importlib.util.find_spec("moviepy"), "optional MoviePy not installed")
class AudioEncoderTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.directory = Path(self.folder.name)
        self.destination = self.directory / "encoded.m4a"
        self.mapping = build_multi_speed_time_map(1, [])

    def tearDown(self):
        self.folder.cleanup()

    def test_streams_bounded_pcm_blocks_without_intermediate_wav_or_decoder(self):
        process = None

        def start(command, **kwargs):
            nonlocal process
            self.assertIn("s32le", command)
            self.assertIn("192k", command)
            process = FakeProcess(command, **kwargs)
            return process

        def chunks(*args, **kwargs):
            yield np.full((30000, 2), 0.125, dtype=np.float32)
            self.assertEqual(sum(len(block) for block in process.input.blocks), 30000 * 8)
            yield np.full((18000, 2), -0.25, dtype=np.float32)

        progress = []
        with patch("apps.video_export_audio.subprocess.Popen", side_effect=start), \
                patch("apps.video_editor._ramped_audio_chunks", side_effect=chunks), \
                patch("moviepy.AudioFileClip", side_effect=AssertionError("no decoder")):
            encode_mapped_audio(ConstantAudio(), self.destination, self.mapping,
                                progress=progress.append)
        self.assertTrue(self.destination.is_file())
        self.assertEqual([len(block) // 8 for block in process.input.blocks], [24000, 6000, 18000])
        decoded = np.frombuffer(b"".join(process.input.blocks), dtype="<i4").reshape(-1, 2)
        np.testing.assert_array_equal(decoded[:30000], 2 ** 28)
        np.testing.assert_array_equal(decoded[30000:], -(2 ** 29))
        self.assertEqual(progress[-1], 1)
        self.assertTrue(process.input.closed)
        self.assertFalse(process.terminated)
        self.assertEqual(list(self.directory.iterdir()), [self.destination])

    def test_invalid_or_incorrect_sample_chunks_fail_and_remove_partial(self):
        invalid = (
            [np.zeros((48000, 1), dtype=np.float32)],
            [np.zeros((48000, 2), dtype=np.int32)],
            [np.full((48000, 2), np.nan, dtype=np.float32)],
            [np.zeros((0, 2), dtype=np.float32)],
            [np.zeros((47999, 2), dtype=np.float32)],
            [np.zeros((48001, 2), dtype=np.float32)],
        )
        for chunks in invalid:
            with self.subTest(shape=chunks[0].shape, dtype=chunks[0].dtype):
                captured = []

                def start(command, **kwargs):
                    process = FakeProcess(command, **kwargs)
                    captured.append(process)
                    return process

                with patch("apps.video_export_audio.subprocess.Popen", side_effect=start), \
                        patch("apps.video_editor._ramped_audio_chunks", return_value=iter(chunks)), \
                        self.assertRaises(ValueError):
                    encode_mapped_audio(ConstantAudio(), self.destination, self.mapping)
                self.assertEqual(list(self.directory.iterdir()), [])
                self.assertTrue(captured[0].terminated)
                self.assertTrue(captured[0].input.closed)

    def test_failed_encoder_returncode_is_not_reported_as_success(self):
        def start(command, **kwargs):
            return FakeProcess(command, returncode=7, **kwargs)

        with patch("apps.video_export_audio.subprocess.Popen", side_effect=start), \
                self.assertRaisesRegex(ValueError, "FFmpeg.*7"):
            encode_mapped_audio(ConstantAudio(), self.destination, self.mapping)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_preexisting_destination_is_protected(self):
        self.destination.write_bytes(b"original")
        with patch("apps.video_export_audio.subprocess.Popen") as start, \
                self.assertRaises(ValueError):
            encode_mapped_audio(ConstantAudio(), self.destination, self.mapping)
        start.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), b"original")

    def test_destination_created_during_encoding_is_protected(self):
        def progress(fraction):
            if not self.destination.exists():
                self.destination.write_bytes(b"new file")

        with patch("apps.video_export_audio.subprocess.Popen", side_effect=FakeProcess), \
                self.assertRaises(ValueError):
            encode_mapped_audio(ConstantAudio(), self.destination, self.mapping, progress=progress)
        self.assertEqual(self.destination.read_bytes(), b"new file")
        self.assertEqual(list(self.directory.iterdir()), [self.destination])

    def test_cancellation_terminates_reaps_and_releases_real_encoder(self):
        # MoviePy resolves its FFmpeg binary once, starting probe processes on
        # its first import. Capture only the audio encoder under test.
        import moviepy.config

        processes = []
        original_popen = subprocess.Popen
        stopped = False

        def start(*args, **kwargs):
            process = original_popen(*args, **kwargs)
            processes.append((process, process.stdin))
            return process

        def progress(fraction):
            nonlocal stopped
            stopped = fraction > 0

        def check_stop():
            if stopped:
                raise Cancelled("stop")

        with patch("apps.video_export_audio.subprocess.Popen", side_effect=start), \
                self.assertRaises(Cancelled):
            encode_mapped_audio(ConstantAudio(), self.destination, self.mapping,
                                check_stop=check_stop, progress=progress)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0][0].poll())
        self.assertTrue(processes[0][1].closed)
        self.assertEqual(list(self.directory.iterdir()), [])


@unittest.skipUnless(importlib.util.find_spec("moviepy"), "optional MoviePy not installed")
class MappedAacIntegrationTests(unittest.TestCase):
    def decode(self, path, channels=2):
        from moviepy.config import FFMPEG_BINARY
        from moviepy.tools import cross_platform_popen_params

        result = subprocess.run([
            FFMPEG_BINARY, "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-vn", "-ar", "48000", "-ac", str(channels),
            "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
        ], check=True, **cross_platform_popen_params({
            "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
        }))
        return np.frombuffer(result.stdout, dtype="<f4").reshape(-1, channels)

    @unittest.skipUnless(importlib.util.find_spec("pedalboard"), "optional pedalboard not installed")
    def test_real_stereo_aac_retains_pitch_pulse_anchors_and_expected_duration(self):
        from apps.video_editor import _encode_audio, _guard_reader_cleanup, _write_ramped_audio
        from apps.video_markers import TimeMarker, build_marker_time_map
        from moviepy import AudioFileClip

        class PulseStereo:
            nchannels = 2
            duration = 3

            def get_frame(self, times):
                envelope = 0.15 + sum(np.exp(-((times - center) / 0.035) ** 2)
                                      for center in (0.3, 1.2, 2.2))
                return np.column_stack((0.2 * envelope * np.sin(2 * np.pi * 220 * times),
                                        0.15 * envelope * np.sin(2 * np.pi * 440 * times)))

        mapping = build_marker_time_map(
            3, [TimeMarker(1, 300), TimeMarker(2, 1200), TimeMarker(3, 2200)],
            [TimeMarker(1, 150), TimeMarker(2, 1600), TimeMarker(3, 2400)],
        )
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            output = directory / "direct.m4a"
            wave = directory / "legacy.wav"
            legacy = directory / "legacy.m4a"
            encode_mapped_audio(PulseStereo(), output, mapping)
            self.assertEqual(list(directory.iterdir()), [output])
            actual = self.decode(output)
            _write_ramped_audio(PulseStereo(), wave, mapping)
            with AudioFileClip(str(wave), fps=48000) as clip:
                _guard_reader_cleanup(clip)
                _encode_audio(clip, legacy, lambda: None, lambda fraction: None)
            expected = self.decode(legacy)
        total = round(mapping.output_duration * 48000)
        self.assertGreaterEqual(len(actual), total)
        self.assertLessEqual(len(actual) - total, 1024)
        self.assertGreater(np.max(np.abs(actual[:, 0])), 0.08)
        self.assertGreater(np.max(np.abs(actual[:, 1])), 0.06)
        self.assertGreater(np.max(np.abs(actual[:, 0] - actual[:, 1])), 0.06)
        for marker_time in (0.15, 1.6, 2.4):
            first = max(0, round((marker_time - 0.08) * 48000))
            last = round((marker_time + 0.08) * 48000)
            actual_peak = np.argmax(np.abs(actual[first:last, 0])) + first
            expected_peak = np.argmax(np.abs(expected[first:last, 0])) + first
            self.assertAlmostEqual(actual_peak / 48000, marker_time, delta=1 / 30)
            self.assertAlmostEqual(actual_peak / 48000, expected_peak / 48000, delta=0.005)
        # Measure pitch inside a stretched interval, away from transients and
        # the adjoining speeds where a short FFT has poor resolution.
        for channel, frequency in ((0, 220), (1, 440)):
            section = actual[round(0.5 * 48000):round(1.1 * 48000), channel]
            spectrum = np.abs(np.fft.rfft(section * np.hanning(len(section))))
            measured = np.fft.rfftfreq(len(section), 1 / 48000)[spectrum.argmax()]
            self.assertAlmostEqual(measured, frequency, delta=3)

    def test_real_mono_aac_is_nonzero_and_stays_within_one_aac_frame(self):
        class SineMono:
            nchannels = 1
            duration = 1.003

            def get_frame(self, times):
                return (0.2 * np.sin(2 * np.pi * 440 * times))[:, None]

        mapping = build_multi_speed_time_map(SineMono.duration, [])
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "mono.m4a"
            encode_mapped_audio(SineMono(), output, mapping)
            actual = self.decode(output, channels=1)
        total = round(mapping.output_duration * 48000)
        self.assertGreaterEqual(len(actual), total)
        self.assertLessEqual(len(actual) - total, 1024)
        self.assertGreater(np.max(np.abs(actual)), 0.1)


if __name__ == "__main__":
    unittest.main()
