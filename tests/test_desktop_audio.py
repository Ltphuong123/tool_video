"""Playback checks without an audio device."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
import soundfile as sf
from apps.desktop_audio import WavePlayer, PlaybackStopped


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "srt.wav"
        self.player = WavePlayer()
        self.chunks = []
        self.player.write = lambda audio, rate: self.chunks.append(audio.copy())
        self.player.finish = Mock()
        self.player.close = Mock()

    def tearDown(self):
        self.temp.cleanup()

    def test_srt_preview_skips_only_initial_silence_without_editing_file(self):
        rate = 8000
        source = np.concatenate([np.zeros(rate*12), np.full(rate,.1),
                                 np.zeros(rate*3), np.full(rate,.2)]).astype(np.float32)
        sf.write(self.path, source, rate, subtype="FLOAT")
        original = self.path.read_bytes()
        starts = []
        self.player.play_file(self.path, skip_initial_silence=True, on_start=starts.append)
        self.assertAlmostEqual(starts[0], 11.9)
        np.testing.assert_array_equal(np.concatenate(self.chunks), source[int(starts[0]*rate):])
        self.assertEqual(self.path.read_bytes(), original)
        self.player.close.assert_called_once()

    def test_ordinary_playback_preserves_initial_silence(self):
        source = np.concatenate([np.zeros(8000), np.full(8000,.1)]).astype(np.float32)
        sf.write(self.path, source, 8000, subtype="FLOAT")
        self.player.play_file(self.path)
        np.testing.assert_array_equal(np.concatenate(self.chunks), source)

    def test_silent_file_reports_error_instead_of_waiting(self):
        sf.write(self.path, np.zeros(24000), 8000)
        with self.assertRaises(ValueError):
            self.player.play_file(self.path, skip_initial_silence=True)
        self.assertEqual(self.chunks, [])
        self.player.close.assert_called_once()

    def test_stop_interrupts_search(self):
        sf.write(self.path, np.zeros(8000), 8000)
        self.player.stop()
        with self.assertRaises(PlaybackStopped):
            self.player.play_file(self.path, skip_initial_silence=True)
        self.player.close.assert_called_once()
