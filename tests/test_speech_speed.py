"""Signal-level Rubber Band checks; listening still determines voice quality."""
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from apps.speech_speed import require_rubberband, rubberband_to_samples, stretch_rubberband


class SpeechSpeedTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("pedalboard"), "optional pedalboard not installed")
    def test_changes_duration_and_preserves_pitch_at_both_rates(self):
        for rate in (16000, 48000):
            original = (0.2 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)).astype(np.float32)
            for speed in (0.8, 1.25):
                with self.subTest(rate=rate, speed=speed):
                    output = stretch_rubberband(original, speed, rate)
                    self.assertEqual(len(output), round(len(original) / speed))
                    middle = output[len(output) // 4:3 * len(output) // 4]
                    spectrum = np.abs(np.fft.rfft(middle * np.hanning(len(middle))))
                    found = np.fft.rfftfreq(len(middle), 1 / rate)[spectrum.argmax()]
                    self.assertAlmostEqual(found, 440, delta=5)

    def test_original_speed_and_empty_audio_need_no_optional_dependency(self):
        source = np.random.default_rng(7).normal(0, 0.1, 2000).astype(np.float32)
        with patch.dict("sys.modules", {"pedalboard": None}):
            np.testing.assert_array_equal(stretch_rubberband(source, 1), source)
            np.testing.assert_array_equal(rubberband_to_samples(source, len(source)), source)
            self.assertEqual(stretch_rubberband([], 1.2).size, 0)

    def test_missing_backend_has_actionable_error(self):
        with patch.dict("sys.modules", {"pedalboard": None}):
            with self.assertRaisesRegex(ValueError, "Pedalboard"):
                require_rubberband()

    def test_silence_keeps_exact_duration_without_running_native_engine(self):
        with patch("apps.speech_speed.require_rubberband", side_effect=AssertionError("unexpected processing")):
            for length in (0, 1, 27, 48000):
                for speed in (.5, .9, 1.15, 2):
                    with self.subTest(length=length, speed=speed):
                        audio = stretch_rubberband(np.zeros(length, dtype=np.float32), speed)
                        self.assertEqual(len(audio), max(1, round(length / speed)) if length else 0)
                        self.assertTrue(np.all(audio == 0))

    def test_rejects_invalid_inputs_before_processing(self):
        for source, speed, rate in (
                ([0.1], 0, 48000), ([0.1], 2.1, 48000), ([0.1], float("nan"), 48000),
                ([np.inf], 1.2, 48000), (np.ones((2, 10)), 1.2, 48000),
                ([0.1], 1.2, 0), ([0.1], 1.2, float("nan"))):
            with self.subTest(speed=speed, rate=rate), self.assertRaises(ValueError):
                stretch_rubberband(source, speed, rate)
        for target in (0, -1, 1.5, True):
            with self.subTest(target=target), self.assertRaises(ValueError):
                rubberband_to_samples([0.1, 0.2], target)

    def test_cancellation_is_checked_before_native_processing(self):
        def stop():
            raise InterruptedError("stop")
        with patch.dict("sys.modules", {"pedalboard": None}):
            with self.assertRaises(InterruptedError):
                stretch_rubberband([0.1, 0.2], 1.2, check_stop=stop)

    @unittest.skipUnless(importlib.util.find_spec("pedalboard"), "optional pedalboard not installed")
    def test_cancellation_is_checked_after_native_processing(self):
        checks = []
        def stop():
            checks.append(True)
            if len(checks) == 2:
                raise InterruptedError("stop")
        with self.assertRaises(InterruptedError):
            rubberband_to_samples(np.full(48000, 0.1, dtype=np.float32), 24000, check_stop=stop)
        self.assertEqual(len(checks), 2)


if __name__ == "__main__":
    unittest.main()
