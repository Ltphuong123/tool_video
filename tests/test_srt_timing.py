"""Sample-accurate SRT fitting; no model downloads or audio devices."""
import unittest
import importlib.util
import numpy as np

from apps.srt_speech import Cue, parse_srt
from apps.srt_timing import (
    cue_sample_windows, fit_clip_to_samples, lay_fitted_timeline, prepare_srt_clip,
    rubberband_to_samples, soften_srt_tail, trim_srt_silence,
)


class SrtTimingTests(unittest.TestCase):
    def test_long_clip_is_accelerated_to_exact_window_with_pitch_preserved(self):
        rate = 48000
        source = (0.2 * np.sin(2 * np.pi * 220 * np.arange(3 * rate) / rate)).astype(np.float32)
        output, factor = fit_clip_to_samples(source, 2 * rate)
        self.assertEqual(len(output), 2 * rate)
        self.assertAlmostEqual(factor, 1.5)
        middle = output[rate // 2:3 * rate // 2]
        spectrum = np.abs(np.fft.rfft(middle * np.hanning(len(middle))))
        pitch = np.fft.rfftfreq(len(middle), 1 / rate)[spectrum.argmax()]
        self.assertAlmostEqual(pitch, 220, delta=3)

    def test_automatic_speed_can_exceed_manual_two_x_limit(self):
        source = np.linspace(-0.1, 0.1, 120000, dtype=np.float32)
        output, factor = fit_clip_to_samples(source, 24000)
        self.assertEqual(len(output), 24000)
        self.assertEqual(factor, 5)
        self.assertTrue(np.isfinite(output).all())

    def test_compression_includes_the_end_of_the_clip_instead_of_just_truncating(self):
        rate = 48000
        clips = [(0.1 * np.sin(2 * np.pi * pitch * np.arange(rate) / rate)).astype(np.float32)
                 for pitch in (220, 330, 660)]
        output, factor = fit_clip_to_samples(np.concatenate(clips), rate)
        self.assertEqual(factor, 3)
        for section, pitch in ((output[4800:12000], 220), (output[-12000:-4800], 660)):
            spectrum = np.abs(np.fft.rfft(section * np.hanning(len(section))))
            found = np.fft.rfftfreq(len(section), 1 / rate)[spectrum.argmax()]
            self.assertAlmostEqual(found, pitch, delta=8)

    def test_equal_and_short_clips_keep_samples_and_speed(self):
        source = np.array([0.02, -0.04, 0.05], dtype=np.float32)
        for target in (3, 9):
            output, factor = fit_clip_to_samples(source, target)
            np.testing.assert_array_equal(output, source)
            self.assertEqual(factor, 1)
        output, factor = fit_clip_to_samples([], 9)
        self.assertEqual(output.size, 0)
        self.assertEqual(factor, 1)

    def test_voice_padding_is_removed_without_cutting_quiet_consonants(self):
        voice = np.concatenate((np.full(100, 0.0003), np.full(48000, 0.2))).astype(np.float32)
        padded = np.concatenate((np.zeros(14400), voice, np.zeros(19200)))
        np.testing.assert_array_equal(trim_srt_silence(padded), voice)
        output, speed = prepare_srt_clip(padded, 96000)
        np.testing.assert_array_equal(output, voice)
        self.assertEqual(speed, 1)

    def test_initial_minimum_speed_is_kept_when_voice_fits(self):
        voice = np.full(24000, 0.1, dtype=np.float32)
        for speed, expected in ((0.8, 30000), (1.2, 20000), (1.5, 16000)):
            with self.subTest(speed=speed):
                output, actual_speed = prepare_srt_clip(voice, 48000, min_speed=speed)
                self.assertEqual(actual_speed, speed)
                self.assertEqual(len(output), expected)

    @unittest.skipUnless(importlib.util.find_spec("pedalboard"), "optional pedalboard not installed")
    def test_rubberband_preserves_duration_and_pitch_when_accelerating_speech(self):
        rate = 48000
        for pitch in (220, 880):
            voice = (0.2 * np.sin(2 * np.pi * pitch * np.arange(rate) / rate)).astype(np.float32)
            for speed in (0.8, 1.5, 2.5):
                with self.subTest(pitch=pitch, speed=speed):
                    target = round(rate / speed)
                    output = rubberband_to_samples(voice, target)
                    self.assertEqual(len(output), target)
                    self.assertTrue(np.isfinite(output).all())
                    middle = output[len(output) // 4:3 * len(output) // 4]
                    spectrum = np.abs(np.fft.rfft(middle * np.hanning(len(middle))))
                    found = np.fft.rfftfreq(len(middle), 1 / rate)[spectrum.argmax()]
                    self.assertAlmostEqual(found, pitch, delta=5)

    def test_tail_fade_removes_abrupt_end_without_shifting_start_or_samples(self):
        voice = np.concatenate((np.full(48000, 0.118, dtype=np.float32), np.zeros(9600, dtype=np.float32)))
        original = voice.copy()
        output = soften_srt_tail(voice)
        np.testing.assert_array_equal(output[:47760], voice[:47760])
        self.assertEqual(len(output), len(voice))
        self.assertEqual(output[47999], 0)
        self.assertTrue(np.all(output[48000:] == 0))
        self.assertLess(float(np.abs(np.diff(output[47759:48001])).max()), .001)
        np.testing.assert_array_equal(voice, original)

    def test_rubberband_validation_and_cancellation(self):
        for voice, target in (([], 10), ([np.nan], 10), ([0.1], 0), ([0.1], 1.5)):
            with self.assertRaises(ValueError):
                rubberband_to_samples(voice, target)
        def stop():
            raise InterruptedError("stop")
        with self.assertRaises(InterruptedError):
            rubberband_to_samples(np.ones(100), 50, check_stop=stop)

    def test_auto_fit_measures_voice_without_padding_and_never_uses_a_slower_speed(self):
        padded = np.concatenate((np.zeros(24000), np.full(72000, 0.1), np.zeros(24000)))
        for minimum in (0.8, 1.0, 1.2):
            output, actual_speed = prepare_srt_clip(padded, 48000, min_speed=minimum)
            self.assertEqual(len(output), 48000)
            self.assertEqual(actual_speed, 1.5)
        output, actual_speed = prepare_srt_clip(padded, 48000, min_speed=2.0)
        self.assertEqual(len(output), 36000)
        self.assertEqual(actual_speed, 2)

    def test_rubberband_handles_a_slow_initial_speed(self):
        voice = np.full(24000, 0.1, dtype=np.float32)
        output, speed = prepare_srt_clip(voice, 96000, min_speed=0.5)
        self.assertEqual(speed, 0.5)
        self.assertEqual(len(output), 48000)
        self.assertTrue(np.isfinite(output).all())

    def test_silent_clips_and_invalid_minimum_speeds_are_rejected(self):
        for speed in (0, 0.4, 2.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                prepare_srt_clip(np.ones(100), 100, min_speed=speed)
        for voice in (np.zeros(100), [], [np.nan]):
            with self.assertRaises(ValueError):
                prepare_srt_clip(voice, 100)

    def test_timeline_preserves_starts_gaps_and_last_end(self):
        cues = [Cue(1, 1000, 2000, "One"), Cue(2, 3000, 5000, "Two")]
        clips = [np.full(48000, 0.1, dtype=np.float32), np.full(24000, 0.2, dtype=np.float32)]
        track = lay_fitted_timeline(clips, cues)
        self.assertEqual(len(track), 5 * 48000)
        self.assertTrue(np.all(track[:48000] == 0))
        np.testing.assert_array_equal(track[48000:96000], clips[0])
        self.assertTrue(np.all(track[96000:144000] == 0))
        np.testing.assert_array_equal(track[144000:168000], clips[1])
        self.assertTrue(np.all(track[168000:] == 0))

    def test_overlap_uses_next_start_as_deadline(self):
        cues = [Cue(1, 1000, 4000, "One"), Cue(2, 2500, 5000, "Two")]
        self.assertEqual(cue_sample_windows(cues), [(48000, 120000), (120000, 240000)])
        with self.assertRaises(ValueError):
            cue_sample_windows([Cue(1, 0, 1000, "One"), Cue(2, 0, 1000, "Two")])

    def test_strict_parser_preserves_short_windows_and_rejects_invalid_ends(self):
        source = "1\n00:00:01,000 --> 00:00:01,025\nOne\n"
        cue = parse_srt(source, strict_timing=True)[0]
        self.assertEqual(cue.end_ms - cue.start_ms, 25)
        self.assertEqual(parse_srt(source)[0].end_ms, 1100)
        for end in ("00:00:01,000", "00:00:00,500"):
            with self.assertRaises(ValueError):
                parse_srt(f"1\n00:00:01,000 --> {end}\nOne", strict_timing=True)
        clip, factor = fit_clip_to_samples(np.ones(4800, dtype=np.float32) * 0.1, 1200)
        self.assertEqual(len(clip), 1200)
        self.assertEqual(factor, 4)

    def test_missing_blank_lines_do_not_merge_cues_or_lose_long_gaps(self):
        source = ("1\n00:00:05,000 --> 00:00:07,000\nFirst line\nsecond line\n"
                  "2\n00:01:30,500 --> 00:01:32,000\nLast line\n")
        cues = parse_srt(source, strict_timing=True)
        self.assertEqual(len(cues), 2)
        self.assertEqual([cue.text for cue in cues], ["First line second line", "Last line"])
        self.assertEqual([cue.start_ms for cue in cues], [5000, 90500])
        self.assertEqual(cues[-1].end_ms, 92000)

    def test_unindexed_numeric_captions_remain_text(self):
        source = ("00:00:01,000 --> 00:00:02,000\n2025\n"
                  "00:00:10,000 --> 00:00:11,000\n2026\n"
                  "00:00:20,000 --> 00:00:21,000\n2027\n")
        self.assertEqual([cue.text for cue in parse_srt(source)], ["2025", "2026", "2027"])

    def test_invalid_audio_and_windows_are_rejected(self):
        for source, target in [(np.ones((2, 10)), 5), ([np.nan], 5), ([0.1], 0), ([0.1], 1.5)]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                fit_clip_to_samples(source, target)
        cues = [Cue(1, 0, 1000, "One")]
        for clips in ([], [np.ones(48001)], [np.array([np.inf])], [np.zeros(0)]):
            with self.assertRaises(ValueError):
                lay_fitted_timeline(clips, cues)

    def test_cancellation_is_checked_during_time_stretching(self):
        calls = []
        def stop():
            calls.append(True)
            if len(calls) == 2:
                raise InterruptedError("stop")
        with self.assertRaises(InterruptedError):
            fit_clip_to_samples(np.ones(144000, dtype=np.float32), 96000, check_stop=stop)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
