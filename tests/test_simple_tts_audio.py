"""Real audio processing and source cache tests for the minimal app."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from apps.simple_tts_audio import AudioOptions, process_audio, save_variant
from apps.simple_tts import SimpleTTS
import test_simple_tts as fixtures


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.rate = 24000
        time = np.arange(self.rate) / self.rate
        self.audio = (0.2 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
        self.source = self.folder / "original.wav"
        sf.write(self.source, self.audio, self.rate, subtype="FLOAT")

    def tearDown(self):
        self.temp.cleanup()

    def test_bypass_preserves_audio_and_source(self):
        before = self.source.read_bytes()
        path = save_variant(self.source, self.folder, AudioOptions())
        actual, rate = sf.read(path, dtype="float32")
        np.testing.assert_array_equal(actual, self.audio)
        self.assertEqual(self.source.read_bytes(), before)
        report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(report["source"], str(self.source.resolve()))
        self.assertEqual(rate, report["sample_rate"])

    def test_real_rubberband_duration_and_pitch(self):
        for speed in [0.75, 1.25, 2.0]:
            wav, rate = process_audio(self.audio, self.rate, AudioOptions(tempo="rubberband", speed=speed))
            self.assertLess(abs(len(wav) - len(self.audio) / speed), self.rate * 0.05)
            middle = wav[len(wav)//4:len(wav)*3//4]
            spectrum = np.abs(np.fft.rfft(middle))
            hz = np.fft.rfftfreq(len(middle), 1/rate)[spectrum.argmax()]
            self.assertLess(abs(hz - 440), 10)
            self.assertTrue(np.isfinite(wav).all())


    def test_reused_effects_reset_to_match_independent_cues(self):
        from apps.simple_tts_audio import build_effect_chain
        options = AudioOptions(bright=True, compress=True, peak_guard=True)
        chain = build_effect_chain(options)
        for scale in [1, 2, .5, 1]:
            wav = self.audio * scale
            expected, _ = process_audio(wav, self.rate, options)
            actual, _ = process_audio(wav, self.rate, options, effects_chain=chain)
            np.testing.assert_array_equal(actual, expected)

    def test_eq_compressor_and_guard_real(self):
        for bright, compress in [(True, False), (False, True), (True, True)]:
            wav, rate = process_audio(self.audio, self.rate, AudioOptions(bright=bright, compress=compress))
            self.assertEqual(len(wav), len(self.audio))
            self.assertTrue(np.isfinite(wav).all())
        loud = self.audio * 10
        wav, _ = process_audio(loud, self.rate, AudioOptions(peak_guard=True))
        self.assertLessEqual(float(np.max(np.abs(wav))), 10**(-1/20) + 1e-6)
        np.testing.assert_allclose(wav / loud.max(), loud / loud.max() * (10**(-1/20) / loud.max()), atol=1e-6)

    def test_float_export_does_not_clip_unprotected_peaks(self):
        path = save_variant(self.source, self.folder, AudioOptions(), self.audio * 10, self.rate)
        self.assertEqual(sf.info(path).subtype, "FLOAT")
        wav, _ = sf.read(path)
        self.assertGreater(np.max(wav), 1)





    def test_bypass_does_not_copy_or_load_pedalboard(self):
        with patch("apps.simple_tts_audio.pedalboard_module", side_effect=AssertionError("unneeded engine")):
            audio, _ = process_audio(self.audio, self.rate, AudioOptions())
            process_audio(self.audio, self.rate, AudioOptions(tempo="rubberband", speed=1))
        self.assertTrue(np.shares_memory(audio, self.audio))

    def test_processing_preserves_source_for_all_effects(self):
        original = self.audio.copy()
        for options in [AudioOptions(tempo="rubberband", speed=1.2),
                        AudioOptions(bright=True, compress=True, peak_guard=True)]:
            process_audio(self.audio, self.rate, options)
            np.testing.assert_array_equal(self.audio, original)
        loud = self.audio * 10
        original_loud = loud.copy()
        process_audio(loud, self.rate, AudioOptions(peak_guard=True))
        np.testing.assert_array_equal(loud, original_loud)

    def test_generated_source_cached_without_disk_read_or_new_inference(self):
        model = fixtures.FakeModel()
        service = SimpleTTS(self.folder, lambda **kwargs: model)
        self.addCleanup(service.close)
        source = service.generate("hello")
        with patch("apps.simple_tts.sf.read", side_effect=AssertionError("unneeded disk read")):
            for _ in range(2):
                service.variant(source, AudioOptions(peak_guard=True))
        self.assertEqual(len(model.calls), 1)

    def test_external_source_cache_reloads_on_file_change(self):
        service = SimpleTTS(self.folder)
        self.addCleanup(service.close)
        read = sf.read
        with patch("apps.simple_tts.sf.read", wraps=read) as reader:
            service.variant(self.source, AudioOptions())
            service.variant(self.source, AudioOptions(bright=True))
            self.assertEqual(reader.call_count, 1)
            sf.write(self.source, self.audio[:1000], self.rate, subtype="FLOAT")
            result = service.variant(self.source, AudioOptions())
            self.assertEqual(reader.call_count, 2)
        self.assertEqual(sf.info(result).frames, 1000)


class ComparisonDesktopTests(fixtures.DesktopTests):
    def test_try_settings_reuses_original_without_inference(self):
        self.app.text.insert("1.0", "hello")
        self.app.generate()
        self.wait_done()
        original = self.app.original_audio
        model = self.service.model
        calls = len(model.calls)
        self.app.tempo_name.set("Rubber Band chất lượng cao")
        self.app.speed.set("1.25")
        self.app.apply_settings()
        self.wait_done()
        self.assertEqual(self.app.original_audio, original)
        self.assertNotEqual(self.app.last_audio, original)
        self.assertEqual(len(model.calls), calls)
        self.assertEqual(len(self.app.results), 2)
        self.assertTrue(self.app.last_audio.with_suffix(".json").exists())
