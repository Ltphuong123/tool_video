"""Fake-engine tests: no weights are downloaded."""
from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from apps.simple_tts import SimpleTTS, SimpleTTSApp
from apps.simple_tts_engine import generate_speech
from apps.simple_tts_audio import AudioOptions


class FakeModel:
    def __init__(self, mode="v3turbo"):
        self.sample_rate = 48000
        self.closed = False
        self.calls = []
        self.audio = np.full(1000, 0.1, dtype=np.float32)

    def list_preset_voices(self):
        return [("Voice Turbo", "turbo_voice")]

    def infer(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return self.audio

    def close(self):
        self.closed = True


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.models = []
        self.options = []
        def factory(**kwargs):
            self.options.append(kwargs)
            model = FakeModel()
            self.models.append(model)
            return model
        self.service = SimpleTTS(self.temp.name, factory)

    def tearDown(self):
        self.service.close()
        self.temp.cleanup()

    def test_turbo_rate_and_factory_options(self):
        path = self.service.generate("  hello  ")
        audio, rate = sf.read(path)
        self.assertEqual(rate, 48000)
        self.assertEqual(len(audio), 1000)
        self.assertEqual(path.parent, Path(self.temp.name).resolve())
        self.assertEqual(self.models[0].calls[-1],
                         ("hello", {"voice": None, "apply_watermark": False, "denoise": False}))
        self.assertEqual(self.options, [{"mode": "v3turbo", "backend": "auto", "device": "auto"}])

    def test_reuses_engine_and_unique_files(self):
        voices = self.service.load()
        paths = [self.service.generate("hello", voices[0][1]) for _ in range(2)]
        self.assertEqual(len(self.models), 1)
        self.assertNotEqual(*paths)
        self.assertEqual(len(list(Path(self.temp.name).glob("*.wav"))), 2)

    def test_empty_text_does_not_load(self):
        with self.assertRaises(ValueError):
            self.service.generate("  ")
        self.assertEqual(self.models, [])

    def test_rejects_unknown_voice(self):
        self.service.load()
        with self.assertRaises(ValueError):
            self.service.generate("hello", "unknown")
        self.assertEqual(self.models[0].calls, [])

    def test_rejects_invalid_audio(self):
        self.service.load()
        for audio in [np.array([]), np.array([np.nan]), np.zeros((2, 10))]:
            self.models[0].audio = audio
            with self.assertRaises(ValueError):
                self.service.generate("hello")
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_failed_write_leaves_no_partial_file(self):
        with patch("apps.simple_tts_engine.sf.write", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.service.generate("hello")
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_busy_operation_is_rejected(self):
        with self.service.operation():
            with self.assertRaises(RuntimeError):
                self.service.load()

    def test_failed_load_can_retry(self):
        factory = self.service.factory
        self.service.factory = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("load failed"))
        with self.assertRaises(RuntimeError):
            self.service.load()
        self.service.factory = factory
        self.assertEqual(len(self.service.load()), 1)

    def test_function_returns_audio_and_applies_effects(self):
        model = FakeModel()
        model.audio *= 20
        result = generate_speech("hello", "turbo_voice", self.temp.name, model=model,
                                 options=AudioOptions(peak_guard=True))
        self.assertTrue(result.path.is_file())
        self.assertEqual(result.sample_rate, 48000)
        audio, rate = sf.read(result.path, dtype="float32")
        np.testing.assert_array_equal(audio, result.audio)
        self.assertLessEqual(float(audio.max()), 10**(-1/20) + 1e-6)
        self.assertEqual(float(model.audio.max()), 2)
        self.assertFalse(model.closed)

    def test_function_reuses_default_engine(self):
        import sys
        from types import SimpleNamespace
        from apps import simple_tts_engine as engine
        engine._close_default_model()
        model = FakeModel()
        with patch.dict(sys.modules, {"vieneu": SimpleNamespace(Vieneu=lambda **kwargs: model)}):
            try:
                generate_speech("one", output_dir=self.temp.name)
                generate_speech("two", output_dir=self.temp.name)
                self.assertEqual(len(model.calls), 2)
                self.assertIs(engine._default_model, model)
            finally:
                engine._close_default_model()
        self.assertTrue(model.closed)


class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = SimpleTTS(self.temp.name, lambda **kwargs: FakeModel())
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = SimpleTTSApp(self.root, self.service)

    def tearDown(self):
        if not self.app.closing:
            self.app.close()
        self.temp.cleanup()

    def wait_done(self):
        deadline = time.monotonic() + 3
        while self.app.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(self.app.busy)

    def test_direct_generation_and_preset_voice(self):
        self.assertIsNone(self.service.model)
        self.assertFalse(hasattr(self.app, "model_box"))
        self.app.text.insert("1.0", "hello")
        self.app.generate()
        self.wait_done()
        self.assertTrue(self.app.last_audio.is_file())
        self.assertIn("turbo_voice", self.app.voice_ids.values())
        self.app.voice_name.set("Voice Turbo")
        self.app.generate()
        self.wait_done()
        self.assertEqual(self.service.model.calls[-1][1]["voice"], "turbo_voice")
        self.assertEqual(sf.info(self.app.last_audio).samplerate, 48000)

    def test_worker_error_restores_controls(self):
        with patch.object(self.service, "load", side_effect=RuntimeError("offline")), \
             patch("apps.simple_tts.messagebox.showerror") as dialog:
            self.app.load_voices()
            self.wait_done()
            dialog.assert_called_once()
        self.assertEqual(str(self.app.voice_box["state"]), "readonly")
        self.assertEqual(str(self.app.generate_button["state"]), "normal")

    def test_close_waits_for_worker_and_releases_model(self):
        self.service.load()
        model = self.service.model
        release = threading.Event()
        self.app.submit(lambda: release.wait(2), lambda result: None)
        self.app.close()
        self.assertTrue(self.app.busy)
        self.assertFalse(model.closed)
        release.set()
        self.wait_done()
        self.assertTrue(model.closed)
