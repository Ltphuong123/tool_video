"""Desktop orchestration tests; no model downloads, GPU or audio devices."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import importlib.util
from unittest.mock import patch
import zipfile

import numpy as np
import soundfile as sf

from apps.v3turbo_tool import Cancelled, Sampling, TurboTool, parse_conversation, read_document
from apps.srt_speech import parse_srt


class FakeTurbo:
    def __init__(self, **kwargs):
        self.backend = "onnx"
        self.watermarker = None
        self.closed = False
        self.stream_closed = False
        self.calls = []
        self._default_voice = "Mai"
        self._preset_voices = {"Mai": self.entry()}

    @staticmethod
    def entry():
        return {"speaker_emb": np.ones(192, dtype=np.float32),
                "codes": np.zeros((2, 16), dtype=np.int64), "description": "Giọng mẫu"}

    def resolve_voice_name(self, name):
        return name if name in self._preset_voices else ("Mai" if name == "Alias" else None)

    def list_preset_voices(self):
        return [(name, name) for name in self._preset_voices]

    def infer(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return np.full(480, len(text) / 100, dtype=np.float32)

    def infer_batch(self, texts, **kwargs):
        return [self.infer(text, **kwargs) for text in texts]

    def infer_stream(self, text, **kwargs):
        self.calls.append((text, kwargs))
        try:
            for _ in range(3):
                yield np.full(480, 0.1, dtype=np.float32)
        finally:
            self.stream_closed = True

    def add_voice(self, name, audio, **kwargs):
        self._preset_voices[name] = self.entry()

    def remove_voice(self, name):
        del self._preset_voices[name]

    def save_voices(self, path):
        Path(path).write_text('{"presets": {}}', encoding="utf-8")

    def encode_reference(self, audio, **kwargs):
        value = self.entry()
        return value["speaker_emb"], value["codes"]

    def denoise(self, audio):
        return np.ones(441, dtype=np.float32) * 0.1, 44100

    def close(self):
        self.closed = True


class DesktopToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.tool = TurboTool(self.folder, factory=FakeTurbo)
        self.tool.load()
        self.reference = self.folder / "ref.wav"
        sf.write(self.reference, np.zeros(480, dtype=np.float32), 48000)

    def tearDown(self):
        self.temp.cleanup()

    def test_output_destination_persists_history_without_moving_saved_voices(self):
        settings = self.folder / "settings.json"
        tool = TurboTool(self.folder, factory=FakeTurbo, settings_path=settings)
        tool.load()
        original = tool.synthesize("First")
        tool.add_voice("Saved", str(self.reference))
        voice_store = tool.voices_path
        destination = self.folder / "new output"
        self.assertEqual(tool.set_output_dir(destination), destination)
        newest = tool.synthesize("Second")
        self.assertEqual(newest.parent, destination)
        self.assertTrue(original.is_file())
        self.assertEqual(tool.voices_path, voice_store)
        self.assertFalse((destination / "user_voices.json").exists())
        restarted = TurboTool(factory=FakeTurbo, settings_path=settings)
        self.assertEqual(restarted.output_dir, destination)
        self.assertEqual(restarted.voices_path, voice_store)
        restarted.load()
        self.assertIn("Saved", restarted.voice_names())
        self.assertEqual({item.path for item in restarted.list_outputs()}, {original, newest})
        # An explicit CLI destination overrides the saved choice, retaining history.
        override = TurboTool(self.folder / "CLI", settings_path=settings)
        self.assertEqual(override.output_dir, self.folder / "CLI")
        self.assertEqual({item.path for item in override.list_outputs()}, {original, newest})

    def test_history_discovers_every_export_and_excludes_input_and_voice_store(self):
        speech = self.tool.synthesize("First")
        subtitles = self.tool.synthesize_with_subtitles("One. Two.")
        archive = self.tool.batch(["One", "Two"])
        voices = self.tool.export_voices()
        embedding = self.tool.export_reference(str(self.reference))
        self.tool.add_voice("Saved", str(self.reference))
        fake = self.folder / "voices_123456789abc.wav"
        fake.write_bytes(b"Input file, not a generated export")
        rows = self.tool.list_outputs()
        paths = {item.path for item in rows}
        self.assertTrue({speech, subtitles.audio, subtitles.subtitles, archive, voices, embedding} <= paths)
        self.assertEqual(len(paths), 8)  # Six exports plus the two batch audio files.
        self.assertNotIn(self.reference, paths)
        self.assertNotIn(fake, paths)
        self.assertNotIn(self.tool.voices_path, paths)
        self.assertTrue(all(row.size == row.path.stat().st_size for row in rows))
        self.assertEqual([row.modified for row in rows], sorted((row.modified for row in rows), reverse=True))

    def test_video_export_uses_current_destination_and_preserves_source_without_a_model(self):
        source = self.folder / "original video.mp4"
        source.write_bytes(b"original video bytes")
        destination = self.folder / "video exports"
        tool = TurboTool(self.folder, factory=FakeTurbo)
        tool.set_output_dir(destination)
        messages = []
        def render(input_path, output_path, start, end, **kwargs):
            self.assertEqual(Path(input_path), source)
            self.assertEqual(Path(output_path).parent, destination)
            self.assertEqual((start, end), (2.5, 8.75))
            self.assertEqual(kwargs["speed"], 0.8)
            self.assertEqual(kwargs["ramp_seconds"], 1.0)
            self.assertTrue(kwargs["keep_audio"])
            self.assertEqual(kwargs["quality"], 20)
            self.assertEqual(kwargs["preset"], "fast")
            self.assertTrue(callable(kwargs["check_stop"]))
            kwargs["check_stop"]()
            kwargs["progress"](0.5)
            Path(output_path).write_bytes(b"rendered MP4")
            return SimpleNamespace(output_duration=21.5)
        with patch("apps.video_editor.render_speed_segment", side_effect=render) as backend, \
             patch.object(tool, "load") as load:
            result, note = tool.edit_video(source, 2.5, 8.75, speed=0.8,
                                          ramp_seconds=1.0, keep_audio=True, progress=messages.append)
        backend.assert_called_once()
        load.assert_not_called()
        self.assertIsNone(tool.tts)
        self.assertTrue(result.is_file())
        self.assertEqual(result.read_bytes(), b"rendered MP4")
        self.assertEqual(source.read_bytes(), b"original video bytes")
        self.assertTrue(note)
        self.assertTrue(messages)
        rows = tool.list_outputs()
        self.assertEqual([(row.path, row.kind) for row in rows], [(result, "Video")])

    def test_video_history_only_accepts_generated_mp4_names_and_deletion_preserves_source(self):
        accepted = self.folder / "VIDEO_123456789ABC.MP4"
        accepted.write_bytes(b"generated video")
        rejected = [self.folder / "input.mp4", self.folder / "speech_123456789abc.mp4",
                    self.folder / "video_bad.mp4", self.folder / "video_123456789abc.wav"]
        for path in rejected:
            path.write_bytes(b"input file")
        rows = self.tool.list_outputs()
        self.assertIn((accepted, "Video"), [(row.path, row.kind) for row in rows])
        self.assertTrue(set(rejected).isdisjoint(row.path for row in rows))
        with self.assertRaises(ValueError):
            self.tool.delete_outputs([accepted, rejected[0]])
        self.assertTrue(accepted.exists())
        self.assertEqual(self.tool.delete_outputs([accepted]), [accepted])
        self.assertTrue(all(path.exists() for path in rejected))

    def test_video_export_failure_removes_partial_output_and_releases_operation(self):
        source = self.folder / "input.mp4"
        source.write_bytes(b"original")
        def failing_render(_source, destination, *args, **kwargs):
            Path(destination).write_bytes(b"partial video")
            raise OSError("Encoder failed")
        with patch("apps.video_editor.render_speed_segment", side_effect=failing_render):
            with self.assertRaisesRegex(OSError, "Encoder failed"):
                self.tool.edit_video(source, 1, 5)
        self.assertEqual(list(self.folder.glob("video_*.mp4")), [])
        self.assertEqual(source.read_bytes(), b"original")
        self.assertTrue(self.tool.synthesize("Again").exists())

    def test_cancelling_video_export_removes_partial_file_and_can_retry_without_loading_model(self):
        source = self.folder / "input.mp4"
        source.write_bytes(b"original")
        tool = TurboTool(self.folder, factory=FakeTurbo)
        def cancel_render(_source, destination, *args, **kwargs):
            Path(destination).write_bytes(b"partial video")
            tool.stop()
            kwargs["check_stop"]()
        with patch("apps.video_editor.render_speed_segment", side_effect=cancel_render):
            with self.assertRaises(Cancelled):
                tool.edit_video(source, 1, 5)
        self.assertEqual(list(self.folder.glob("video_*.mp4")), [])
        def successful_render(_source, destination, *args, **kwargs):
            kwargs["check_stop"]()
            Path(destination).write_bytes(b"completed")
            return SimpleNamespace(output_duration=12.0)
        with patch("apps.video_editor.render_speed_segment", side_effect=successful_render):
            output, _ = tool.edit_video(source, 1, 5)
        self.assertTrue(output.exists())
        self.assertIsNone(tool.tts)
        self.assertEqual(source.read_bytes(), b"original")

    def test_video_export_cannot_start_during_another_operation(self):
        source = self.folder / "input.mp4"
        source.write_bytes(b"original")
        with self.tool.operation(), patch("apps.video_editor.render_speed_segment") as render:
            with self.assertRaises(RuntimeError):
                self.tool.edit_video(source, 1, 5)
            render.assert_not_called()

    def test_video_segments_export_forwards_every_segment_without_loading_tts(self):
        from apps.video_editor import SpeedSegment

        source = self.folder / "original video.mp4"
        source.write_bytes(b"original video bytes")
        destination = self.folder / "multiple segment exports"
        tool = TurboTool(self.folder, factory=FakeTurbo)
        tool.set_output_dir(destination)
        segments = [SpeedSegment(1, 3, speed=0.75, ramp_seconds=0.5),
                    SpeedSegment(5, 8, speed=2, ramp_seconds=0.25)]
        messages = []

        def render(input_path, output_path, selected, **kwargs):
            self.assertEqual(Path(input_path), source)
            self.assertEqual(Path(output_path).parent, destination)
            self.assertEqual(list(selected), segments)
            self.assertFalse(kwargs["keep_audio"])
            self.assertEqual(kwargs["quality"], 23)
            self.assertEqual(kwargs["preset"], "medium")
            kwargs["check_stop"]()
            kwargs["progress"]("Exporting the second segment")
            Path(output_path).write_bytes(b"rendered multi-segment MP4")
            return SimpleNamespace(output_duration=21.5)

        with patch("apps.video_editor.render_speed_segments", side_effect=render) as backend, \
             patch.object(tool, "load") as load:
            result, note = tool.edit_video_segments(source, segments, keep_audio=False,
                                                   quality=23, preset="medium", progress=messages.append)
        backend.assert_called_once()
        load.assert_not_called()
        self.assertIsNone(tool.tts)
        self.assertEqual(result.read_bytes(), b"rendered multi-segment MP4")
        self.assertEqual(source.read_bytes(), b"original video bytes")
        self.assertTrue(note)
        self.assertEqual(messages, ["Exporting the second segment"])
        self.assertEqual([(row.path, row.kind) for row in tool.list_outputs()], [(result, "Video")])

    def test_video_segments_cancellation_removes_output_and_allows_retry_without_tts(self):
        from apps.video_editor import SpeedSegment

        source = self.folder / "input.mp4"
        source.write_bytes(b"original")
        tool = TurboTool(self.folder, factory=FakeTurbo)
        segments = [SpeedSegment(1, 3), SpeedSegment(5, 8, speed=0.8)]

        for check_in_backend in (True, False):
            def cancel_render(_source, destination, selected, **kwargs):
                self.assertEqual(list(selected), segments)
                Path(destination).write_bytes(b"partial video")
                tool.stop()
                if check_in_backend:
                    kwargs["check_stop"]()
                return SimpleNamespace(output_duration=12.0)

            with self.subTest(check_in_backend=check_in_backend), \
                 patch("apps.video_editor.render_speed_segments", side_effect=cancel_render):
                with self.assertRaises(Cancelled):
                    tool.edit_video_segments(source, segments)
            self.assertEqual(list(self.folder.glob("video_*.mp4")), [])
            self.assertEqual(source.read_bytes(), b"original")

        def successful_render(_source, destination, selected, **kwargs):
            kwargs["check_stop"]()
            self.assertEqual(list(selected), segments)
            Path(destination).write_bytes(b"completed")
            return SimpleNamespace(output_duration=12.0)

        with patch("apps.video_editor.render_speed_segments", side_effect=successful_render):
            output, _ = tool.edit_video_segments(source, segments)
        self.assertEqual(output.read_bytes(), b"completed")
        self.assertIsNone(tool.tts)

    def test_video_segments_encoder_failure_removes_partial_output_and_releases_lock(self):
        from apps.video_editor import SpeedSegment

        source = self.folder / "input.mp4"
        source.write_bytes(b"original")

        def failing_render(_source, destination, selected, **kwargs):
            Path(destination).write_bytes(b"partial video")
            raise OSError("Encoder failed on the second segment")

        with patch("apps.video_editor.render_speed_segments", side_effect=failing_render):
            with self.assertRaisesRegex(OSError, "second segment"):
                self.tool.edit_video_segments(source, [SpeedSegment(1, 3), SpeedSegment(5, 8)])
        self.assertEqual(list(self.folder.glob("video_*.mp4")), [])
        self.assertEqual(source.read_bytes(), b"original")
        self.assertTrue(self.tool.synthesize("Again").exists())

    def test_video_segments_export_cannot_start_during_another_operation(self):
        from apps.video_editor import SpeedSegment

        source = self.folder / "input.mp4"
        source.write_bytes(b"original")
        with self.tool.operation(), patch("apps.video_editor.render_speed_segments") as render:
            with self.assertRaises(RuntimeError):
                self.tool.edit_video_segments(source, [SpeedSegment(1, 3), SpeedSegment(5, 8)])
            render.assert_not_called()
        self.assertEqual(list(self.folder.glob("video_*.mp4")), [])

    def test_video_segments_export_rejects_invalid_selection_before_rendering(self):
        from apps.video_editor import SpeedSegment

        source = self.folder / "input.mp4"
        source.write_bytes(b"original")
        selections = [[], [SpeedSegment(-1, 3)], [SpeedSegment(3, 3)],
                      [SpeedSegment(1, float("inf"))], [SpeedSegment(float("nan"), 3)],
                      [SpeedSegment(1, 3, speed=0)], [SpeedSegment(1, 3, speed=5)],
                      [SpeedSegment(1, 3, ramp_seconds=-0.1)],
                      [SpeedSegment(1, 3, ramp_seconds=1.1)],
                      [SpeedSegment(1, 4), SpeedSegment(3, 5)]]
        with patch("apps.video_editor.render_speed_segments") as render:
            for segments in selections:
                with self.subTest(segments=segments), self.assertRaises(ValueError):
                    self.tool.edit_video_segments(source, segments)
                render.assert_not_called()
            for options in (dict(quality=100), dict(preset="unknown")):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    self.tool.edit_video_segments(source, [SpeedSegment(1, 3)], **options)
                render.assert_not_called()
        self.assertEqual(list(self.folder.glob("video_*.mp4")), [])

    def test_video_segments_export_passes_an_immutable_snapshot_to_renderer(self):
        from apps.video_editor import SpeedSegment

        source = self.folder / "input.mp4"
        source.write_bytes(b"original")
        segments = [SpeedSegment(1, 3), SpeedSegment(5, 8)]
        expected = tuple(segments)

        def render(_source, destination, selected, **kwargs):
            segments.clear()
            self.assertIsInstance(selected, tuple)
            self.assertEqual(selected, expected)
            Path(destination).write_bytes(b"completed")
            return SimpleNamespace(output_duration=12.0)

        with patch("apps.video_editor.render_speed_segments", side_effect=render):
            output, _ = self.tool.edit_video_segments(source, segments)
        self.assertEqual(output.read_bytes(), b"completed")

    def test_video_export_rejects_invalid_numbers_before_rendering(self):
        source = self.folder / "input.mp4"
        source.write_bytes(b"original")
        cases = [dict(start=-1, end=5), dict(start=5, end=5),
                 dict(start=float("nan"), end=5), dict(start=1, end=float("inf")),
                 dict(speed=0), dict(speed=5), dict(speed=float("nan")),
                 dict(ramp_seconds=-1), dict(ramp_seconds=3),
                 dict(ramp_seconds=float("inf")), dict(quality=100), dict(preset="unknown")]
        with patch("apps.video_editor.render_speed_segment") as render:
            for case in cases:
                with self.subTest(case=case):
                    options = dict(start=1, end=5)
                    options.update(case)
                    with self.assertRaises(ValueError):
                        self.tool.edit_video(source, **options)
                    render.assert_not_called()
        self.assertEqual(list(self.folder.glob("video_*.mp4")), [])

    def test_delete_history_validates_entire_selection_and_removes_paired_subtitles(self):
        generated = self.tool.synthesize_with_subtitles("One. Two.")
        self.tool.add_voice("Saved", str(self.reference))
        with self.assertRaises(ValueError):
            self.tool.delete_outputs([generated.audio, self.reference])
        self.assertTrue(generated.audio.exists())
        self.assertTrue(generated.subtitles.exists())
        with self.assertRaises(ValueError):
            self.tool.delete_outputs([self.tool.voices_path])
        removed = self.tool.delete_outputs([generated.audio, generated.audio])
        self.assertEqual(removed, [generated.audio, generated.subtitles])
        self.assertFalse(generated.audio.exists())
        self.assertFalse(generated.subtitles.exists())
        self.assertTrue(self.reference.exists())
        self.assertTrue(self.tool.voices_path.exists())

    def test_delete_history_preserves_subtitle_when_requested_and_old_folders_are_known(self):
        generated = self.tool.synthesize_with_subtitles("One.")
        self.tool.set_output_dir(self.folder / "next")
        self.assertEqual(self.tool.delete_outputs([generated.audio], include_subtitles=False), [generated.audio])
        self.assertTrue(generated.subtitles.exists())
        self.assertEqual(self.tool.delete_outputs([generated.subtitles]), [generated.subtitles])

    def test_output_settings_and_delete_cannot_run_during_generation(self):
        speech = self.tool.synthesize("First")
        with self.tool.operation():
            with self.assertRaises(RuntimeError):
                self.tool.set_output_dir(self.folder / "next")
            with self.assertRaises(RuntimeError):
                self.tool.delete_outputs([speech])
        self.assertEqual(self.tool.output_dir, self.folder)
        self.assertTrue(speech.is_file())
        self.assertFalse((self.folder / "next").exists())

    def test_invalid_output_settings_and_failed_save_do_not_break_existing_destination(self):
        settings = self.folder / "settings.json"
        settings.write_text("{invalid", encoding="utf-8")
        tool = TurboTool(self.folder, settings_path=settings)
        self.assertIsNotNone(tool.settings_error)
        self.assertEqual(tool.output_dir, self.folder)
        with self.assertRaises(ValueError):
            tool.set_output_dir("")
        with self.assertRaises(ValueError):
            tool.set_output_dir(self.reference)
        with patch.object(tool, "_write_settings", side_effect=PermissionError("Read-only settings")):
            with self.assertRaises(PermissionError):
                tool.set_output_dir(self.folder / "next")
        self.assertEqual(tool.output_dir, self.folder)
        self.assertEqual(tool.output_dirs, (self.folder,))
        tool.set_output_dir(self.folder / "next")
        self.assertIsNone(tool.settings_error)
        self.assertEqual(json.loads(settings.read_text(encoding="utf-8"))["output_dir"], str(self.folder / "next"))
        self.assertEqual(list(self.folder.glob("tmp*.tmp")), [])

    def test_history_uses_case_insensitive_names_and_excludes_symlinks(self):
        generated = self.folder / "SPEECH_123456789ABC.WAV"
        generated.write_bytes(b"output")
        self.assertEqual([item.path for item in self.tool.list_outputs()], [generated])
        symlink = self.folder / "speech_abcdef123456.wav"
        try:
            symlink.symlink_to(generated)
        except OSError:
            self.skipTest("This Windows account cannot create symbolic links")
        self.assertEqual([item.path for item in self.tool.list_outputs()], [generated])
        with self.assertRaises(ValueError):
            self.tool.delete_outputs([symlink])
        self.assertTrue(generated.exists())

    def test_delete_rejects_generated_looking_file_outside_visited_folders(self):
        outside = self.folder / "unvisited"
        outside.mkdir()
        path = outside / "speech_123456789abc.wav"
        path.write_bytes(b"external")
        with self.assertRaises(ValueError):
            self.tool.delete_outputs([path])
        self.assertTrue(path.exists())

    def test_speech_forwards_sampling_and_clone_and_writes_audio(self):
        path = self.tool.synthesize("Xin chào", "Mai", str(self.reference), Sampling(top_k=17), "flac")
        info = sf.info(path)
        self.assertEqual((info.samplerate, info.frames), (48000, 480))
        kw = self.tool.tts.calls[-1][1]
        self.assertEqual(kw["ref_audio"], str(self.reference))
        self.assertNotIn("voice", kw)
        self.assertEqual(kw["top_k"], 17)

    def test_invalid_sampling_is_rejected_before_model_call(self):
        for values in ({"temperature": float("nan")}, {"top_p": 0}, {"top_k": 2.5},
                       {"top_k": 2.0}, {"batch_size": 0}, {"speed": 0},
                       {"speed": 2.1}, {"speed": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.tool.synthesize("Text", sampling=Sampling(**values))
        self.assertEqual(self.tool.tts.calls, [])

    def test_watermark_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "Watermark"):
            self.tool.synthesize("Text", sampling=Sampling(apply_watermark=True))

    def test_stream_saves_every_chunk_and_closes_iterator(self):
        chunks = []
        results = list(self.tool.stream("Text", on_audio=lambda chunk, rate: chunks.append((len(chunk), rate))))
        path = results[-1][0]
        self.assertEqual(sf.info(path).frames, 1440)
        self.assertEqual(chunks, [(480, 48000)] * 3)
        self.assertTrue(self.tool.tts.stream_closed)
        self.assertNotIn("batch_size", self.tool.tts.calls[0][1])

    def test_stop_stream_removes_partial_file_and_releases_lock(self):
        stream = self.tool.stream("Text")
        next(stream)
        self.tool.stop()
        with self.assertRaises(Cancelled):
            next(stream)
        self.assertEqual(list(self.folder.glob("stream_*.wav")), [])
        self.assertTrue(self.tool.tts.stream_closed)
        self.assertTrue(self.tool.synthesize("Again").exists())

    def test_closing_stream_cancels_and_releases_lock(self):
        stream = self.tool.stream("Text")
        next(stream)
        stream.close()
        self.assertTrue(self.tool.tts.stream_closed)
        self.assertEqual(list(self.folder.glob("stream_*.wav")), [])
        self.assertTrue(self.tool.synthesize("Again").exists())

    def test_batch_archive_preserves_order_and_manifest(self):
        texts = ["A", "BB", "CCC"]
        archive = self.tool.batch(texts, sampling=Sampling(batch_size=2))
        with zipfile.ZipFile(archive) as file:
            self.assertEqual(file.namelist(), ["0001.wav", "0002.wav", "0003.wav", "manifest.json"])
            data = json.loads(file.read("manifest.json"))
        self.assertEqual([entry["text"] for entry in data], texts)
        self.assertEqual([text for text, kw in self.tool.tts.calls], texts)

    def test_srt_preserves_timeline_and_sampling(self):
        path = self.folder / "test.srt"
        path.write_text("1\n00:00:01,000 --> 00:00:02,000\nXin chào\n", encoding="utf-8")
        audio, note = self.tool.srt(path, "Mai", Sampling(top_k=19))
        wav, rate = sf.read(audio)
        self.assertEqual(rate, 48000)
        self.assertTrue(np.all(wav[:48000] == 0))
        self.assertGreater(float(wav[48000]), 0)
        self.assertEqual(self.tool.tts.calls[0][1]["top_k"], 19)
        self.assertIn("0 câu", note)

    def test_srt_auto_fit_accelerates_only_long_cues_and_keeps_the_timeline(self):
        path = self.folder / "fit.srt"
        path.write_text("1\n00:00:01,000 --> 00:00:03,000\nOne\n\n"
                        "2\n00:00:04,000 --> 00:00:05,000\nTwo\n", encoding="utf-8")
        long_clip = np.full(144000, 0.1, dtype=np.float32)
        short_clip = np.full(24000, 0.2, dtype=np.float32)
        messages = []
        with patch.object(self.tool.tts, "infer_batch", return_value=[long_clip, short_clip]) as infer:
            audio, note = self.tool.srt(path, "Mai", Sampling(top_k=19), fit_to_timing=True, progress=messages.append)
        track, rate = sf.read(audio)
        self.assertEqual(rate, 48000)
        self.assertEqual(len(track), 5 * rate)
        self.assertTrue(np.all(track[:rate] == 0))
        np.testing.assert_allclose(track[rate + rate // 2:3 * rate - rate // 2], 0.1, atol=1e-4)
        self.assertGreater(abs(track[rate]), .001)
        self.assertEqual(track[3 * rate - 1], 0)
        self.assertTrue(np.all(track[3 * rate:4 * rate] == 0))
        np.testing.assert_allclose(track[4 * rate:4 * rate + 23760], short_clip[:23760], atol=1 / 32768)
        self.assertEqual(track[4 * rate + 23999], 0)
        self.assertTrue(np.all(track[4 * rate + 24000:] == 0))
        self.assertIn("1.50x", note)
        self.assertTrue(any("1.50x" in message for message in messages))
        self.assertEqual(infer.call_args.kwargs["top_k"], 19)
        self.assertNotIn("fit_to_timing", infer.call_args.kwargs)

    def test_srt_voice_starts_at_exact_cue_time_after_removing_model_silence(self):
        path = self.folder / "padded_fit.srt"
        path.write_text("1\n00:00:01,123 --> 00:00:02,123\nOne\n\n"
                        "2\n00:00:03,456 --> 00:00:04,456\nTwo\n", encoding="utf-8")
        clips = [np.concatenate((np.zeros(14400), np.full(72000, 0.1), np.zeros(19200))),
                 np.concatenate((np.zeros(9600), np.full(12000, 0.2), np.zeros(9600)))]
        with patch.object(self.tool.tts, "infer_batch", return_value=clips):
            audio, note = self.tool.srt(path, fit_to_timing=True)
        track, rate = sf.read(audio)
        first_start, first_end = 1123 * rate // 1000, 2123 * rate // 1000
        second_start = 3456 * rate // 1000
        self.assertEqual(np.flatnonzero(np.abs(track) > 0.001)[0], first_start)
        self.assertTrue(np.all(track[:first_start] == 0))
        np.testing.assert_allclose(track[first_start + 12000:first_end - 12000], 0.1, atol=1e-4)
        self.assertEqual(track[first_end - 1], 0)
        self.assertTrue(np.all(track[first_end:second_start] == 0))
        self.assertGreater(abs(track[second_start]), 0.001)
        np.testing.assert_allclose(track[second_start:second_start + 11760], 0.2, atol=1 / 32768)
        self.assertEqual(track[second_start + 11999], 0)
        self.assertEqual(len(track), 4456 * rate // 1000)
        self.assertIn("1.50x", note)

    def test_default_srt_preserves_long_initial_and_intercue_silence_without_blank_lines(self):
        path = self.folder / "long_gaps.srt"
        path.write_text("1\n00:00:05,123 --> 00:00:06,123\nOne\n"
                        "2\n00:01:30,456 --> 00:01:32,456\nTwo\n", encoding="utf-8")
        messages = []
        with patch.object(self.tool.tts, "infer_batch", return_value=[np.full(72000, 0.1), np.full(24000, 0.2)]) as infer:
            audio, _ = self.tool.srt(path, progress=messages.append)
        track, rate = sf.read(audio)
        first_start, first_end = 5123 * rate // 1000, 6123 * rate // 1000
        second_start, second_end = 90456 * rate // 1000, 92456 * rate // 1000
        self.assertEqual(infer.call_args.args[0], ["One", "Two"])
        self.assertEqual(len(track), second_end)
        self.assertTrue(np.all(track[:first_start] == 0))
        np.testing.assert_allclose(track[first_start + 12000:first_end - 12000], 0.1, atol=1e-4)
        self.assertGreater(abs(track[first_start]), .001)
        self.assertEqual(track[first_end - 1], 0)
        self.assertTrue(np.all(track[first_end:second_start] == 0))
        np.testing.assert_allclose(track[second_start:second_start + 23760], 0.2, atol=1 / 32768)
        self.assertEqual(track[second_start + 23999], 0)
        self.assertTrue(np.all(track[second_start + 24000:] == 0))
        self.assertTrue(any("92.456s" in message for message in messages))

    def test_srt_minimum_speed_is_independent_of_global_speed(self):
        path = self.folder / "minimum_fit.srt"
        path.write_text("1\n00:00:01,000 --> 00:00:02,000\nOne\n", encoding="utf-8")
        with patch.object(self.tool.tts, "infer_batch", return_value=[np.full(24000, 0.1)]) as infer:
            audio, note = self.tool.srt(path, sampling=Sampling(speed=2), min_speed=0.8, fit_to_timing=True)
        track, rate = sf.read(audio)
        np.testing.assert_allclose(track[55500:70500], 0.1, atol=1e-4)
        self.assertTrue(np.any(track[76000:78000] != 0))
        self.assertTrue(np.all(track[78000:] == 0))
        self.assertIn("0.80x", note)
        self.assertNotIn("speed", infer.call_args.kwargs)
        self.assertNotIn("min_speed", infer.call_args.kwargs)

    def test_srt_rejects_invalid_minimum_before_generation(self):
        path = self.folder / "bad_minimum.srt"
        path.write_text("1\n00:00:00,000 --> 00:00:01,000\nOne\n", encoding="utf-8")
        with patch.object(self.tool.tts, "infer_batch") as infer:
            for minimum in (0, float("nan"), 2.1):
                with self.assertRaises(ValueError):
                    self.tool.srt(path, min_speed=minimum, fit_to_timing=True)
            infer.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("pedalboard"), "optional pedalboard not installed")
    def test_srt_rubberband_keeps_timing_gaps_and_fades_tails(self):
        path = self.folder / "quality_fit.srt"
        path.write_text("1\n00:00:01,000 --> 00:00:01,500\nOne\n\n"
                        "2\n00:00:05,000 --> 00:00:06,000\nTwo\n", encoding="utf-8")
        messages = []
        with patch.object(self.tool.tts, "infer_batch", return_value=[self.tone(), self.tone()]) as infer:
            audio, note = self.tool.srt(path, progress=messages.append)
        track, rate = sf.read(audio)
        self.assertEqual(len(track), 6 * rate)
        self.assertTrue(np.all(track[:rate] == 0))
        self.assertTrue(np.all(track[rate + rate // 2:5 * rate] == 0))
        self.assertGreater(float(np.abs(track[rate:rate + rate // 2]).max()), .05)
        self.assertGreater(float(np.abs(track[5 * rate:]).max()), .05)
        self.assertEqual(track[rate + rate // 2 - 1], 0)
        self.assertNotIn("tempo_method", infer.call_args.kwargs)
        self.assertNotIn("speed", infer.call_args.kwargs)
        self.assertTrue(any("rubberband" in message for message in messages))

    def test_missing_rubberband_is_rejected_before_generating_srt_audio(self):
        path = self.folder / "missing_quality.srt"
        path.write_text("1\n00:00:00,000 --> 00:00:01,000\nOne\n", encoding="utf-8")
        with patch.dict("sys.modules", {"pedalboard": None}), patch.object(self.tool.tts, "infer_batch") as infer:
            with self.assertRaisesRegex(ValueError, "Pedalboard"):
                self.tool.srt(path)
            infer.assert_not_called()

    def test_srt_auto_fit_supports_more_than_two_x_and_multiple_batches(self):
        path = self.folder / "fast_fit.srt"
        path.write_text("1\n00:00:00,000 --> 00:00:00,200\nOne\n\n"
                        "2\n00:00:00,200 --> 00:00:00,400\nTwo\n", encoding="utf-8")
        with patch.object(self.tool.tts, "infer_batch", return_value=[self.tone()]) as infer:
            audio, note = self.tool.srt(path, sampling=Sampling(batch_size=1), fit_to_timing=True)
        self.assertEqual(sf.info(audio).frames, 19200)
        self.assertEqual(infer.call_count, 2)
        self.assertIn("5.00x", note)

    def test_srt_auto_fit_applies_manual_speed_before_measuring_the_window(self):
        path = self.folder / "manual_fit.srt"
        path.write_text("1\n00:00:00,000 --> 00:00:01,000\nOne\n", encoding="utf-8")
        with patch.object(self.tool.tts, "infer_batch", return_value=[self.tone()]):
            audio, note = self.tool.srt(path, sampling=Sampling(speed=2), fit_to_timing=True)
        track, rate = sf.read(audio)
        self.assertEqual(len(track), rate)
        self.assertTrue(np.all(track[rate // 2:] == 0))
        self.assertIn("0 câu", note)

    def test_srt_auto_fit_watermarks_once_after_all_speed_adjustments(self):
        path = self.folder / "marked_fit.srt"
        path.write_text("1\n00:00:00,000 --> 00:00:00,500\nOne\n", encoding="utf-8")
        self.tool.tts.watermarker = object()
        with patch.object(self.tool.tts, "infer_batch", return_value=[self.tone()]) as infer, \
             patch.object(self.tool.tts, "_apply_watermark", side_effect=lambda audio: audio, create=True) as mark:
            audio, _ = self.tool.srt(path, sampling=Sampling(speed=0.8, apply_watermark=True), fit_to_timing=True)
        self.assertFalse(infer.call_args.kwargs["apply_watermark"])
        self.assertEqual(mark.call_count, 1)
        self.assertEqual(len(mark.call_args.args[0]), 24000)
        self.assertEqual(sf.info(audio).frames, 24000)

    def test_srt_auto_fit_rejects_invalid_or_equal_start_timestamps_before_inference(self):
        path = self.folder / "invalid_fit.srt"
        for text in ("1\n00:00:01,000 --> 00:00:01,000\nOne\n",
                     "1\n00:00:00,000 --> 00:00:01,000\nOne\n\n"
                     "2\n00:00:00,000 --> 00:00:02,000\nTwo\n"):
            path.write_text(text, encoding="utf-8")
            with patch.object(self.tool.tts, "infer_batch") as infer:
                with self.assertRaises(ValueError):
                    self.tool.srt(path, fit_to_timing=True)
                infer.assert_not_called()
        with self.assertRaises(ValueError):
            self.tool.srt(path, keep_timing=False, fit_to_timing=True)

    def test_srt_auto_fit_handles_overlapping_cues_without_shifting(self):
        path = self.folder / "overlap_fit.srt"
        path.write_text("1\n00:00:00,000 --> 00:00:02,000\nOne\n\n"
                        "2\n00:00:01,000 --> 00:00:03,000\nTwo\n", encoding="utf-8")
        with patch.object(self.tool.tts, "infer_batch", return_value=[np.full(96000, 0.1), np.full(48000, 0.2)]):
            audio, note = self.tool.srt(path, fit_to_timing=True)
        track, rate = sf.read(audio)
        np.testing.assert_allclose(track[rate // 4:3 * rate // 4], 0.1, atol=1e-4)
        np.testing.assert_allclose(track[rate:2 * rate - 240], 0.2, atol=1 / 32768)
        self.assertGreater(abs(track[0]), .001)
        self.assertGreater(abs(track[rate]), .001)
        self.assertEqual(track[rate - 1], 0)
        self.assertEqual(track[2 * rate - 1], 0)
        self.assertTrue(np.all(track[2 * rate:] == 0))
        self.assertEqual(len(track), 3 * rate)

    def test_cancelling_srt_auto_fit_does_not_export_partial_audio(self):
        path = self.folder / "cancel_fit.srt"
        path.write_text("1\n00:00:00,000 --> 00:00:00,500\nOne\n", encoding="utf-8")
        with patch.object(self.tool.tts, "infer_batch", return_value=[self.tone()]):
            with self.assertRaises(Cancelled):
                self.tool.srt(path, fit_to_timing=True, progress=lambda _: self.tool.stop())
        self.assertEqual(list(self.folder.glob("srt_*.wav")), [])
        self.assertTrue(self.tool.synthesize("Again").exists())

    def test_conversation_validates_all_voices_before_generation(self):
        self.tool.tts._preset_voices["Bình"] = FakeTurbo.entry()
        self.tool.conversation("A: One\nB: Two: three", {"A": "Mai", "B": "Bình"})
        self.assertEqual([kw["voice"] for text, kw in self.tool.tts.calls], ["Mai", "Bình"])
        self.tool.tts.calls.clear()
        with self.assertRaises(ValueError):
            self.tool.conversation("A: One\nB: Two", {"A": "Mai", "B": "Missing"})
        self.assertEqual(self.tool.tts.calls, [])

    def test_conversation_batches_consecutive_same_voice_turns_with_size_limit(self):
        self.tool.tts._preset_voices["Other"] = FakeTurbo.entry()
        script = "A: One\nA: Two\nA: Three\nB: Four\nB: Five\nA: Six"
        batches, messages = [], []
        clips = {text: np.full(480, (index + 1) / 10, dtype=np.float32)
                 for index, text in enumerate(("One", "Two", "Three", "Four", "Five", "Six"))}
        def generate(texts, **kwargs):
            batches.append((list(texts), kwargs))
            return [clips[text] for text in texts]
        with patch.object(self.tool.tts, "infer_batch", side_effect=generate), \
             patch.object(self.tool.tts, "infer") as single:
            path = self.tool.conversation(script, {"A": "Mai", "B": "Other"},
                                          Sampling(batch_size=2, top_k=19), gap=.01,
                                          progress=messages.append)
        single.assert_not_called()
        self.assertEqual([(texts, kw["voice"]) for texts, kw in batches],
                         [(["One", "Two"], "Mai"), (["Three"], "Mai"),
                          (["Four", "Five"], "Other"), (["Six"], "Mai")])
        self.assertTrue(all(kw["top_k"] == 19 for texts, kw in batches))
        self.assertEqual(len(messages), 6)
        self.assertTrue(all(f"{index}/6" in message for index, message in enumerate(messages, 1)))
        track, rate = sf.read(path)
        expected = np.concatenate([part for index, clip in enumerate(clips.values())
                                   for part in ((np.zeros(480), clip) if index else (clip,))])
        self.assertEqual(rate, 48000)
        np.testing.assert_allclose(track, expected, atol=1 / 32768)

    def test_conversation_rejects_batch_result_count_before_export(self):
        with patch.object(self.tool.tts, "infer_batch", return_value=[self.tone()]):
            with self.assertRaises(RuntimeError):
                self.tool.conversation("A: One\nA: Two", {"A": "Mai"})
        self.assertEqual(list(self.folder.glob("conversation_*.wav")), [])

    def test_conversation_cancellation_stops_later_batches_and_exports_nothing(self):
        with patch.object(self.tool.tts, "infer_batch", side_effect=lambda texts, **kw: [self.tone() for _ in texts]) as infer:
            with self.assertRaises(Cancelled):
                self.tool.conversation("A: One\nA: Two\nA: Three", {"A": "Mai"},
                                       Sampling(batch_size=2), progress=lambda _: self.tool.stop())
        self.assertEqual(infer.call_count, 1)
        self.assertEqual(list(self.folder.glob("conversation_*.wav")), [])
        self.assertTrue(self.tool.synthesize("Again").exists())

    def test_voices_survive_reload_and_builtin_alias_is_protected(self):
        self.tool.add_voice("My voice", str(self.reference))
        self.assertTrue(self.tool.voices_path.exists())
        self.tool.load()
        self.assertIn("My voice", self.tool.voice_names())
        with self.assertRaises(ValueError):
            self.tool.add_voice("Alias", str(self.reference))
        with self.assertRaises(ValueError):
            self.tool.delete_voice("Mai")
        self.tool.delete_voice("My voice")
        self.tool.load()
        self.assertNotIn("My voice", self.tool.voice_names())

    def test_reference_export_contains_embedding_and_codes(self):
        path = self.tool.export_reference(str(self.reference))
        with np.load(path, allow_pickle=False) as arrays:
            self.assertEqual(arrays["speaker_emb"].shape, (192,))
            self.assertEqual(arrays["ref_codes"].shape, (2, 16))

    def test_denoise_preserves_its_actual_sample_rate(self):
        path = self.tool.denoise(str(self.reference))
        self.assertEqual(sf.info(path).samplerate, 44100)

    def test_import_is_validated_before_any_voice_is_mutated(self):
        path = self.folder / "voices.json"
        valid = {"speaker_emb": [1] * 192, "codes": [[0] * 16]}
        path.write_text(json.dumps({"presets": {"New": valid, "Bad": {"speaker_emb": [1]}}}))
        with self.assertRaises(ValueError):
            self.tool.import_voices(path)
        self.assertNotIn("New", self.tool.voice_names())
        path.write_text(json.dumps({"presets": {"New": valid, "Mai": valid}}))
        note = self.tool.import_voices(path)
        self.assertIn("1 giọng riêng", note)
        self.assertIn("New", self.tool.voice_names())
        self.tool.load()
        self.assertIn("New", self.tool.voice_names())

    def test_busy_model_cannot_be_unloaded_during_stream(self):
        stream = self.tool.stream("Text")
        next(stream)
        with self.assertRaises(RuntimeError):
            self.tool.unload()
        stream.close()
        model = self.tool.tts
        self.tool.unload()
        self.assertTrue(model.closed)

    def test_empty_document_and_malformed_script_are_rejected(self):
        path = self.folder / "empty.txt"
        path.write_text(" ", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_document(path)
        with self.assertRaises(ValueError):
            parse_conversation("Missing colon", {"A": "Mai"})

    @staticmethod
    def tone():
        return (0.2 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)).astype(np.float32)

    def test_speed_changes_duration_while_preserving_pitch_and_file_rate(self):
        tone = self.tone()
        with patch.object(self.tool.tts, "infer", return_value=tone) as infer:
            for speed in (0.8, 1.0, 1.25, 2.0):
                with self.subTest(speed=speed):
                    path = self.tool.synthesize("Test", sampling=Sampling(speed=speed))
                    wav, rate = sf.read(path)
                    self.assertEqual(rate, 48000)
                    self.assertEqual(len(wav), round(len(tone) / speed))
                    middle = wav[len(wav) // 4:3 * len(wav) // 4]
                    spectrum = np.abs(np.fft.rfft(middle * np.hanning(len(middle))))
                    frequency = np.fft.rfftfreq(len(middle), d=1 / rate)[spectrum.argmax()]
                    self.assertAlmostEqual(frequency, 440, delta=3)
                    self.assertNotIn("speed", infer.call_args.kwargs)
                    self.assertNotIn("speed_method", infer.call_args.kwargs)


    @unittest.skipUnless(importlib.util.find_spec("pedalboard"), "optional pedalboard not installed")
    def test_rubberband_is_applied_to_speech_batch_conversation_and_sentence_subtitles(self):
        tone = self.tone()
        sampling = Sampling(speed=1.25)
        with patch.object(self.tool.tts, "infer", return_value=tone) as infer, \
             patch.object(self.tool.tts, "infer_batch", return_value=[tone, tone]):
            speech = self.tool.synthesize("One", sampling=sampling)
            archive = self.tool.batch(["One", "Two"], sampling=sampling)
            conversation = self.tool.conversation("A: One\nA: Two", {"A": "Mai"}, sampling=sampling, gap=0.4)
            subtitled = self.tool.synthesize_with_subtitles("One. Two.", sampling=sampling)
        self.assertEqual(sf.info(speech).frames, 38400)
        self.assertEqual(sf.info(conversation).frames, 2 * 38400 + 19200)
        with zipfile.ZipFile(archive) as file:
            import io
            self.assertEqual(sf.info(io.BytesIO(file.read("0001.wav"))).frames, 38400)
            self.assertEqual(json.loads(file.read("manifest.json"))[0]["speed_method"], "rubberband")
        cues = parse_srt(subtitled.subtitles.read_text(encoding="utf-8"))
        self.assertEqual(len(cues), 2)
        self.assertLessEqual(cues[0].end_ms - cues[0].start_ms, 800)
        self.assertGreaterEqual(cues[1].start_ms, cues[0].end_ms)
        self.assertNotIn("speed_method", infer.call_args.kwargs)
        self.assertNotIn("speed", infer.call_args.kwargs)

    @unittest.skipUnless(importlib.util.find_spec("pedalboard"), "optional pedalboard not installed")
    def test_rubberband_stream_waits_for_generation_and_saves_exactly_what_is_played(self):
        tone = self.tone()
        closed = []
        played = []
        def chunks(*args, **kwargs):
            try:
                yield from np.array_split(tone, 3)
            finally:
                closed.append(True)
        def playback(audio, rate):
            self.assertTrue(closed)
            self.assertEqual(rate, 48000)
            played.append(audio.copy())
        with patch.object(self.tool.tts, "infer_stream", side_effect=chunks):
            results = list(self.tool.stream("One", sampling=Sampling(speed=1.25),
                                           on_audio=playback))
        exported, rate = sf.read(results[-1][0])
        self.assertEqual(len(exported), 38400)
        np.testing.assert_allclose(exported, np.concatenate(played), atol=1 / 32768)

    def test_speed_processing_receives_unquantized_stream_source(self):
        source = self.tone()
        native_sound_file = sf.SoundFile
        def stretch(audio, speed, **kwargs):
            np.testing.assert_array_equal(audio, source)
            self.assertEqual(list(self.folder.glob("stream_*.wav")), [])
            files.assert_not_called()
            return audio[:38400]
        with patch.object(self.tool.tts, "infer_stream", return_value=iter([source])), \
             patch("apps.v3turbo_tool.sf.SoundFile", wraps=native_sound_file) as files, \
             patch("apps.v3turbo_tool.stretch_rubberband", side_effect=stretch):
            list(self.tool.stream("One", sampling=Sampling(speed=1.25)))
        self.assertEqual(files.call_count, 1)

    def test_buffered_stream_captures_mutable_chunks_before_generator_reuses_them(self):
        original = np.concatenate([np.full(480, value, dtype=np.float32) for value in (.1, .2, .3)])
        def chunks(*args, **kwargs):
            buffer = np.empty(480, dtype=np.float32)
            try:
                for value in (.1, .2, .3):
                    buffer.fill(value)
                    yield buffer
                buffer.fill(.9)
            finally:
                self.tool.tts.stream_closed = True
        def stretch(audio, speed, **kwargs):
            self.assertTrue(self.tool.tts.stream_closed)
            np.testing.assert_array_equal(audio, original)
            return audio[:1152]
        with patch.object(self.tool.tts, "infer_stream", side_effect=chunks), \
             patch("apps.v3turbo_tool.stretch_rubberband", side_effect=stretch):
            results = list(self.tool.stream("One", sampling=Sampling(speed=1.25)))
        self.assertEqual(sf.info(results[-1][0]).frames, 1152)

    def test_buffered_stream_reads_spilled_float_source_and_closes_spool(self):
        source = self.tone()
        spools = []
        native_spool = tempfile.SpooledTemporaryFile
        def make_spool(*args, **kwargs):
            kwargs["max_size"] = 1024
            value = native_spool(*args, **kwargs)
            spools.append(value)
            return value
        def stretch(audio, speed, **kwargs):
            self.assertTrue(spools[0]._rolled)
            np.testing.assert_array_equal(audio, source)
            return audio[:38400]
        with patch.object(self.tool.tts, "infer_stream", return_value=iter(np.array_split(source, 3))), \
             patch("apps.v3turbo_tool.tempfile.SpooledTemporaryFile", side_effect=make_spool), \
             patch("apps.v3turbo_tool.stretch_rubberband", side_effect=stretch):
            results = list(self.tool.stream("One", sampling=Sampling(speed=1.25)))
        self.assertEqual(sf.info(results[-1][0]).frames, 38400)
        self.assertTrue(spools[0].closed)

    def test_cancelling_buffered_stream_closes_iterator_spool_and_releases_lock(self):
        spools = []
        native_spool = tempfile.SpooledTemporaryFile
        def make_spool(*args, **kwargs):
            value = native_spool(*args, **kwargs)
            spools.append(value)
            return value
        with patch("apps.v3turbo_tool.tempfile.SpooledTemporaryFile", side_effect=make_spool):
            stream = self.tool.stream("One", sampling=Sampling(speed=1.25))
            next(stream)
            self.tool.stop()
            with self.assertRaises(Cancelled):
                next(stream)
        self.assertTrue(self.tool.tts.stream_closed)
        self.assertTrue(spools[0].closed)
        self.assertEqual(list(self.folder.glob("stream_*.wav")), [])
        self.assertTrue(self.tool.synthesize("Again").exists())

    def test_missing_rubberband_is_rejected_before_speech_or_stream_inference(self):
        with patch.dict("sys.modules", {"pedalboard": None}), \
             patch.object(self.tool.tts, "infer") as infer, \
             patch.object(self.tool.tts, "infer_stream") as stream:
            sampling = Sampling(speed=1.2)
            with self.assertRaisesRegex(ValueError, "Pedalboard"):
                self.tool.synthesize("One", sampling=sampling)
            with self.assertRaisesRegex(ValueError, "Pedalboard"):
                list(self.tool.stream("One", sampling=sampling))
            infer.assert_not_called()
            stream.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("pedalboard"), "optional pedalboard not installed")
    def test_rubberband_applies_watermark_after_final_speed_adjustment(self):
        self.tool.tts.watermarker = object()
        with patch.object(self.tool.tts, "infer", return_value=self.tone()) as infer, \
             patch.object(self.tool.tts, "_apply_watermark", side_effect=lambda audio: audio, create=True) as mark:
            audio = self.tool.synthesize("One", sampling=Sampling(speed=1.25, apply_watermark=True))
        self.assertFalse(infer.call_args.kwargs["apply_watermark"])
        self.assertEqual(mark.call_count, 1)
        self.assertEqual(len(mark.call_args.args[0]), 38400)
        self.assertEqual(sf.info(audio).frames, 38400)

    def test_nonfinite_stream_chunk_is_rejected_and_partial_file_removed(self):
        with patch.object(self.tool.tts, "infer_stream", return_value=iter([np.array([np.nan])])):
            with self.assertRaises(ValueError):
                list(self.tool.stream("One", sampling=Sampling(speed=1.2)))
        self.assertEqual(list(self.folder.glob("stream_*.wav")), [])

    def test_speed_is_applied_to_batch_and_conversation_without_scaling_turn_gap(self):
        tone = self.tone()
        with patch.object(self.tool.tts, "infer_batch", return_value=[tone, tone]):
            archive = self.tool.batch(["One", "Two"], sampling=Sampling(speed=2))
        with zipfile.ZipFile(archive) as file:
            import io
            wav, rate = sf.read(io.BytesIO(file.read("0001.wav")))
            manifest = json.loads(file.read("manifest.json"))
        self.assertEqual(len(wav), 24000)
        self.assertEqual(manifest[0]["speed"], 2)
        with patch.object(self.tool.tts, "infer_batch", return_value=[tone, tone]):
            path = self.tool.conversation("A: One\nA: Two", {"A": "Mai"}, Sampling(speed=2), gap=0.25)
        self.assertEqual(sf.info(path).frames, 24000 * 2 + 12000)

    def test_adjusted_stream_waits_for_generation_and_saves_what_it_plays(self):
        tone = self.tone()
        def source(*args, **kwargs):
            try:
                for chunk in np.array_split(tone, 3):
                    yield chunk
            finally:
                self.tool.tts.stream_closed = True
        played = []
        def playback(chunk, rate):
            self.assertTrue(self.tool.tts.stream_closed)
            self.assertEqual(rate, 48000)
            played.append(chunk.copy())
        with patch.object(self.tool.tts, "infer_stream", side_effect=source):
            results = list(self.tool.stream("Text", sampling=Sampling(speed=1.25), on_audio=playback))
        wav, rate = sf.read(results[-1][0])
        self.assertEqual(len(wav), 38400)
        np.testing.assert_allclose(wav, np.concatenate(played), atol=1 / 32768)
        self.assertTrue(any("Đang chỉnh tốc độ" in message for path, message in results))

    def test_speed_keeps_srt_start_and_applies_watermark_after_stretch(self):
        path = self.folder / "speed.srt"
        path.write_text("1\n00:00:01,000 --> 00:00:02,000\nXin chào\n", encoding="utf-8")
        tone = self.tone()
        self.tool.tts.watermarker = object()
        with patch.object(self.tool.tts, "infer_batch", return_value=[tone]) as infer, \
             patch.object(self.tool.tts, "_apply_watermark", side_effect=lambda audio: audio, create=True) as watermark:
            output, _ = self.tool.srt(path, sampling=Sampling(speed=2, apply_watermark=True), fit_to_timing=False)
            self.assertFalse(infer.call_args.kwargs["apply_watermark"])
            self.assertEqual(len(watermark.call_args.args[0]), 24000)
        wav, rate = sf.read(output)
        self.assertTrue(np.all(wav[:48000] == 0))
        self.assertEqual(len(wav), 48000 + 24000 + 24000)

    def test_text_generates_audio_and_sentence_srt_with_actual_timing(self):
        text = "Câu một. Câu hai?\nCâu ba!"
        clips = [np.full(length, 0.1, dtype=np.float32) for length in (48000, 96000, 48000)]
        with patch.object(self.tool.tts, "infer_batch", return_value=clips) as infer:
            result = self.tool.synthesize_with_subtitles(text, "Mai", sampling=Sampling(batch_size=3))
        self.assertEqual(infer.call_args.args[0], ["Câu một.", "Câu hai?", "Câu ba!"])
        self.assertEqual(result.subtitles, result.audio.with_suffix(".srt"))
        cues = parse_srt(result.subtitles.read_text(encoding="utf-8"))
        self.assertEqual([cue.text for cue in cues], ["Câu một.", "Câu hai?", "Câu ba!"])
        self.assertEqual([(cue.start_ms, cue.end_ms) for cue in cues], [(0, 1000), (1500, 3500), (4200, 5200)])
        self.assertEqual(sf.info(result.audio).frames, 249600)
        self.assertEqual(result.sentences, 3)
        self.assertAlmostEqual(result.duration, 5.2)

    def test_subtitle_timing_tracks_speed_and_excludes_edge_silence(self):
        clip = np.concatenate([np.zeros(4800), np.full(48000, 0.1), np.zeros(9600)]).astype(np.float32)
        with patch.object(self.tool.tts, "infer_batch", return_value=[clip]):
            original = self.tool.synthesize_with_subtitles("Một câu.")
        cue = parse_srt(original.subtitles.read_text(encoding="utf-8"))[0]
        self.assertEqual((cue.start_ms, cue.end_ms), (100, 1100))
        with patch.object(self.tool.tts, "infer_batch", return_value=[self.tone()]):
            faster = self.tool.synthesize_with_subtitles("Một câu.", sampling=Sampling(speed=2))
        cue = parse_srt(faster.subtitles.read_text(encoding="utf-8"))[0]
        self.assertEqual(sf.info(faster.audio).frames, 24000)
        self.assertLessEqual(cue.end_ms, 500)
        self.assertGreater(cue.end_ms, 400)

    def test_sentence_subtitles_pass_clone_and_sampling_through_every_batch(self):
        text = "Câu một. Câu hai! Câu ba?"
        with patch.object(self.tool.tts, "infer_batch", side_effect=lambda texts, **kw: [self.tone() for _ in texts]) as infer:
            result = self.tool.synthesize_with_subtitles(text, "Mai", str(self.reference), Sampling(top_k=19, batch_size=2), "flac")
        self.assertEqual(infer.call_count, 2)
        self.assertEqual(result.audio.suffix, ".flac")
        for call in infer.call_args_list:
            self.assertEqual(call.kwargs["ref_audio"], str(self.reference))
            self.assertEqual(call.kwargs["top_k"], 19)
            self.assertNotIn("voice", call.kwargs)

    def test_cancelled_subtitle_generation_does_not_export_partial_pair(self):
        with self.assertRaises(Cancelled):
            self.tool.synthesize_with_subtitles("Câu một. Câu hai.", sampling=Sampling(batch_size=1),
                                                progress=lambda _: self.tool.stop())
        self.assertEqual(list(self.folder.glob("speech_subtitled_*")), [])
        self.assertTrue(self.tool.synthesize("Again").exists())

    def test_subtitle_write_failure_removes_its_new_audio(self):
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.tool.synthesize_with_subtitles("Một câu.")
        self.assertEqual(list(self.folder.glob("speech_subtitled_*")), [])


if __name__ == "__main__":
    unittest.main()
