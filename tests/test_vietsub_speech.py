"""Source SRT cues map one-to-one to speech subtitles and old/new markers."""
from pathlib import Path
import importlib.util
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from apps.srt_speech import parse_srt
from apps.video_markers import parse_marker_text
from apps.vietsub_prepare import marker_text
from apps.v3turbo_tool import Cancelled, Sampling, TurboTool
from test_v3turbo_desktop import FakeTurbo
from test_vietsub_prepare import SRT, EXPECTED


class VietsubSpeechTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source = self.folder / "source.srt"
        self.source.write_text(SRT, encoding="utf-8")
        self.factory = Mock(side_effect=FakeTurbo)
        self.tool = TurboTool(self.folder / "results", factory=self.factory)

    def test_prepares_five_files_and_srt_and_both_marker_lists_match_one_to_one(self):
        before = self.source.read_bytes()
        prepared = []
        result = self.tool.prepare_vietsub_with_speech(self.source, sampling=Sampling(),
                                                     model_settings={"threads": 2}, on_prepared=prepared.append)
        self.factory.assert_called_once_with(mode="v3turbo", threads=2)
        self.assertEqual(prepared, [result.preparation])
        self.assertEqual(result.preparation.document.read_text(encoding="utf-8"), " ".join(EXPECTED))
        self.assertEqual(result.preparation.starts_ms, (60, 3130, 6290))
        cues = parse_srt(result.speech.subtitles.read_text(encoding="utf-8"), strict_timing=True)
        self.assertEqual(cues[0].start_ms, result.preparation.starts_ms[0])
        audio, rate = sf.read(result.speech.audio)
        self.assertTrue(np.all(audio[:60 * rate // 1000] == 0))
        self.assertAlmostEqual(cues[-1].end_ms / 1000, sf.info(result.speech.audio).duration, places=3)
        self.assertEqual(len(cues), result.speech.sentences)
        self.assertEqual([cue.text for cue in cues], list(EXPECTED))
        self.assertEqual(len(cues), len(result.preparation.starts_ms))
        self.assertEqual(" ".join(c.text for c in cues), " ".join(EXPECTED))
        self.assertEqual(result.speech.audio.stem, result.speech.subtitles.stem)
        self.assertEqual(result.starts_ms, tuple(cue.start_ms for cue in cues))
        new_markers = parse_marker_text(result.markers.read_text(encoding="utf-8"))
        self.assertEqual([(mark.index, mark.time_ms) for mark in new_markers],
                         [(index, cue.start_ms) for index, cue in enumerate(cues, 1)])
        self.assertEqual(result.markers.stem[-12:], result.preparation.markers.stem[-12:])
        self.assertEqual(len(self.tool.list_outputs()), 5)
        self.assertIn((result.markers, "Mốc thời gian mới"),
                      [(row.path, row.kind) for row in self.tool.list_outputs()])
        self.assertEqual(self.source.read_bytes(), before)

    def test_cue_boundaries_survive_missing_punctuation_and_multiple_sentences(self):
        lines = ("Một mục không có dấu câu", "Câu thứ nhất. Câu thứ hai!", "Một mục không có dấu câu")
        self.source.write_text("\n\n".join(
            f"{i+1}\n00:00:{i*5:02},100 --> 00:00:{i*5+4:02},500\n{line}"
            for i, line in enumerate(lines)), encoding="utf-8")
        self.tool.load()
        sampling = Sampling(speed=1.25, batch_size=2, top_k=19)
        raw = np.full(48000, .1, dtype=np.float32)
        calls = []
        def infer(texts, **kwargs):
            calls.extend(texts)
            self.assertEqual(kwargs["top_k"], 19)
            return [raw.copy() for _ in texts]
        with patch.object(self.tool.tts, "infer_batch", side_effect=infer):
            result = self.tool.prepare_vietsub_with_speech(self.source, sampling=sampling)
        cues = parse_srt(result.speech.subtitles.read_text(encoding="utf-8"), strict_timing=True)
        self.assertEqual(calls, list(lines))
        self.assertEqual([cue.text for cue in cues], list(lines))
        self.assertEqual(len(cues), 3)
        self.assertAlmostEqual(sf.info(result.speech.audio).duration, result.speech.duration)
        for cue in cues:
            self.assertLessEqual(abs(cue.end_ms-cue.start_ms-800), 1)
        self.assertEqual(result.markers.read_text(encoding="utf-8"),
                         marker_text(tuple(cue.start_ms for cue in cues)))

    def test_zero_source_origin_includes_lead_in_without_cutting_generated_audio(self):
        from apps.video_markers import build_marker_time_map
        self.source.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        self.tool.load()
        wav = np.concatenate((np.zeros(4800), np.full(4800, .1))).astype(np.float32)
        with patch.object(self.tool.tts, "infer_batch", return_value=[wav]):
            result = self.tool.prepare_vietsub_with_speech(self.source)
        actual, _ = sf.read(result.speech.audio, dtype="float32")
        np.testing.assert_allclose(actual, wav, atol=1/32768)
        self.assertEqual(result.starts_ms, (0,))
        mapping = build_marker_time_map(2, parse_marker_text(result.preparation.markers.read_text()),
                                        parse_marker_text(result.markers.read_text()))
        self.assertEqual(mapping.output_duration, 2)

    @unittest.skipUnless(importlib.util.find_spec("moviepy"), "optional MoviePy absent")
    def test_generated_markers_retime_a_real_video_without_changing_frame_payloads(self):
        from apps.video_fast_export import _ffmpeg_binary
        from apps.video_markers import build_marker_time_map, read_marker_file
        from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos
        video = self.folder / "source.mp4"
        binary = _ffmpeg_binary()
        subprocess.run([binary, "-hide_banner", "-loglevel", "error", "-nostdin",
                        "-f", "lavfi", "-i", "testsrc2=s=64x48:r=24:d=10",
                        "-c:v", "libx264", "-preset", "ultrafast", "-bf", "2", "-an", str(video)],
                       check=True, capture_output=True)
        speech = self.tool.prepare_vietsub_with_speech(self.source, model_settings={})
        old, new = read_marker_file(speech.preparation.markers), read_marker_file(speech.markers)
        mapping = build_marker_time_map(10, old, new)
        output, _ = self.tool.edit_video_markers(video, old, new, export_mode="copy")
        def payloads(path):
            result = subprocess.run([binary, "-hide_banner", "-loglevel", "error", "-nostdin",
                                     "-i", str(path), "-map", "0:v:0", "-c:v", "copy",
                                     "-f", "framehash", "-hash", "sha256", "-"],
                                    check=True, capture_output=True, text=True)
            return [line.rsplit(",", 1)[-1].strip() for line in result.stdout.splitlines()
                    if line.strip() and not line.startswith("#")]
        source_hashes, output_hashes = payloads(video), payloads(output)
        self.assertEqual(len(source_hashes), 240)
        self.assertEqual(output_hashes, source_hashes)
        self.assertEqual(mapping.segments[-1].speed, 1)
        self.assertAlmostEqual(ffmpeg_parse_infos(str(output))['duration'], mapping.output_duration, delta=.04)

    def test_existing_model_is_reused_with_selected_voice_settings_and_format(self):
        self.tool.load()
        model = self.tool.tts
        self.factory.reset_mock()
        result = self.tool.prepare_vietsub_with_speech(self.source, "Alias", Sampling(top_k=19), "flac",
                                                     model_settings={"threads": 10})
        self.factory.assert_not_called()
        self.assertIs(self.tool.tts, model)
        self.assertFalse(model.closed)
        self.assertEqual(result.speech.audio.suffix, ".flac")
        self.assertTrue(all(kwargs["voice"] == "Alias" and kwargs["top_k"] == 19 for text, kwargs in model.calls))

    def test_lock_is_held_across_both_preparation_and_speech(self):
        def prepared(result):
            with self.assertRaises(RuntimeError):
                self.tool.set_output_dir(self.folder / "another")
            with self.assertRaises(RuntimeError):
                self.tool.synthesize_with_subtitles("Other job.")
        result = self.tool.prepare_vietsub_with_speech(self.source, model_settings={}, on_prepared=prepared)
        self.assertEqual(result.speech.audio.parent, result.preparation.document.parent)

    def test_stop_after_preparation_does_not_clear_cancel_or_start_model(self):
        completed = []
        def prepared(result):
            completed.append(result)
            self.tool.stop()
        with self.assertRaises(Cancelled):
            self.tool.prepare_vietsub_with_speech(self.source, model_settings={}, on_prepared=prepared)
        self.factory.assert_not_called()
        self.assertTrue(completed[0].document.is_file())
        self.assertTrue(completed[0].markers.is_file())
        self.assertEqual(len(self.tool.list_outputs()), 2)
        self.assertTrue(self.tool.stop_event.is_set())
        self.assertTrue(self.tool.prepare_vietsub(self.source).document.is_file())

    def test_stop_during_model_load_prevents_synthesis(self):
        def factory(**kwargs):
            self.tool.stop()
            return FakeTurbo(**kwargs)
        self.tool._factory = factory
        with self.assertRaises(Cancelled):
            self.tool.prepare_vietsub_with_speech(self.source, model_settings={})
        self.assertEqual(self.tool.tts.calls, [])
        self.assertEqual(len(self.tool.list_outputs()), 2)

    def test_stop_during_speech_keeps_prepared_files_and_no_partial_audio_pair(self):
        def progress(message):
            if "Audio + SRT:" in message:
                self.tool.stop()
        with self.assertRaises(Cancelled):
            self.tool.prepare_vietsub_with_speech(self.source, model_settings={}, progress=progress)
        self.assertEqual(len(self.tool.list_outputs()), 2)
        self.assertFalse(list(self.tool.output_dir.glob("speech_subtitled_*")))

    def test_model_load_failure_preserves_txt_markers_and_releases_lock(self):
        self.factory.side_effect = RuntimeError("load failed")
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            self.tool.prepare_vietsub_with_speech(self.source, model_settings={})
        self.assertEqual(len(self.tool.list_outputs()), 2)
        self.assertIsNone(self.tool.tts)
        self.assertTrue(self.tool.prepare_vietsub(self.source).document.is_file())

    def test_subtitle_write_failure_removes_new_audio_but_preserves_preparation(self):
        original_write = Path.write_text
        def write(path, *args, **kwargs):
            if path.name.startswith("speech_subtitled_") and path.suffix == ".srt":
                raise OSError("subtitle write failed")
            return original_write(path, *args, **kwargs)
        with patch.object(Path, "write_text", write), self.assertRaisesRegex(OSError, "subtitle write failed"):
            self.tool.prepare_vietsub_with_speech(self.source, model_settings={})
        self.assertEqual(len(self.tool.list_outputs()), 2)
        self.assertFalse(list(self.tool.output_dir.glob("speech_subtitled_*")))

    def test_bad_srt_does_not_load_model(self):
        self.source.write_text("invalid", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.tool.prepare_vietsub_with_speech(self.source, model_settings={})
        self.factory.assert_not_called()
        self.assertEqual(self.tool.list_outputs(), [])

    def test_new_marker_write_failure_removes_new_outputs_and_keeps_source_markers(self):
        old_markers = marker_text((60, 3130, 6290))
        with patch("apps.vietsub_prepare.marker_text", side_effect=[old_markers, OSError("new marker write failed")]):
            with self.assertRaisesRegex(OSError, "new marker write failed"):
                self.tool.prepare_vietsub_with_speech(self.source, model_settings={})
        self.assertEqual(len(self.tool.list_outputs()), 2)
        self.assertFalse(list(self.tool.output_dir.glob("speech_subtitled_*")))
        self.assertFalse(list(self.tool.output_dir.glob("vietsub_new_marks_*")))
        self.assertEqual(next(self.tool.output_dir.glob("vietsub_marks_*")).read_text(encoding="utf-8"), old_markers)
        self.assertTrue(self.tool.prepare_vietsub(self.source).markers.is_file())

    def test_stop_at_new_marker_export_removes_new_outputs_and_preserves_preparation(self):
        def progress(message):
            if "mốc bắt đầu của SRT mới" in message:
                self.tool.stop()
        with self.assertRaises(Cancelled):
            self.tool.prepare_vietsub_with_speech(self.source, model_settings={}, progress=progress)
        self.assertEqual(len(self.tool.list_outputs()), 2)
        self.assertFalse(list(self.tool.output_dir.glob("speech_subtitled_*")))
        self.assertFalse(list(self.tool.output_dir.glob("vietsub_new_marks_*")))

    def test_existing_new_marker_file_is_never_overwritten_or_removed_on_failure(self):
        existing = []
        def prepared(result):
            path = result.markers.with_name(result.markers.name.replace("vietsub_marks_", "vietsub_new_marks_", 1))
            path.write_text("keep this file", encoding="utf-8")
            existing.append(path)
        with self.assertRaises(FileExistsError):
            self.tool.prepare_vietsub_with_speech(self.source, model_settings={}, on_prepared=prepared)
        self.assertEqual(existing[0].read_text(encoding="utf-8"), "keep this file")
        self.assertFalse(list(self.tool.output_dir.glob("speech_subtitled_*")))


if __name__ == "__main__":
    unittest.main()
