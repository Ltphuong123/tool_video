"""SRT -> one plain-text paragraph and exact numbered start markers."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from apps.vietsub_prepare import prepare_vietsub_files
from apps.video_markers import parse_marker_text
from apps.v3turbo_tool import Cancelled, TurboTool


SRT = """42
00:00:00,060 --> 00:00:03,000
<b>Xin chào &amp; cảm ơn.</b>
Dòng thứ hai.

77
00:00:03,130 --> 00:00:06,000
2026

91
00:00:06,290 --> 00:00:08,700
{\\an8}Hẹn gặp lại!
"""
EXPECTED = ("Xin chào & cảm ơn. Dòng thứ hai.", "2026", "Hẹn gặp lại!")


class VietsubPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source = self.folder / "input.srt"
        self.source.write_text(SRT, encoding="utf-8-sig")
        self.document = self.folder / "script.txt"
        self.markers = self.folder / "starts.txt"

    def export(self, **kwargs):
        return prepare_vietsub_files(self.source, self.document, self.markers, **kwargs)

    def test_text_contains_one_dialogue_paragraph_and_markers_match_every_cue(self):
        before = self.source.read_bytes()
        result = self.export()
        content = result.document.read_text(encoding="utf-8")
        self.assertEqual(content, " ".join(EXPECTED))
        self.assertNotIn("\n", content)
        self.assertNotIn("\r", content)
        self.assertEqual(result.lines, EXPECTED)
        self.assertNotIn("-->", content)
        self.assertEqual(result.starts_ms, (60, 3130, 6290))
        markers = parse_marker_text(result.markers.read_text(encoding="utf-8"))
        self.assertEqual([(m.index, m.time_ms) for m in markers], [(1, 60), (2, 3130), (3, 6290)])
        self.assertEqual(len(markers), len(EXPECTED))
        self.assertFalse(list(self.folder.glob("*.docx")))
        self.assertEqual(self.source.read_bytes(), before)

    def test_utf16_bom_and_missing_blank_lines_preserve_vietnamese(self):
        self.source.write_text(SRT.replace("\n\n", "\n"), encoding="utf-16")
        self.assertEqual(self.export().lines, EXPECTED)
        self.assertEqual(self.document.read_text(encoding="utf-8"), " ".join(EXPECTED))

    def test_invalid_srt_never_leaves_output_files(self):
        bad_inputs = ("", "not subtitles", "1\n00:00:03,000 --> 00:00:02,000\nText",
                      "1\n00:99:00,000 --> 01:00:01,000\nText",
                      "1\n00:00:00,000 --> invalid\nText", SRT + "\x00",
                      "1\n00:00:05,000 --> 00:00:06,000\nA\n\n2\n00:00:01,000 --> 00:00:02,000\nB")
        for text in bad_inputs:
            with self.subTest(text=text):
                self.source.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.export()
                self.assertFalse(self.document.exists())
                self.assertFalse(self.markers.exists())

    def test_bad_encoding_is_reported_before_export(self):
        self.source.write_bytes(b"\xffinvalid")
        with self.assertRaisesRegex(ValueError, "UTF"):
            self.export()
        self.assertFalse(self.document.exists())

    def test_simultaneous_cues_are_not_lost_or_shifted(self):
        self.source.write_text("1\n00:00:01,000 --> 00:00:02,000\nA\n\n2\n00:00:01,000 --> 00:00:03,000\nB", encoding="utf-8")
        result = self.export()
        self.assertEqual(result.lines, ("A", "B"))
        self.assertEqual(result.starts_ms, (1000, 1000))
        self.assertEqual(self.markers.read_text(encoding="utf-8").count("00:00:01,000"), 2)

    def test_existing_outputs_are_protected(self):
        for path in (self.document, self.markers):
            with self.subTest(path=path):
                path.write_bytes(b"existing")
                with self.assertRaises(ValueError):
                    self.export()
                self.assertEqual(path.read_bytes(), b"existing")
                path.unlink()

    def test_cancellation_after_text_starts_cleans_both_files(self):
        stopped = False

        def progress(message):
            nonlocal stopped
            stopped = "TXT" in message

        def check():
            if stopped:
                raise Cancelled("stopped")

        with self.assertRaises(Cancelled):
            self.export(check_stop=check, progress=progress)
        self.assertFalse(self.document.exists())
        self.assertFalse(self.markers.exists())

    def test_failure_in_second_file_removes_completed_text_and_partial_markers(self):
        with patch("apps.vietsub_prepare.marker_text", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                self.export()
        self.assertFalse(self.document.exists())
        self.assertFalse(self.markers.exists())

    def test_tool_exports_to_selected_folder_without_model_and_registers_both_files(self):
        tool = TurboTool(self.folder / "outputs")
        destination = self.folder / "chosen"
        tool.set_output_dir(destination)
        with patch.object(tool, "load", side_effect=AssertionError("model unused")):
            result = tool.prepare_vietsub(self.source)
        self.assertEqual(result.document.parent, destination)
        self.assertEqual(result.markers.parent, destination)
        self.assertEqual(result.document.stem[-12:], result.markers.stem[-12:])
        self.assertEqual({(row.path, row.kind) for row in tool.list_outputs()},
                         {(result.document, "Lời thoại"), (result.markers, "Mốc thời gian")})
        self.assertIsNone(tool.tts)
        self.assertEqual(set(tool.delete_outputs([result.document, result.markers])),
                         {result.document, result.markers})
        self.assertTrue(self.source.is_file())

    def test_history_does_not_claim_unrelated_files(self):
        self.assertIsNone(TurboTool._output_kind(Path("speech_123456789abc.docx")))
        self.assertIsNone(TurboTool._output_kind(Path("video_123456789abc.txt")))
        self.assertIsNone(TurboTool._output_kind(Path("vietsub_marks_123456789abc.wav")))


if __name__ == "__main__":
    unittest.main()
