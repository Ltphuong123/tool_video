"""Marker file editing workflows without model, decoder or audio devices."""
from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from apps.v3turbo_tool import Cancelled, TurboTool
from apps.video_markers import TimeMarker


OLD_MARKERS = tuple(TimeMarker(index, milliseconds) for index, milliseconds in
                    enumerate((60, 3130, 6290, 8950), 1))
NEW_MARKERS = tuple(TimeMarker(index, milliseconds) for index, milliseconds in
                    enumerate((133, 2995, 5666, 7666), 1))


class MarkerToolWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source = self.folder / "original video.mp4"
        self.source.write_bytes(b"original video bytes")
        self.tool = TurboTool(self.folder)

    def _successful_render(self, source, destination, old_markers, new_markers, **kwargs):
        kwargs["check_stop"]()
        Path(destination).write_bytes(b"rendered aligned video")
        return SimpleNamespace(output_duration=10.716)

    def test_export_forwards_both_marker_files_options_and_current_destination_without_tts(self):
        output_dir = self.folder / "aligned exports"
        self.tool.set_output_dir(output_dir)
        messages = []

        def render(source, destination, old_markers, new_markers, **kwargs):
            self.assertEqual(Path(source), self.source)
            self.assertEqual(Path(destination).parent, output_dir)
            self.assertEqual(old_markers, OLD_MARKERS)
            self.assertEqual(new_markers, NEW_MARKERS)
            self.assertFalse(kwargs["keep_audio"])
            self.assertEqual(kwargs["quality"], 23)
            self.assertEqual(kwargs["preset"], "medium")
            kwargs["progress"]("Aligning marker 4")
            return self._successful_render(source, destination, old_markers, new_markers, **kwargs)

        with patch("apps.video_editor.render_marker_alignment", side_effect=render) as backend, \
             patch.object(self.tool, "load") as load:
            result, note = self.tool.edit_video_markers(
                self.source, OLD_MARKERS, NEW_MARKERS, keep_audio=False, quality=23,
                preset="medium", progress=messages.append,
            )

        backend.assert_called_once()
        load.assert_not_called()
        self.assertIsNone(self.tool.tts)
        self.assertTrue(note)
        self.assertEqual(messages, ["Aligning marker 4"])
        self.assertEqual(result.read_bytes(), b"rendered aligned video")
        self.assertEqual(self.source.read_bytes(), b"original video bytes")
        self.assertEqual([(item.path, item.kind) for item in self.tool.list_outputs()], [(result, "Video")])

    def test_export_snapshots_both_marker_lists_before_renderer_can_change_inputs(self):
        old_markers, new_markers = list(OLD_MARKERS), list(NEW_MARKERS)

        def render(source, destination, old_snapshot, new_snapshot, **kwargs):
            old_markers.clear()
            new_markers.clear()
            self.assertIsInstance(old_snapshot, tuple)
            self.assertIsInstance(new_snapshot, tuple)
            self.assertEqual(old_snapshot, OLD_MARKERS)
            self.assertEqual(new_snapshot, NEW_MARKERS)
            return self._successful_render(source, destination, old_snapshot, new_snapshot, **kwargs)

        with patch("apps.video_editor.render_marker_alignment", side_effect=render):
            output, _ = self.tool.edit_video_markers(self.source, old_markers, new_markers)
        self.assertEqual(output.read_bytes(), b"rendered aligned video")

    def test_invalid_markers_pairing_and_encoder_options_never_start_renderer_or_create_output(self):
        mismatched_ids = (TimeMarker(5, 133), *NEW_MARKERS[1:])
        pairs = (((), NEW_MARKERS), (OLD_MARKERS, ()),
                 (OLD_MARKERS, NEW_MARKERS[:-1]), (OLD_MARKERS, mismatched_ids),
                 ((TimeMarker(1, 0),), (TimeMarker(1, 133),)),
                 ((TimeMarker(1, 60), TimeMarker(2, 60)), NEW_MARKERS[:2]),
                 (OLD_MARKERS[:2], (TimeMarker(1, 133), TimeMarker(2, 100))),
                 (OLD_MARKERS[:2], (TimeMarker(1, 133), TimeMarker(1, 2995))))
        with patch("apps.video_editor.render_marker_alignment") as render:
            for old_markers, new_markers in pairs:
                with self.subTest(old=old_markers, new=new_markers), self.assertRaises(ValueError):
                    self.tool.edit_video_markers(self.source, old_markers, new_markers)
                render.assert_not_called()
            for options in (dict(quality=100), dict(quality=True), dict(preset="unknown")):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    self.tool.edit_video_markers(self.source, OLD_MARKERS, NEW_MARKERS, **options)
                render.assert_not_called()
        self.assertEqual(list(self.folder.glob("video_*.mp4")), [])

    def test_missing_source_never_starts_renderer(self):
        with patch("apps.video_editor.render_marker_alignment") as render:
            with self.assertRaises(ValueError):
                self.tool.edit_video_markers(self.folder / "missing.mp4", OLD_MARKERS, NEW_MARKERS)
        render.assert_not_called()

    def test_cancellation_during_or_after_render_removes_partial_output_and_allows_retry(self):
        for backend_checks_stop in (True, False):
            def render(source, destination, old_markers, new_markers, **kwargs):
                Path(destination).write_bytes(b"partial video")
                self.tool.stop()
                if backend_checks_stop:
                    kwargs["check_stop"]()
                return SimpleNamespace(output_duration=10.716)

            with self.subTest(backend_checks_stop=backend_checks_stop), \
                 patch("apps.video_editor.render_marker_alignment", side_effect=render):
                with self.assertRaises(Cancelled):
                    self.tool.edit_video_markers(self.source, OLD_MARKERS, NEW_MARKERS)
            self.assertEqual(list(self.folder.glob("video_*.mp4")), [])
            self.assertEqual(self.source.read_bytes(), b"original video bytes")

        with patch("apps.video_editor.render_marker_alignment", side_effect=self._successful_render):
            output, _ = self.tool.edit_video_markers(self.source, OLD_MARKERS, NEW_MARKERS)
        self.assertTrue(output.is_file())
        self.assertIsNone(self.tool.tts)

    def test_encoder_failure_removes_partial_output_preserves_source_and_releases_lock(self):
        def render(source, destination, old_markers, new_markers, **kwargs):
            Path(destination).write_bytes(b"partial video")
            raise OSError("Encoder failed")

        with patch("apps.video_editor.render_marker_alignment", side_effect=render):
            with self.assertRaisesRegex(OSError, "Encoder failed"):
                self.tool.edit_video_markers(self.source, OLD_MARKERS, NEW_MARKERS)
        self.assertEqual(list(self.folder.glob("video_*.mp4")), [])
        self.assertEqual(self.source.read_bytes(), b"original video bytes")
        with patch("apps.video_editor.render_marker_alignment", side_effect=self._successful_render):
            output, _ = self.tool.edit_video_markers(self.source, OLD_MARKERS, NEW_MARKERS)
        self.assertTrue(output.is_file())

    def test_export_rejects_concurrent_operation_before_renderer_starts(self):
        with self.tool.operation(require_model=False), \
             patch("apps.video_editor.render_marker_alignment") as render:
            with self.assertRaises(RuntimeError):
                self.tool.edit_video_markers(self.source, OLD_MARKERS, NEW_MARKERS)
        render.assert_not_called()
        self.assertEqual(list(self.folder.glob("video_*.mp4")), [])
