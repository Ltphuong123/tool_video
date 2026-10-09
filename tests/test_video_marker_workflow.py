"""Marker file editing workflows without model, decoder or audio devices."""
from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import Mock, patch

from apps.v3turbo_desktop import DesktopApp
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


class MarkerFakeVideoPreview(ttk.Frame):
    """A layout surface with observable controls and no decoder thread."""

    def __init__(self, parent, on_position=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, width=480, height=270, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.ready = self.playing = False
        self.duration = self.position = 0.0
        self.fps = 30.0
        self.on_position = on_position or (lambda seconds, playing: None)
        self.open = Mock()
        self.set_segments = Mock()
        self.set_time_map = Mock()
        self.set_audio_enabled = Mock()
        self.seek = Mock(side_effect=self._seek)
        self.play = Mock(side_effect=lambda: setattr(self, "playing", True))
        self.pause = Mock(side_effect=lambda: setattr(self, "playing", False))
        self.unload = Mock()
        self.close = Mock()

    def _seek(self, seconds):
        self.position = seconds
        self.on_position(seconds, self.playing)


class MarkerDesktopWorkflowTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display is unavailable: {exc}")
        self.root.withdraw()
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.tool = TurboTool(self.folder / "output")
        self.preview_patch = patch("apps.v3turbo_desktop.VideoPreview", MarkerFakeVideoPreview)
        self.preview_patch.start()
        self.addCleanup(self.preview_patch.stop)
        try:
            self.app = DesktopApp(self.root, self.tool)
        except Exception:
            self.root.destroy()
            self.temp.cleanup()
            raise
        self.root.update_idletasks()
        self.old_file = self._marker_file("old times.txt", OLD_MARKERS)
        self.new_file = self._marker_file("new times.txt", NEW_MARKERS)

    def tearDown(self):
        if hasattr(self, "app"):
            for identifier in self.root.tk.call("after", "info"):
                self.root.after_cancel(identifier)
            self.app.busy = False
            self.app.close()
        if hasattr(self, "temp"):
            self.temp.cleanup()

    def _marker_file(self, name, markers):
        path = self.folder / name
        lines = []
        for marker in markers:
            seconds, milliseconds = divmod(marker.time_ms, 1000)
            minutes, seconds = divmod(seconds, 60)
            hours, minutes = divmod(minutes, 60)
            lines.append(f"{marker.index}\n{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}\n")
        path.write_text("\n".join(lines), encoding="utf-8-sig")
        return path

    def _load_video(self, duration=12):
        source = self.folder / "original.mp4"
        source.write_bytes(b"original video")
        self.app.video_path.set(str(source))
        self.app.video_preview.ready = True
        self.app.video_preview.duration = duration
        self.app._video_loaded(dict(duration=duration, fps=30, width=1920, height=1080,
                                    has_audio=True))
        return source

    def _import(self, kind, path):
        with patch("apps.v3turbo_desktop.filedialog.askopenfilename", return_value=str(path)):
            self.app.import_video_markers(kind)

    def _import_pair(self):
        self._import("old", self.old_file)
        self._import("new", self.new_file)

    def _manual_segment(self):
        self.app.video_start.set("2")
        self.app.video_end.set("4")
        self.app.video_speed.set("2")
        self.app.video_ramp.set("0")
        self.app.add_video_segment()
        return list(self.app.video_segments)

    def test_import_pins_original_indices_and_attaches_target_markers(self):
        self._load_video()
        self._import("old", self.old_file)
        self.assertEqual(self.app.video_old_markers, OLD_MARKERS)
        self.assertEqual(self.app.video_old_marks_path.get(), str(self.old_file))
        self.assertEqual(self.app.video_timeline.markers, OLD_MARKERS)
        self.assertIsNone(self.app.video_marker_mapping)
        self._import("new", self.new_file)
        self.assertEqual(self.app.video_new_markers, NEW_MARKERS)
        self.assertEqual(self.app.video_new_marks_path.get(), str(self.new_file))
        self.assertEqual(self.app.video_timeline.markers, OLD_MARKERS)
        self.assertEqual(self.app.video_timeline.target_markers, NEW_MARKERS)

    def test_alignment_updates_preview_timeline_and_exact_anchor_times_with_one_x_tail(self):
        self._load_video()
        self._import_pair()
        self.app.align_video_markers()
        mapping = self.app.video_marker_mapping
        self.assertIsNotNone(mapping)
        self.app.video_preview.set_time_map.assert_called_with(mapping)
        self.assertEqual(self.app.video_segments, list(mapping.segments))
        self.assertEqual(self.app.video_timeline.segments, list(mapping.segments))
        self.assertFalse(self.app.video_timeline.editable)
        for old_marker, new_marker in zip(OLD_MARKERS, NEW_MARKERS):
            self.assertAlmostEqual(mapping.output_time(old_marker.time_ms / 1000),
                                   new_marker.time_ms / 1000, places=12)
        self.assertAlmostEqual(mapping.output_duration, 10.716, places=12)
        self.assertEqual(mapping.segments[-1].speed, 1)
        self.assertIsNone(self.tool.tts)

    def test_export_aligned_video_routes_both_marker_snapshots_and_audio_options(self):
        source = self._load_video()
        self._import_pair()
        self.app.align_video_markers()
        result = self.folder / "output" / "video_000000000001.mp4"
        with patch.object(self.tool, "edit_video_markers", return_value=(result, "Aligned")) as aligned, \
             patch.object(self.tool, "edit_video_segments") as manual, \
             patch.object(self.app, "_job", side_effect=lambda function, *args: function()):
            self.app.export_video()
            self.assertEqual(aligned.call_args.args, (str(source), OLD_MARKERS, NEW_MARKERS))
            self.assertEqual(aligned.call_args.kwargs["keep_audio"], False)
            self.assertEqual(aligned.call_args.kwargs["quality"], 20)
            self.assertEqual(aligned.call_args.kwargs["preset"], "fast")
            self.assertTrue(callable(aligned.call_args.kwargs["progress"]))
        manual.assert_not_called()
        self.assertIsNone(self.tool.tts)

    def test_invalid_file_and_cancelled_import_preserve_applied_alignment_and_paths(self):
        self._load_video()
        self._import_pair()
        self.app.align_video_markers()
        previous_mapping = self.app.video_marker_mapping
        broken = self.folder / "broken.txt"
        broken.write_text("1\nnot a timestamp", encoding="utf-8")
        with self.assertRaises(ValueError):
            self._import("old", broken)
        self.assertIs(self.app.video_marker_mapping, previous_mapping)
        self.assertEqual(self.app.video_old_markers, OLD_MARKERS)
        self.assertEqual(self.app.video_old_marks_path.get(), str(self.old_file))
        with patch("apps.v3turbo_desktop.filedialog.askopenfilename", return_value=""):
            self.app.import_video_markers("new")
        self.assertIs(self.app.video_marker_mapping, previous_mapping)
        self.assertEqual(self.app.video_new_markers, NEW_MARKERS)
        self.assertEqual(self.app.video_new_marks_path.get(), str(self.new_file))

    def test_failed_alignment_preserves_applied_manual_segments(self):
        self._load_video()
        manual_segments = self._manual_segment()
        self._import("old", self.old_file)
        incomplete = self._marker_file("missing marker.txt", NEW_MARKERS[:-1])
        self._import("new", incomplete)
        with self.assertRaises(ValueError):
            self.app.align_video_markers()
        self.assertIsNone(self.app.video_marker_mapping)
        self.assertEqual(self.app.video_segments, manual_segments)
        self.assertEqual(self.app.video_timeline.segments, manual_segments)
        self.assertTrue(self.app.video_timeline.editable)

    def test_clear_alignment_restores_manual_plan_and_retains_marker_pins(self):
        self._load_video()
        manual_segments = self._manual_segment()
        self._import_pair()
        self.app.align_video_markers()
        self.app.clear_video_marker_alignment()
        self.assertIsNone(self.app.video_marker_mapping)
        self.assertEqual(self.app.video_segments, manual_segments)
        self.assertEqual(self.app.video_timeline.segments, manual_segments)
        self.app.video_preview.set_segments.assert_called_with(tuple(manual_segments))
        self.assertTrue(self.app.video_timeline.editable)
        self.assertEqual(self.app.video_timeline.markers, OLD_MARKERS)
        self.assertEqual(self.app.video_old_markers, OLD_MARKERS)
        self.assertEqual(self.app.video_new_markers, NEW_MARKERS)

    def test_changing_source_clears_alignment_and_marker_files(self):
        self._load_video()
        self._import_pair()
        self.app.align_video_markers()
        self.app.video_path.set(str(self.folder / "different.mp4"))
        self.assertIsNone(self.app.video_marker_mapping)
        self.assertEqual(self.app.video_old_markers, ())
        self.assertEqual(self.app.video_new_markers, ())
        self.assertEqual(self.app.video_old_marks_path.get(), "")
        self.assertEqual(self.app.video_new_marks_path.get(), "")
        self.assertEqual(self.app.video_timeline.markers, ())
        self.assertEqual(self.app.video_timeline.target_markers, ())
        self.assertEqual(self.app.video_segments, [])

    def test_out_of_bounds_old_file_is_rejected_before_replacing_pins(self):
        self._load_video(duration=12)
        self._import("old", self.old_file)
        beyond_end = self._marker_file("outside video.txt", (TimeMarker(1, 12001),))
        with self.assertRaises(ValueError):
            self._import("old", beyond_end)
        self.assertEqual(self.app.video_old_markers, OLD_MARKERS)
        self.assertEqual(self.app.video_timeline.markers, OLD_MARKERS)

    def test_applied_alignment_locks_manual_edits_but_allows_marker_seeking(self):
        self._load_video()
        self._import_pair()
        self.app.align_video_markers()
        mapping = self.app.video_marker_mapping
        segments = list(self.app.video_segments)
        for edit in (self.app.add_video_segment, self.app.apply_video_segment,
                     self.app.delete_video_segment, self.app.clear_video_segments):
            with self.subTest(edit=edit.__name__), self.assertRaises(ValueError):
                edit()
        self.app._video_speed_drag("4")
        self.assertIs(self.app.video_marker_mapping, mapping)
        self.assertEqual(self.app.video_segments, segments)
        self.assertFalse(self.app.video_timeline.editable)
        for widget in [*self.app.video_manual_fields, self.app.video_speed_control,
                       *self.app.video_manual_buttons]:
            self.assertIn("disabled", widget.state())
        self.app.video_marker_table.selection_set("2")
        self.app._video_marker_selected()
        self.app.video_preview.seek.assert_called_with(3.13)
