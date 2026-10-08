"""Desktop workflow checks without model downloads or audio device access."""
from __future__ import annotations

import atexit
from pathlib import Path
import tempfile
import time
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import Mock, patch
import zipfile

import soundfile as sf

from apps.v3turbo_desktop import DesktopApp
from apps.v3turbo_tool import Sampling, TurboTool
from apps.video_editor import SpeedSegment
from apps.video_timeline import VideoTimeline


class FakeVideoPreview(ttk.Frame):
    """A real Tk layout surface with no decoder thread or FFmpeg reader."""

    def __init__(self, parent, on_position=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, width=480, height=270, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.ready = False
        self.playing = False
        self.duration = self.position = 0.0
        self.fps = 30.0
        self.on_position = on_position or (lambda seconds, playing: None)
        self.open = Mock()
        self.set_segments = Mock()
        self.set_audio_enabled = Mock()
        self.seek = Mock(side_effect=self._seek)
        self.play = Mock(side_effect=lambda: setattr(self, "playing", True))
        self.pause = Mock(side_effect=lambda: setattr(self, "playing", False))
        self.unload = Mock()
        self.close = Mock()

    def _seek(self, seconds):
        self.position = seconds
        self.on_position(seconds, self.playing)


class DesktopUITests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display is unavailable: {exc}")
        self.root.withdraw()
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.destination = self.folder / "first"
        self.destination.mkdir()
        self.settings = self.folder / "desktop_settings.json"
        self.speech = self.destination / "speech_000000000001.wav"
        self.paired_audio = self.destination / "speech_subtitled_000000000002.wav"
        self.subtitle = self.paired_audio.with_suffix(".srt")
        for path in (self.speech, self.paired_audio):
            sf.write(path, [.0, .1, -.1, .0], 48000)
        self.subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nExample.\n", encoding="utf-8")
        self.archive = self.destination / "batch_000000000003.zip"
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("manifest.json", "[]")
        self.voice_export = self.destination / "voices_000000000004.json"
        self.voice_export.write_text('{"presets": {}}', encoding="utf-8")
        self.reference = self.destination / "reference_000000000005.npz"
        self.reference.write_bytes(b"reference")
        self.library = self.destination / "user_voices.json"
        self.library.write_text('{"presets": {}}', encoding="utf-8")
        self.input_audio = self.destination / "my_input.wav"
        self.input_audio.write_bytes(b"input")
        self.tool = TurboTool(self.destination, settings_path=self.settings)
        self.preview_patch = patch("apps.v3turbo_desktop.VideoPreview", FakeVideoPreview)
        self.preview_patch.start()
        self.addCleanup(self.preview_patch.stop)
        try:
            self.app = DesktopApp(self.root, self.tool)
        except Exception:
            self.root.destroy()
            self.temp.cleanup()
            raise
        self.root.update_idletasks()

    def tearDown(self):
        if hasattr(self, "app"):
            self._dispose_app()
        if hasattr(self, "temp"):
            self.temp.cleanup()

    def _dispose_app(self):
        # A second Tk interpreter should not inherit callbacks from an old root.
        for identifier in self.root.tk.call("after", "info"):
            self.root.after_cancel(identifier)
        self.app.busy = False
        self.app.recorder = None
        self.app.close()
        atexit.unregister(self.app._terminate_children)

    def _paths(self):
        return set(self.app.audio_files.values())

    def _select(self, *paths):
        selected = [item for item, path in self.app.audio_files.items() if path in paths]
        self.assertEqual(len(selected), len(paths))
        self.app.history.selection_set(selected)
        self.app._history_selection()

    def _layout_diagnostics(self, widget=None):
        widget = self.app.book if widget is None else widget
        current = self.app.book.select()
        page = next((page for page in self.app.pages.values() if str(page) == current), self.app.book)
        return (f"root={self.root.geometry()}, root_state={self.root.state()}, "
                f"root_height={self.root.winfo_height()}, book_height={self.app.book.winfo_height()}, "
                f"current_tab={current}, page_height={page.winfo_height()}, widget={widget}, "
                f"mapped={widget.winfo_ismapped()}, viewable={widget.winfo_viewable()}, "
                f"widget_y={widget.winfo_rooty()}, widget_height={widget.winfo_height()}")

    def _pump_until(self, predicate, widget=None):
        deadline = time.monotonic() + 2
        while True:
            self.root.update()
            if predicate():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def _wait_for_window(self):
        mapped = self._pump_until(lambda: self.root.winfo_ismapped() and self.root.winfo_viewable()
                                  and self.app.book.winfo_ismapped() and self.app.book.winfo_viewable())
        self.assertTrue(mapped, self._layout_diagnostics())

    def _wait_for_page(self, title):
        page = self.app.pages[title]
        self._pump_until(lambda: page.winfo_ismapped() and page.winfo_viewable(), page)
        self.assertTrue(page.winfo_ismapped(), self._layout_diagnostics(page))

    def _load_video(self, duration=40):
        source = self.folder / "original video.mp4"
        source.write_bytes(b"input video")
        self.app.video_path.set(str(source))
        self.app.video_preview.ready = True
        self.app.video_preview.duration = duration
        self.app._video_loaded(dict(duration=duration, fps=30, width=1920, height=1080,
                                    has_audio=True))
        return source

    def _add_segment(self, start, end, speed=1.5, ramp=0.5):
        self.app.video_start.set(str(start))
        self.app.video_end.set(str(end))
        self.app.video_speed.set(str(speed))
        self.app.video_ramp.set(str(ramp))
        self.app.add_video_segment()

    def test_startup_restores_generated_history_and_preserves_speed_controls(self):
        self.assertEqual(len(self.app.pages), 9)
        self.assertEqual(self.app.book.select(), str(self.app.pages["Văn bản"]))
        self.assertEqual(self._paths(), {
            self.speech, self.paired_audio, self.subtitle,
            self.archive, self.voice_export, self.reference,
        })
        self.assertNotIn(self.library, self._paths())
        self.assertNotIn(self.input_audio, self._paths())
        self.assertIsNone(self.tool.tts)
        self.assertEqual(self.app._sampling(), Sampling())
        self.assertNotIn("speed_method", self.app.vars)
        self.assertEqual(float(self.app.srt_min_speed.get()), 1.0)
        self.assertEqual(str(self.app.history.cget("selectmode")), "extended")
        self.assertIn("Video", self.app.pages)
        self.assertEqual(self.app.video_path.get(), "")
        self.assertEqual(float(self.app.video_start.get()), 0.0)
        self.assertEqual(float(self.app.video_end.get()), 10.0)
        self.assertEqual(float(self.app.video_speed.get()), 1.5)
        self.assertEqual(float(self.app.video_ramp.get()), 0.5)
        self.assertTrue(self.app.video_keep_audio.get())

    def test_video_export_forwards_all_applied_segments_and_audio_without_loading_tts(self):
        source = self._load_video()
        self._add_segment(2, 5, speed=2, ramp=0.5)
        self._add_segment(12.5, 23.75, speed=0.8, ramp=1.25)
        expected = tuple(self.app.video_segments)
        result = self.destination / "video_000000000006.mp4"
        with patch.object(self.tool, "edit_video_segments", return_value=(result, "Finished")) as render, \
             patch.object(self.app, "_job", side_effect=lambda function, *args: function()):
            for keep_audio in (True, False):
                self.app.video_keep_audio.set(keep_audio)
                self.app.export_video()
                self.assertEqual(render.call_args.args, (str(source), expected))
                self.assertEqual(render.call_args.kwargs["keep_audio"], keep_audio)
                self.assertEqual(render.call_args.kwargs["quality"], 20)
                self.assertEqual(render.call_args.kwargs["preset"], "fast")
                self.assertTrue(callable(render.call_args.kwargs["progress"]))
        self.assertIsNone(self.tool.tts)

    def test_preview_sound_toggle_is_independent_of_export_sound(self):
        toggle = next(widget for widget in self.app.video_play_button.master.winfo_children()
                      if isinstance(widget, ttk.Checkbutton) and widget.cget("text") == "Âm thanh")
        self.assertTrue(self.app.video_sound.get())
        self.app.video_keep_audio.set(True)
        toggle.invoke()
        self.assertFalse(self.app.video_sound.get())
        self.app.video_preview.set_audio_enabled.assert_called_once_with(False)
        self.assertTrue(self.app.video_keep_audio.get())
        toggle.invoke()
        self.app.video_preview.set_audio_enabled.assert_called_with(True)

    def test_video_segment_form_rejects_non_numeric_fields_before_mutating_selection(self):
        self._load_video()
        self._add_segment(2, 5)
        expected = list(self.app.video_segments)
        for variable in (self.app.video_start, self.app.video_end,
                         self.app.video_speed, self.app.video_ramp):
            previous = variable.get()
            with self.subTest(field=str(variable)), patch.object(self.app, "_job") as job:
                variable.set("invalid number")
                try:
                    for action in (self.app.add_video_segment, self.app.apply_video_segment):
                        with self.assertRaises(ValueError):
                            action()
                    self.assertEqual(self.app.video_segments, expected)
                    job.assert_not_called()
                finally:
                    variable.set(previous)

    def test_video_segments_add_update_delete_keep_preview_and_timeline_in_sync(self):
        self._load_video()
        self._add_segment(12, 16, speed=0.8, ramp=1)
        self._add_segment(2, 5, speed=2, ramp=0.5)
        self.assertEqual(self.app.video_segments,
                         [SpeedSegment(2, 5, 2, 0.5), SpeedSegment(12, 16, 0.8, 1)])
        self.app._video_select(2, 5, 0)
        self.assertEqual(float(self.app.video_speed.get()), 2)
        self.assertEqual(float(self.app.video_ramp.get()), 0.5)
        self.app.video_speed.set("0.75")
        self.app.video_ramp.set("0.25")
        self.app.apply_video_segment()
        expected = [SpeedSegment(2, 5, 0.75, 0.25), SpeedSegment(12, 16, 0.8, 1)]
        self.assertEqual(self.app.video_segments, expected)
        self.assertEqual(self.app.video_timeline.segments, expected)
        self.assertEqual(list(self.app.video_preview.set_segments.call_args.args[0]), expected)
        self.app.delete_video_segment()
        self.assertEqual(self.app.video_segments, expected[1:])
        self.assertEqual(self.app.video_timeline.segments, expected[1:])
        self.assertEqual(list(self.app.video_preview.set_segments.call_args.args[0]), expected[1:])

    def test_video_speed_edits_preserve_exact_adjacent_fractional_boundaries(self):
        boundary, duration = 1 / 3, 2 / 3
        self._load_video(duration=duration)
        first = SpeedSegment(0, boundary, speed=1, ramp_seconds=0)
        second = SpeedSegment(boundary, duration, speed=0.8, ramp_seconds=0)
        self.app._video_segments_changed([first, second], 1)
        self.app._video_select(boundary, duration, 1)

        with patch("apps.v3turbo_desktop.messagebox.showerror") as error:
            self.app._video_speed_drag("2.0")
        error.assert_not_called()
        self.assertEqual(self.app.video_segments[0], first)
        self.assertEqual(self.app.video_segments[1], SpeedSegment(boundary, duration, 2, 0))

        self.app.video_speed.set("1.25")
        self.app.apply_video_segment()
        expected = [first, SpeedSegment(boundary, duration, 1.25, 0)]
        self.assertEqual(self.app.video_segments, expected)
        self.assertEqual(self.app.video_timeline.segments, expected)
        self.assertEqual(list(self.app.video_preview.set_segments.call_args.args[0]), expected)

    def test_video_segments_reject_overlap_and_bounds_without_changing_applied_plan(self):
        self._load_video(duration=20)
        self._add_segment(2, 5)
        expected = list(self.app.video_segments)
        for start, end in ((4, 7), (-1, 1), (17, 21), (9, 9)):
            self.app.video_start.set(str(start))
            self.app.video_end.set(str(end))
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.app.add_video_segment()
            self.assertEqual(self.app.video_segments, expected)
            self.assertEqual(self.app.video_timeline.segments, expected)

    def test_video_timeline_callbacks_apply_drag_and_seek_the_embedded_preview(self):
        self._load_video(duration=20)
        self._add_segment(2, 5, speed=0.8)
        self._add_segment(10, 14, speed=2)
        moved = [SpeedSegment(4, 8, speed=0.8), SpeedSegment(10, 14, speed=2)]
        self.app._video_segments_changed(moved, 0)
        self.assertEqual(self.app.video_segments, moved)
        self.assertEqual(self.app.video_selected, 0)
        self.assertEqual(float(self.app.video_start.get()), 4)
        self.assertEqual(float(self.app.video_end.get()), 8)
        self.assertEqual(list(self.app.video_preview.set_segments.call_args.args[0]), moved)
        self.app._video_seek(11.5)
        self.app.video_preview.seek.assert_called_with(11.5)
        self.assertEqual(self.app.video_timeline.position, 11.5)
        self.app._video_select(16, 18, None)
        self.assertIsNone(self.app.video_selected)
        self.assertEqual(float(self.app.video_start.get()), 16)
        self.assertEqual(float(self.app.video_end.get()), 18)
        self.assertEqual(self.app.video_segments, moved)

    def test_video_export_uses_a_snapshot_and_ignores_unapplied_draft_edits(self):
        source = self._load_video()
        self._add_segment(2, 5, speed=2)
        expected = tuple(self.app.video_segments)
        self.app.video_speed.set("unapplied invalid draft")
        result = self.destination / "video_000000000006.mp4"
        with patch.object(self.app, "_job") as job:
            self.app.export_video()
        work = job.call_args.args[0]
        self.app.video_segments.clear()
        with patch.object(self.tool, "edit_video_segments", return_value=(result, "Finished")) as render:
            work()
        self.assertEqual(render.call_args.args, (str(source), expected))

    def test_video_export_requires_an_applied_segment_before_starting_a_job(self):
        self._load_video()
        with patch.object(self.app, "_job") as job:
            with self.assertRaises(ValueError):
                self.app.export_video()
        job.assert_not_called()

    def test_changing_video_path_discards_the_old_plan_and_unloads_preview(self):
        source = self._load_video()
        self._add_segment(2, 5)
        self.app.video_preview.unload.reset_mock()
        self.app.video_path.set(str(self.folder / "another video.mp4"))
        self.app.video_preview.unload.assert_called_once()
        self.assertEqual(self.app.video_duration, 0)
        self.assertEqual(self.app.video_segments, [])
        self.assertIsNone(self.app.video_selected)
        self.assertEqual(self.app.video_timeline.duration, 0)
        self.assertEqual(self.app.video_segment_list.get_children(), ())
        self.assertIn("disabled", self.app.video_play_button.state())
        self.assertEqual(source.read_bytes(), b"input video")

    def test_busy_export_blocks_plan_changes_and_opening_another_reader(self):
        self._load_video()
        self._add_segment(2, 5)
        expected = list(self.app.video_segments)
        self.app.busy = True
        try:
            for action in (self.app.add_video_segment, self.app.apply_video_segment,
                           self.app.delete_video_segment, self.app.clear_video_segments,
                           self.app.inspect_video, self.app.toggle_video_play):
                with self.subTest(action=action.__name__), self.assertRaises(RuntimeError):
                    action()
                self.assertEqual(self.app.video_segments, expected)
        finally:
            self.app.busy = False
        self.app.video_preview.open.assert_not_called()

    def test_generated_video_history_can_save_open_and_delete_without_touching_input(self):
        video = self.destination / "video_000000000006.mp4"
        video.write_bytes(b"generated MP4")
        source = self.destination / "input.mp4"
        source.write_bytes(b"original MP4")
        other = self.destination / "speech_000000000007.mp4"
        other.write_bytes(b"not a generated video")
        self.app.refresh_history()
        self.assertIn(video, self._paths())
        self.assertNotIn(source, self._paths())
        self.assertNotIn(other, self._paths())
        self.app.history_kind.set("Video")
        self.assertEqual(self._paths(), {video})
        self._select(video)
        with patch.object(self.app, "_open_file") as open_file, patch.object(self.app, "_job") as job:
            self.app.play_selected()
        open_file.assert_called_once_with(video)
        job.assert_not_called()
        destination = self.folder / "saved copy.mp4"
        with patch("apps.v3turbo_desktop.filedialog.asksaveasfilename", return_value=str(destination)):
            self.app.save_selected()
        self.assertEqual(destination.read_bytes(), video.read_bytes())
        with patch.object(self.app, "_open_folder") as open_folder:
            self.app.open_selected_folder()
        open_folder.assert_called_once_with(self.destination)
        with patch("apps.v3turbo_desktop.messagebox.askyesno", return_value=True):
            self.app.delete_selected_files()
        self.assertFalse(video.exists())
        self.assertTrue(source.exists())
        self.assertTrue(other.exists())
        self.assertTrue(destination.exists())

    def test_directory_choice_persists_and_restart_loads_history_from_both_folders(self):
        second = self.folder / "second destination"
        with patch("apps.v3turbo_desktop.filedialog.askdirectory", return_value=str(second)):
            self.app.choose_output_dir()
        self.assertEqual(self.tool.output_dir, second)
        self.assertEqual(self.app.output_location.get(), str(second))
        self.assertEqual(self.app.output_dir_var.get(), str(second))
        new_audio = self.tool._path("speech", ".wav")
        sf.write(new_audio, [.0, .2, .0], 48000)
        self.app._result(new_audio)
        self.assertIn(new_audio, self._paths())
        self._dispose_app()
        self.root = tk.Tk()
        self.root.withdraw()
        self.tool = TurboTool(settings_path=self.settings)
        self.app = DesktopApp(self.root, self.tool)
        self.root.update_idletasks()
        self.assertEqual(self.tool.output_dir, second)
        self.assertEqual(self.tool.voices_path, self.library)
        self.assertEqual(self.app.output_location.get(), str(second))
        self.assertTrue({self.speech, self.paired_audio, self.subtitle, new_audio} <= self._paths())

    def test_search_kind_filters_and_refresh_preserve_selected_paths(self):
        for kind in {record.kind for record in self.app.history_records}:
            with self.subTest(kind=kind):
                self.app.history_kind.set(kind)
                self.assertEqual(self._paths(), {record.path for record in self.app.history_records
                                                 if record.kind == kind})
        self.app.history_kind.set("Tất cả")
        self.app.history_query.set("SUBTITLED")
        self.assertEqual(self._paths(), {self.paired_audio, self.subtitle})
        self._select(self.paired_audio)
        self.app.refresh_history()
        self.assertEqual([self.app.audio_files[item] for item in self.app.history.selection()],
                         [self.paired_audio])
        self.assertEqual(self.app.history_detail.get(), str(self.paired_audio))
        self.app.history_query.set("missing result")
        self.assertEqual(self._paths(), set())
        self.assertTrue(self.app.history_count.get().startswith("0 / 6"))

    def test_new_result_is_selected_and_visible_with_prior_filters_active(self):
        self.app.history_kind.set("Audio")
        self.app.history_query.set("no matches")
        item = self.app._result(self.subtitle)
        self.root.update_idletasks()
        self.assertEqual(self.app.history_kind.get(), "Tất cả")
        self.assertEqual(self.app.history_query.get(), "")
        self.assertEqual(self.app.audio_files[item], self.subtitle)
        self.assertEqual(self.app.history.selection(), (item,))
        self.assertEqual(self.app.book.select(), str(self.app.pages["Kết quả / Log"]))

    def test_cancel_delete_and_confirmed_multiple_delete_with_paired_subtitle(self):
        self._select(self.paired_audio, self.archive)
        with patch("apps.v3turbo_desktop.messagebox.askyesno", return_value=False):
            self.app.delete_selected_files()
        self.assertTrue(all(path.exists() for path in (self.paired_audio, self.subtitle, self.archive)))
        with patch("apps.v3turbo_desktop.messagebox.askyesno", return_value=True) as confirm:
            self.app.delete_selected_files()
        self.assertIn("3 file", confirm.call_args.args[1])
        self.assertTrue(all(not path.exists() for path in (self.paired_audio, self.subtitle, self.archive)))
        self.assertEqual(self._paths(), {self.speech, self.voice_export, self.reference})
        self.assertTrue(self.library.exists())
        self.assertTrue(self.input_audio.exists())

    def test_delete_can_preserve_the_associated_subtitle(self):
        self._select(self.paired_audio)
        self.app.delete_with_subtitles.set(False)
        with patch("apps.v3turbo_desktop.messagebox.askyesno", return_value=True):
            self.app.delete_selected_files()
        self.assertFalse(self.paired_audio.exists())
        self.assertTrue(self.subtitle.exists())
        self.assertIn(self.subtitle, self._paths())

    def test_busy_and_recording_block_destination_changes_and_deletion(self):
        self._select(self.speech)
        self.app.output_dir_var.set(str(self.folder / "not applied"))
        for busy, recorder in ((True, None), (False, object())):
            with self.subTest(busy=busy, recorder=recorder is not None):
                self.app.busy, self.app.recorder = busy, recorder
                with patch("apps.v3turbo_desktop.messagebox.askyesno") as confirm:
                    with self.assertRaises(RuntimeError):
                        self.app.apply_output_dir()
                    with self.assertRaises(RuntimeError):
                        self.app.delete_selected_files()
                    confirm.assert_not_called()
                self.assertEqual(self.tool.output_dir, self.destination)
                self.assertTrue(self.speech.exists())
        self.app.busy, self.app.recorder = False, None

    def test_video_viewer_playback_and_timeline_fit_the_tab_when_resized(self):
        self.root.attributes("-alpha", 0.0)
        self.root.deiconify()
        self._wait_for_window()
        self.app.nav_buttons["Video"].invoke()
        self._wait_for_page("Video")
        tab = self.app.pages["Video"]
        widgets = (self.app.video_preview.master, self.app.video_preview,
                   self.app.video_preview.canvas, self.app.video_timeline,
                   self.app.video_play_button)
        for width, height in ((980, 620), (1280, 850)):
            self.root.geometry(f"{width}x{height}")
            self._pump_until(lambda: self.root.winfo_width() == width and self.root.winfo_height() == height)
            left, top = tab.winfo_rootx(), tab.winfo_rooty()
            right, bottom = left + tab.winfo_width(), top + tab.winfo_height()
            for widget in widgets:
                with self.subTest(size=(width, height), widget=type(widget).__name__):
                    self._pump_until(lambda: widget.winfo_ismapped() and widget.winfo_viewable(), widget)
                    self.assertTrue(widget.winfo_ismapped(), self._layout_diagnostics(widget))
                    self.assertGreater(widget.winfo_width(), 5)
                    self.assertGreater(widget.winfo_height(), 5)
                    self.assertGreaterEqual(widget.winfo_rootx(), left)
                    self.assertGreaterEqual(widget.winfo_rooty(), top)
                    self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(), right)
                    self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(), bottom)
            self.assertGreaterEqual(self.app.video_preview.canvas.winfo_height(), 50)
            self.assertGreaterEqual(self.app.video_timeline.winfo_height(), 70)

    def test_navigation_and_primary_actions_remain_accessible_when_resized(self):
        # Map an invisible window so Tk computes real geometry without showing UI.
        self.root.attributes("-alpha", 0.0)
        self.root.deiconify()
        self._wait_for_window()
        for width, height in ((980, 620), (1280, 850)):
            self.root.geometry(f"{width}x{height}")
            self._pump_until(lambda: self.root.winfo_width() == width and self.root.winfo_height() == height)
            for title, button in self.app.nav_buttons.items():
                with self.subTest(size=(width, height), page=title):
                    button.invoke()
                    self._wait_for_page(title)
                    self.assertEqual(self.app.book.select(), str(self.app.pages[title]))
                    self.assertIn("selected", button.state())
                    self.assertGreater(button.winfo_height(), 5)
                    self.assertLessEqual(button.winfo_y() + button.winfo_height(),
                                         button.master.winfo_height())
                    if title == "Kết quả / Log":
                        self.assertGreaterEqual(self.app.history.winfo_height(), 80)
                        delete = self.app.delete_files_button
                        self._pump_until(lambda: delete.winfo_ismapped() and delete.winfo_viewable(), delete)
                        self.assertTrue(delete.winfo_ismapped(), self._layout_diagnostics(delete))
                        self.assertGreater(delete.winfo_height(), 5)
                        self.assertLessEqual(delete.winfo_rooty() + delete.winfo_height(),
                                             self.app.book.winfo_rooty() + self.app.book.winfo_height())
            for title, action in (("Văn bản", "Tạo audio"), ("Hàng loạt", "Tạo batch và ZIP"),
                                  ("Hội thoại", "Tạo hội thoại"), ("Video", "Xuất video")):
                with self.subTest(size=(width, height), primary_action=action):
                    self.app.nav_buttons[title].invoke()
                    self._wait_for_page(title)
                    generate = next(button for button in self.app.action_buttons
                                    if str(button.cget("text")) == action)
                    self._pump_until(lambda: generate.winfo_ismapped() and generate.winfo_viewable(), generate)
                    self.assertTrue(generate.winfo_ismapped(), self._layout_diagnostics(generate))
                    self.assertGreater(generate.winfo_height(), 5)
                    self.assertLessEqual(generate.winfo_rooty() + generate.winfo_height(),
                                         self.app.book.winfo_rooty() + self.app.book.winfo_height())
            self.app.nav_buttons["Cấu hình"].invoke()
            self._wait_for_page("Cấu hình")
            canvas = next(child for child in self.app.pages["Cấu hình"].winfo_children()
                          if isinstance(child, tk.Canvas))
            self.assertGreater(float(canvas.cget("scrollregion").split()[3]), canvas.winfo_height())
            canvas.yview_moveto(1.0)
            self.root.update()
            self.assertGreater(canvas.yview()[0], 0)


class VideoTimelineTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display is unavailable: {exc}")
        self.root.attributes("-alpha", 0.0)
        self.root.geometry("640x140")
        self.selected = []
        self.changed = []
        self.seeked = []
        self.timeline = VideoTimeline(
            self.root, on_select=lambda *values: self.selected.append(values),
            on_change=lambda *values: self.changed.append(values), on_seek=self.seeked.append,
        )
        self.timeline.pack(fill="x")
        self.root.update()
        self.timeline.set_video(20)

    def tearDown(self):
        if hasattr(self, "root"):
            self.root.destroy()

    def _drag(self, start, end, y=55):
        for sequence, seconds in (("<ButtonPress-1>", start),
                                  ("<B1-Motion>", end), ("<ButtonRelease-1>", end)):
            self.timeline.event_generate(sequence, x=round(self.timeline.time_to_x(seconds)), y=y)
            self.root.update()

    def test_dragging_empty_track_selects_source_times_and_ruler_seeks(self):
        self._drag(3, 7)
        start, end, index = self.selected[-1]
        self.assertAlmostEqual(start, 3, delta=0.02)
        self.assertAlmostEqual(end, 7, delta=0.02)
        self.assertIsNone(index)
        self.assertEqual(self.changed, [])
        self._drag(5, 12, y=15)
        self.assertAlmostEqual(self.seeked[-1], 12, delta=0.02)
        self.assertAlmostEqual(self.timeline.position, 12, delta=0.02)
        self.assertEqual(self.timeline.segments, [])

    def test_dragging_a_segment_stops_at_its_neighbor_and_preserves_speed(self):
        first = SpeedSegment(2, 6, speed=0.8, ramp_seconds=0.5)
        neighbor = SpeedSegment(10, 14, speed=2, ramp_seconds=0.5)
        self.timeline.set_segments([first, neighbor], selected=0)
        self._drag(4, 18)
        segments, selected = self.changed[-1]
        self.assertEqual(selected, 0)
        self.assertAlmostEqual(segments[0].start, 6)
        self.assertAlmostEqual(segments[0].end, 10)
        self.assertEqual(segments[0].speed, 0.8)
        self.assertEqual(segments[0].ramp_seconds, 0.5)
        self.assertEqual(segments[1], neighbor)

    def test_resizing_an_edge_clamps_to_video_and_shortens_ramp(self):
        self.timeline.set_segments([SpeedSegment(2, 6, speed=2, ramp_seconds=1)], selected=0)
        self._drag(2, -5)
        self.assertEqual(self.changed[-1][0][0].start, 0)
        self._drag(6, 0)
        resized = self.changed[-1][0][0]
        self.assertGreater(resized.end, resized.start)
        self.assertLessEqual(resized.ramp_seconds, (resized.end - resized.start) / 2)
        self.assertEqual(resized.speed, 2)

    def test_new_selection_cannot_cross_an_existing_segment_and_disabled_track_is_idle(self):
        segment = SpeedSegment(5, 8)
        self.timeline.set_segments([segment])
        self._drag(2, 12)
        start, end, index = self.selected[-1]
        self.assertAlmostEqual(start, 2, delta=0.02)
        self.assertEqual(end, 5)
        self.assertIsNone(index)
        selected = list(self.selected)
        self.timeline.enabled = False
        self._drag(1, 3)
        self._drag(1, 19, y=15)
        self.assertEqual(self.selected, selected)
        self.assertEqual(self.seeked, [])
        self.assertEqual(self.timeline.segments, [segment])


if __name__ == "__main__":
    unittest.main()
