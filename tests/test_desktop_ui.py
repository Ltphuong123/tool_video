"""Desktop workflow checks without model downloads or audio device access."""
from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
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
        try:
            self.app = DesktopApp(self.root, self.tool, autoload=False)
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
            self.root.tk.call("after", "cancel", identifier)
        self.app.busy = False
        self.app.close()

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


    def test_text_and_srt_hide_storage_and_reference(self):
        self.assertFalse(hasattr(self.app, "reference"))
        self.assertIs(self.app.speech_speed, self.app.vars["speed"])
        for page in ("Văn bản", "SRT"):
            self.app.book.select(self.app.pages[page])
            self.app._page_changed()
            self.assertEqual(self.app.storage_bar.winfo_manager(), "")
        self.app.book.select(self.app.pages["Cấu hình"])
        self.app._page_changed()
        self.assertEqual(self.app.storage_bar.winfo_manager(), "pack")

    def test_many_texts_and_file_dialog_queue_inputs(self):
        self.app.speech.delete("1.0", "end")
        self.app.speech.insert("1.0", "One paragraph.\n\nMore words.\n===\nTwo.\n===\n")
        self.app.add_text_queue()
        tasks = self.app.queue_panels["Văn bản"].tasks
        self.assertEqual([task.text for task in tasks], ["One paragraph.\n\nMore words.\n===\nTwo.\n==="])
        paths = (str(self.folder / "a.txt"), str(self.folder / "b.txt"))
        with patch("apps.v3turbo_desktop.filedialog.askopenfilenames", return_value=paths):
            self.app.choose_queue_files("Văn bản")
        self.assertEqual([task.source for task in tasks[-2:]], [Path(path) for path in paths])
        with patch("apps.v3turbo_desktop.filedialog.askopenfilenames", return_value=(str(self.subtitle),)):
            self.app.choose_queue_files("SRT")
        self.assertEqual(self.app.queue_panels["SRT"].tasks[0].source, self.subtitle)

    def test_text_queue_generates_separate_files_and_stays_in_tab(self):
        from test_v3turbo_desktop import FakeTurbo
        from apps.desktop_queue import QueueTask
        self.tool._factory = FakeTurbo
        self.tool.load()
        self.app.refresh_voices()
        self.app.speech_speed.set("1.0")
        panel = self.app.queue_panels["Văn bản"]
        for label, text in [("one", "Hello"), ("bad", ""), ("two", "World")]:
            panel.add(QueueTask(label, text=text))
        self.app.start_queue("Văn bản")
        self.assertTrue(self._pump_until(lambda: not self.app.busy))
        states = [panel.rows[task.identifier]["state"] for task in panel.tasks]
        self.assertEqual(states, ["done", "error", "done"])
        self.assertEqual(len(list(self.destination.glob("speech_*.wav"))), 4)
        self.assertEqual(self.app.book.select(), str(self.app.pages["Văn bản"]))
        self.assertEqual(len(list(self.destination.glob("speech_subtitled_*.srt"))), 3)
        self.assertEqual(len(panel.pending()), 1)
        self.assertEqual(len(self.app.inline_histories["Văn bản"][1]), 4)

    def test_srt_queue_progress_speed_and_inline_history(self):
        from apps.desktop_queue import QueueTask
        panel = self.app.queue_panels["SRT"]
        panel.add(QueueTask("first.srt", source=self.subtitle))
        panel.add(QueueTask("second.srt", source=self.subtitle))
        self.tool.tts = Mock()
        self.app.srt_min_speed.set("0.8")
        def fake_srt(path, voice, sampling, keep_timing, fmt, progress, **kwargs):
            self.assertEqual(kwargs["min_speed"], 0.8)
            self.assertTrue(kwargs["fit_to_timing"])
            progress("SRT 1/2 câu")
            result = self.tool._save([0, 0.1, 0], label="srt")
            progress("SRT 2/2 câu")
            return result, "done"
        with patch.object(self.tool, "srt", side_effect=fake_srt), \
             patch.object(self.app, "refresh_voices"):
            self.app.start_queue("SRT")
            self.assertTrue(self._pump_until(lambda: not self.app.busy))
        self.tool.tts = None
        self.assertEqual(len(self.app.inline_histories["SRT"][1]), 2)
        for task in panel.tasks:
            row = panel.rows[task.identifier]
            self.assertEqual(row["state"], "done")
            self.assertEqual(float(row["progress"]), 100)

    def test_queue_rows_progress_remove_and_busy_guard(self):
        from apps.desktop_queue import QueueTask
        panel = self.app.queue_panels["SRT"]
        task = QueueTask("test.srt", source=self.subtitle)
        panel.add(task)
        panel.update_task(task.identifier, "running", None)
        panel.update_task(task.identifier, "progress", "SRT 3/10 câu")
        self.assertEqual(float(panel.rows[task.identifier]["progress"]), 30)
        panel.update_task(task.identifier, "cancelled", "stop")
        self.assertEqual(panel.pending(), (task,))
        panel.rows[task.identifier]["selected"].set(True)
        self.app.busy = True
        with self.assertRaises(RuntimeError):
            self.app.remove_queue_items("SRT")
        self.app.busy = False
        self.app.remove_queue_items("SRT")
        self.assertEqual(panel.tasks, [])

    def test_removed_tabs_and_voice_refresh_are_pruned(self):
        self.assertEqual(set(self.app.pages), {"Văn bản", "SRT", "Tự động vietsub video", "Kết quả / Log", "Cấu hình"})
        self.assertFalse(hasattr(self.app, "video_preview"))
        self.assertFalse(hasattr(self.app, "training_job"))
        self.assertFalse(hasattr(self.app, "record_microphone"))
        with patch.object(self.tool, "voice_names", wraps=self.tool.voice_names) as voices:
            self.app.refresh_voices()
            self.app.refresh_voices()
            self.assertEqual(voices.call_count, 1)

    def _restart_with_autoload(self):
        self._dispose_app()
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = DesktopApp(self.root, self.tool)

    def _fake_loaded_model(self):
        self.tool.tts = SimpleNamespace(
            backend="pytorch", _default_voice="Test voice", close=Mock(),
            list_preset_voices=Mock(return_value=[("test", "Test voice")]))
        return "Model ready"

    def test_startup_loads_once_in_background_and_keeps_ui_responsive(self):
        release = threading.Event()
        self.addCleanup(release.set)
        caller_threads = []

        def load(**kwargs):
            caller_threads.append(threading.get_ident())
            release.wait(3)
            return self._fake_loaded_model()

        with patch.object(self.tool, "load", side_effect=load) as loading:
            self._restart_with_autoload()
            self.assertTrue(self._pump_until(lambda: bool(caller_threads)))
            self.assertTrue(self.app.busy)
            self.assertNotEqual(caller_threads[0], threading.get_ident())
            self.assertIn("Đang tải", self.app.model_state.get())
            self.assertTrue(all("disabled" in button.state() for button in self.app.action_buttons))
            ui_tick = []
            self.root.after(0, lambda: ui_tick.append(True))
            self.assertTrue(self._pump_until(lambda: bool(ui_tick)))
            release.set()
            self.assertTrue(self._pump_until(lambda: not self.app.busy))
            loading.assert_called_once()
            self.assertEqual(loading.call_args.kwargs["backbone_repo"], self.app.vars["backbone_repo"].get())
            self.assertEqual(self.app.voice.get(), "Test voice")
            self.assertIn("PYTORCH", self.app.model_state.get())
            self.assertTrue(all("disabled" not in button.state() for button in self.app.action_buttons))

    def test_startup_load_failure_restores_controls_and_allows_retry(self):
        with patch.object(self.tool, "load", side_effect=RuntimeError("Cannot load model")) as loading, \
             patch("apps.v3turbo_desktop.messagebox.showerror") as error:
            self._restart_with_autoload()
            self.assertTrue(self._pump_until(lambda: loading.called and not self.app.busy))
            error.assert_called_once()
            self.assertEqual(self.app.status.get(), "Cannot load model")
            self.assertEqual(self.app.model_state.get(), "Model chưa tải")
            loading.side_effect = lambda **kwargs: self._fake_loaded_model()
            self.app.load_model()
            self.assertTrue(self._pump_until(lambda: not self.app.busy))
            self.assertEqual(loading.call_count, 2)
            self.assertEqual(self.app.status.get(), "Model ready")

    def test_startup_reuses_an_already_loaded_model(self):
        self._restart_with_autoload()
        self._fake_loaded_model()
        with patch.object(self.tool, "load") as loading:
            self.root.update()
            loading.assert_not_called()
            self.assertIn("PYTORCH", self.app.model_state.get())
            self.assertFalse(self.app.busy)

    def test_close_before_startup_cancels_model_loading(self):
        with patch.object(self.tool, "load") as loading:
            self._restart_with_autoload()
            self.assertIsNotNone(self.app._autoload_handle)
            self.app.close()
            loading.assert_not_called()
            self.assertIsNone(self.app._autoload_handle)

    def test_check_mode_can_build_ui_without_loading_model(self):
        with patch.object(self.tool, "load") as loading:
            self.root.update()
            loading.assert_not_called()
            self.assertIsNone(self.app._autoload_handle)

    def test_close_during_model_loading_releases_model_after_worker_finishes(self):
        release = threading.Event()
        self.addCleanup(release.set)
        closed_model = []

        def load(**kwargs):
            release.wait(3)
            result = self._fake_loaded_model()
            closed_model.append(self.tool.tts.close)
            return result

        with patch.object(self.tool, "load", side_effect=load):
            self._restart_with_autoload()
            self.assertTrue(self._pump_until(lambda: self.app.busy))
            self.app.close()
            release.set()
            self.app._worker.join(3)
            self.assertFalse(self.app._worker.is_alive())
            self.assertIsNone(self.tool.tts)
            closed_model[0].assert_called_once()
            self.assertTrue(self.app.events.empty())

    def test_close_after_worker_returns_before_ui_drains_releases_model(self):
        with patch.object(self.tool, "load", side_effect=lambda **kwargs: self._fake_loaded_model()):
            self.app.load_model()
            self.app._worker.join(3)
            self.assertTrue(self.app.busy)
            model_close = self.tool.tts.close
            self.app.close()
            model_close.assert_called_once()
            self.assertIsNone(self.tool.tts)

    def test_progress_flood_finishes_queue_without_repeated_history_scans(self):
        from apps.desktop_queue import QueueTask
        panel = self.app.queue_panels["SRT"]
        task = QueueTask("source.srt", source=self.subtitle)
        panel.add(task)
        self.app.busy = True
        self.app._post("queue", ("SRT", task.identifier, "running", None))
        for index in range(1000):
            self.app._post("queue", ("SRT", task.identifier, "progress", f"SRT {index}/1000 câu"))
        self.app._post("queue", ("SRT", task.identifier, "done", (self.paired_audio, "done")))
        self.app._post("done", (None, lambda _: self.app.refresh_history(), None))
        with patch.object(self.tool, "list_outputs", wraps=self.tool.list_outputs) as scan:
            self.assertTrue(self._pump_until(lambda: not self.app.busy))
            scan.assert_called_once()
        self.assertEqual(panel.rows[task.identifier]["state"], "done")
        self.assertEqual(panel.rows[task.identifier]["progress"], 100)

    def test_refresh_keeps_inline_selection_and_unchanged_rows(self):
        table, paths = self.app.inline_histories["Văn bản"]
        identifier = next(key for key, path in paths.items() if path == self.paired_audio)
        table.selection_set(identifier)
        self.app.refresh_history()
        self.assertEqual(table.selection(), (identifier,))
        sf.write(self.destination / "speech_000000000018.wav", [0, .1, 0], 48000)
        self.app.refresh_history()
        self.assertEqual([paths[key] for key in table.selection()], [self.paired_audio])


    def test_configuration_has_no_effect_controls_and_keeps_reference_codes(self):
        self.assertNotIn("use_ref_codes", self.app.vars)
        sampling = self.app._sampling()
        self.assertTrue(sampling.use_ref_codes)
        for key in ("bright", "compress", "peak_guard"):
            self.assertNotIn(key, self.app.vars)
            self.assertFalse(hasattr(sampling, key))


    def test_text_page_buttons_are_compact_and_queue_only(self):
        labels = {str(button.cget("text")) for button in self.app.action_buttons}
        for removed in ("Nhập TXT / PDF", "Tạo audio", "Audio + SRT từng câu", "Streaming và nghe", "Tách nhiều văn bản (===)"):
            self.assertNotIn(removed, labels)
        self.assertIn("Thêm văn bản", labels)
        self.assertIn("Thêm TXT", labels)
        self.assertIn("Chạy hàng đợi", labels)

    def test_srt_history_playback_skips_initial_silence(self):
        path = self.destination / "srt_000000000099.wav"
        sf.write(path, [0, 0.1, 0], 48000)
        self.app.refresh_history()
        table, paths = self.app.inline_histories["SRT"]
        selected = next(identifier for identifier, value in paths.items() if value == path)
        table.selection_set(selected)
        player = Mock()
        with patch("apps.v3turbo_desktop.WavePlayer", return_value=player):
            self.app.play_inline_history("SRT")
            self.assertTrue(self._pump_until(lambda: not self.app.busy))
        self.assertEqual(player.play_file.call_args.args, (path,))
        self.assertTrue(player.play_file.call_args.kwargs["skip_initial_silence"])
        self.assertTrue(callable(player.play_file.call_args.kwargs["on_start"]))

    def test_srt_inline_history_opens_and_saves_generated_pair_when_main_history_is_filtered(self):
        from test_v3turbo_desktop import FakeTurbo
        self.tool.tts = FakeTurbo()
        audio, _ = self.tool.srt(self.subtitle)
        paired = audio.with_suffix('.srt')
        self.app.refresh_history()
        self.app.history_kind.set('Phụ đề')
        table, paths = self.app.inline_histories['SRT']
        table.selection_set(next(key for key, value in paths.items() if value == audio))
        folder = self.folder / 'saved_pair'
        folder.mkdir()
        with patch.object(self.app, '_open_file') as opening:
            self.app.open_inline_subtitle('SRT')
            opening.assert_called_once_with(paired)
        with patch('apps.v3turbo_desktop.filedialog.askdirectory', return_value=str(folder)):
            self.app._save_subtitle_pair(self.app._inline_selected('SRT'))
        self.assertEqual((folder / audio.name).read_bytes(), audio.read_bytes())
        self.assertEqual((folder / paired.name).read_bytes(), paired.read_bytes())

    def test_inline_playback_uses_internal_player_without_opening_another_app(self):
        table, paths = self.app.inline_histories["Văn bản"]
        selected = next(identifier for identifier, path in paths.items() if path == self.speech)
        table.selection_set(selected)
        player = Mock()
        with patch("apps.v3turbo_desktop.WavePlayer", return_value=player), \
             patch.object(self.app, "_open_file") as external:
            self.app.play_inline_history("Văn bản")
            self.assertTrue(self._pump_until(lambda: not self.app.busy))
        player.play_file.assert_called_once_with(self.speech)
        external.assert_not_called()
        self.assertIsNone(self.app.player)

    def test_inline_delete_audio_and_paired_srt_refreshes_both_histories(self):
        table, paths = self.app.inline_histories["Văn bản"]
        selected = next(identifier for identifier, path in paths.items() if path == self.paired_audio)
        table.selection_set(selected)
        with patch("apps.v3turbo_desktop.messagebox.askyesno", return_value=False):
            self.app.delete_inline_history("Văn bản")
        self.assertTrue(self.paired_audio.exists())
        with patch("apps.v3turbo_desktop.messagebox.askyesno", return_value=True):
            self.app.delete_inline_history("Văn bản")
        self.assertFalse(self.paired_audio.exists())
        self.assertFalse(self.subtitle.exists())
        self.assertTrue(self.speech.exists())
        self.assertNotIn(self.paired_audio, self._paths())
        self.assertNotIn(self.paired_audio, self.app.inline_histories["Văn bản"][1].values())

    def test_inline_delete_rejects_running_job(self):
        self.app.busy = True
        with self.assertRaises(RuntimeError), patch("apps.v3turbo_desktop.messagebox.askyesno") as dialog:
            self.app.delete_inline_history("Văn bản")
        dialog.assert_not_called()
        self.app.busy = False

    def test_startup_restores_generated_history_and_preserves_speed_controls(self):
        self.assertEqual(len(self.app.pages), 5)
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
        self.assertNotIn("Video", self.app.pages)


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
        self.app = DesktopApp(self.root, self.tool, autoload=False)
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
        for busy, recorder in ((True, None),):
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
            for title in ("Văn bản", "SRT"):
                action = "Chạy hàng đợi"
                with self.subTest(size=(width, height), primary_action=action):
                    self.app.nav_buttons[title].invoke()
                    self._wait_for_page(title)
                    generate = next(button for button in self.app.action_buttons
                                    if str(button.cget("text")) == action
                                    and str(button).startswith(str(self.app.pages[title]) + "."))
                    self._pump_until(lambda: generate.winfo_ismapped() and generate.winfo_viewable(), generate)
                    self.assertTrue(generate.winfo_ismapped(), self._layout_diagnostics(generate))
                    self.assertGreater(generate.winfo_height(), 5)
                    self.assertLessEqual(generate.winfo_rooty() + generate.winfo_height(),
                                         self.app.book.winfo_rooty() + self.app.book.winfo_height())
            self.app.nav_buttons["Cấu hình"].invoke()
            self._wait_for_page("Cấu hình")
            self.app.show_model_options.set(True)
            self.app._toggle_options(self.app.model_options_panel, self.app.show_model_options)
            self.app.show_sampling_options.set(True)
            self.app._toggle_options(self.app.sampling_options_panel, self.app.show_sampling_options)
            self.root.update_idletasks()
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
