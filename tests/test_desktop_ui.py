"""Desktop workflow checks without model downloads or audio device access."""
from __future__ import annotations

import atexit
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch
import zipfile

import soundfile as sf

from apps.v3turbo_desktop import DesktopApp
from apps.v3turbo_tool import Sampling, TurboTool


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

    def test_startup_restores_generated_history_and_preserves_speed_controls(self):
        self.assertEqual(len(self.app.pages), 8)
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

    def test_navigation_and_primary_actions_remain_accessible_when_resized(self):
        # Map an invisible window so Tk computes real geometry without showing UI.
        self.root.attributes("-alpha", 0.0)
        self.root.deiconify()
        for width, height in ((980, 620), (1280, 850)):
            self.root.geometry(f"{width}x{height}")
            self.root.update()
            for title, button in self.app.nav_buttons.items():
                with self.subTest(size=(width, height), page=title):
                    button.invoke()
                    self.root.update()
                    self.assertEqual(self.app.book.select(), str(self.app.pages[title]))
                    self.assertIn("selected", button.state())
                    self.assertGreater(button.winfo_height(), 5)
                    self.assertLessEqual(button.winfo_y() + button.winfo_height(),
                                         button.master.winfo_height())
                    if title == "Kết quả / Log":
                        self.assertGreaterEqual(self.app.history.winfo_height(), 80)
                        delete = self.app.delete_files_button
                        self.assertTrue(delete.winfo_ismapped())
                        self.assertGreater(delete.winfo_height(), 5)
                        self.assertLessEqual(delete.winfo_rooty() + delete.winfo_height(),
                                             self.app.book.winfo_rooty() + self.app.book.winfo_height())
            for title, action in (("Văn bản", "Tạo audio"), ("Hàng loạt", "Tạo batch và ZIP"),
                                  ("Hội thoại", "Tạo hội thoại")):
                with self.subTest(size=(width, height), primary_action=action):
                    self.app.nav_buttons[title].invoke()
                    self.root.update()
                    generate = next(button for button in self.app.action_buttons
                                    if str(button.cget("text")) == action)
                    self.assertTrue(generate.winfo_ismapped())
                    self.assertGreater(generate.winfo_height(), 5)
                    self.assertLessEqual(generate.winfo_rooty() + generate.winfo_height(),
                                         self.app.book.winfo_rooty() + self.app.book.winfo_height())
            self.app.nav_buttons["Cấu hình"].invoke()
            self.root.update()
            canvas = next(child for child in self.app.pages["Cấu hình"].winfo_children()
                          if isinstance(child, tk.Canvas))
            self.assertGreater(float(canvas.cget("scrollregion").split()[3]), canvas.winfo_height())
            canvas.yview_moveto(1.0)
            self.root.update()
            self.assertGreater(canvas.yview()[0], 0)


if __name__ == "__main__":
    unittest.main()
