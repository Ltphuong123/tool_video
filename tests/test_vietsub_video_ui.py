"""File-only vietsub workflow, always-visible queue and result actions."""
from pathlib import Path
from unittest.mock import Mock, patch
import unittest

from apps.vietsub_video import VietsubProgress, VietsubVideo
from apps.v3turbo_tool import Cancelled
import test_desktop_ui as fixtures


class AutoVietsubUITests(unittest.TestCase):
    setUp = fixtures.DesktopUITests.setUp
    tearDown = fixtures.DesktopUITests.tearDown
    _dispose_app = fixtures.DesktopUITests._dispose_app
    _pump_until = fixtures.DesktopUITests._pump_until
    _layout_diagnostics = fixtures.DesktopUITests._layout_diagnostics
    _paths = fixtures.DesktopUITests._paths

    def inputs(self):
        panel=self.app.vietsub_panel
        video,srt=self.folder/'source.mp4',self.folder/'source.srt'
        video.write_bytes(b'source video')
        srt.write_text('1\n00:00:00,000 --> 00:00:01,000\nHello\n',encoding='utf-8')
        panel.source_video.set(str(video))
        panel.source_srt.set(str(srt))
        return panel,video,srt

    def result(self):
        video=self.destination/'video_000000000078.mp4'
        srt=video.with_suffix('.srt')
        audio=video.with_suffix('.wav')
        video.write_bytes(b'voiced video')
        audio.write_bytes(b'voice')
        srt.write_text('1\n00:00:00,000 --> 00:00:01,000\nHello\n',encoding='utf-8')
        return VietsubVideo(video,srt,audio,1,1,.5)

    def test_single_job_captures_own_voice_speed_and_returns_three_visible_results(self):
        panel,video,srt=self.inputs()
        panel.voice.set('Selected voice')
        panel.speed.set('1.25')
        self.app.book.select(self.app.pages['Tự động vietsub video'])
        def run(tool,source_video,source_srt,voice,sampling,**kwargs):
            self.assertEqual((source_video,source_srt),(video,srt))
            self.assertEqual(voice,'Selected voice')
            self.assertEqual(sampling.speed,1.25)
            kwargs['progress'](VietsubProgress(3,'Reading','1/1',1))
            return self.result()
        with patch('apps.vietsub_video._create_vietsub_video',side_effect=run) as work:
            self.app.start_vietsub()
            self.assertTrue(panel.stop_button.instate(['!disabled']))
            self.assertTrue(panel.speed_box.instate(['disabled']))
            self.assertTrue(self._pump_until(lambda:not self.app.busy))
            work.assert_called_once()
        self.assertTrue(Path(panel.output_video.get()).is_file())
        self.assertTrue(Path(panel.output_srt.get()).is_file())
        self.assertIn('Hoàn tất',panel.stage.get())
        self.assertTrue(Path(panel.output_audio.get()).is_file())
        self.assertEqual(len(panel.result_controls),3)
        self.assertEqual(self.app.book.select(),str(self.app.pages['Tự động vietsub video']))
        self.assertFalse(hasattr(panel, 'previews'))
        self.assertFalse(self.app.vietsub_running)

    def test_results_open_files_and_folder_in_default_app_without_video_decoders(self):
        panel,video,srt=self.inputs()
        result=self.result()
        panel.finish(result)
        with patch.object(panel, '_open_file') as opening:
            for button, variable in panel.result_controls:
                button.invoke()
            panel.folder_button.invoke()
            self.assertEqual([call.args[0] for call in opening.call_args_list],
                             [result.video, result.subtitles, result.audio, result.video.parent])
        self.assertFalse(hasattr(panel, 'previews'))
        self.assertFalse(hasattr(panel, 'views'))

    def test_progress_error_and_stop_do_not_show_internal_files(self):
        panel,_,_=self.inputs()
        def fail(*args,**kwargs):
            kwargs['progress'](VietsubProgress(4,'Căn tốc độ',fraction=.5))
            self.app.queue_cancel.set()
            raise Cancelled('Đã dừng tác vụ')
        with patch('apps.vietsub_video._create_vietsub_video',side_effect=fail):
            self.app.start_vietsub()
            self.assertTrue(self._pump_until(lambda:not self.app.busy))
        self.assertTrue(panel.stage.get().startswith('Đã dừng'))
        self.assertEqual(panel.output_video.get(),'')
        self.assertEqual(panel.output_srt.get(),'')
        self.assertEqual(panel.output_audio.get(),'')
        self.assertTrue(panel.start_button.instate(['!disabled']))

    def test_invalid_speed_never_starts_a_job(self):
        panel,_,_=self.inputs()
        with patch.object(self.app,'_job') as job:
            for speed in ('0','nan','3'):
                panel.speed.set(speed)
                with self.assertRaises(ValueError):
                    self.app.start_vietsub()
            job.assert_not_called()

    def test_changing_inputs_clears_stale_results_and_disables_output_actions(self):
        panel,_,_=self.inputs()
        panel.finish(self.result())
        panel.source_srt.set(str(self.folder/'other.srt'))
        self.assertEqual(panel.output_video.get(),'')
        self.assertEqual(panel.output_srt.get(),'')
        self.assertEqual(panel.output_audio.get(), '')
        self.assertTrue(panel.folder_button.instate(['disabled']))
        self.assertTrue(all(button.instate(['disabled']) for button, _ in panel.result_controls))
        self.assertEqual(panel.stage.get(),'Sẵn sàng')

    def test_minimum_window_keeps_queue_actions_progress_and_results_visible_together(self):
        self.root.attributes('-alpha', 0.0)
        self.root.geometry('980x620+0+0')
        self.root.deiconify()
        self.app.book.select(self.app.pages['Tự động vietsub video'])
        panel = self.app.vietsub_panel
        self.assertTrue(self._pump_until(lambda: panel.queue.tree.winfo_viewable()
                                        and panel.queue.tree.winfo_height() >= 130), self._layout_diagnostics(panel))
        self.assertEqual(self.app.storage_bar.winfo_manager(), '')
        self.assertEqual(self.app.footer_bar.winfo_manager(), '')
        self.assertEqual(self.app.header_stop.winfo_manager(), '')
        for widget in (panel.inputs, panel.queue.tree, panel.start_button, panel.stop_button,
                       panel.progress, panel.results, panel.folder_button):
            self.assertTrue(widget.winfo_viewable(), self._layout_diagnostics(widget))
            self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(),
                                 self.root.winfo_rooty() + self.root.winfo_height())
            self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(),
                                 self.root.winfo_rootx() + self.root.winfo_width())
        self.app.book.select(self.app.pages['Cấu hình'])
        self.app._page_changed()
        self.assertEqual(self.app.storage_bar.winfo_manager(), 'pack')
        self.assertEqual(self.app.footer_bar.winfo_manager(), 'pack')
        self.assertEqual(self.app.header_stop.winfo_manager(), 'pack')

    def test_selecting_video_auto_pairs_same_named_srt_and_clears_mismatched_srt(self):
        panel, video, srt = self.inputs()
        with patch('apps.vietsub_panel.filedialog.askopenfilename', return_value=str(video)):
            panel._choose('video')
        self.assertEqual(panel.source_srt.get(), str(srt))
        other = self.folder / 'other.mp4'
        other.write_bytes(b'video')
        with patch('apps.vietsub_panel.filedialog.askopenfilename', return_value=str(other)):
            panel._choose('video')
        self.assertEqual(panel.source_srt.get(), '')

    def test_queue_snapshots_settings_continues_after_error_and_browses_results(self):
        panel, video, srt = self.inputs()
        panel.voice.set('First')
        panel.speed.set('1.2')
        first = panel.add_current()
        self.assertEqual((panel.source_video.get(), panel.source_srt.get()), ('', ''))
        self.assertEqual(panel.voice.get(), panel._default_voice)
        self.assertEqual(panel.speed.get(), '1.0')
        self.assertTrue(panel.start_button.instate(['!disabled']))
        panel.source_video.set(str(video))
        panel.source_srt.set(str(srt))
        panel.voice.set('Second')
        panel.speed.set('.8')
        second = panel.add_current()
        panel.source_video.set(str(video))
        panel.source_srt.set(str(srt))
        panel.voice.set('Third')
        panel.speed.set('1')
        third = panel.add_current()
        # Changing inputs/settings afterwards must not change queued snapshots.
        panel.speed.set('nan')
        self.assertEqual(first.sampling.speed, 1.2)
        self.assertEqual(second.sampling.speed, .8)
        calls = []
        def run(tool, source_video, source_srt, voice, sampling, **kwargs):
            calls.append((voice, sampling.speed))
            if voice == 'Second':
                raise ValueError('bad item')
            return self.result()
        with patch('apps.vietsub_video._create_vietsub_video', side_effect=run):
            self.app.start_vietsub()
            self.assertTrue(self._pump_until(lambda: not self.app.busy))
        self.assertEqual(calls, [('First', 1.2), ('Second', .8), ('Third', 1)])
        self.assertEqual([panel.queue.rows[task.identifier]['state'] for task in (first, second, third)],
                         ['done', 'error', 'done'])
        self.assertEqual(panel.queue.pending(), (second,))
        self.assertIn('2 video xong, 1 lỗi', panel.stage.get())
        panel.queue.tree.selection_set(first.identifier)
        panel.queue._selected()
        self.assertTrue(Path(panel.output_audio.get()).is_file())
        output = Path(panel.output_video.get())
        panel.queue.clear_done()
        self.assertEqual(panel.queue.tasks, [second])
        self.assertTrue(output.is_file())

    def test_queue_stop_and_resume_does_not_rerun_completed_items(self):
        panel, video, srt = self.inputs()
        tasks = []
        for _ in range(3):
            panel.source_video.set(str(video))
            panel.source_srt.set(str(srt))
            tasks.append(panel.add_current())
        calls = []
        def run(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                self.app.queue_cancel.set()
                self.tool.stop()
                raise Cancelled('Stopped')
            return self.result()
        with patch('apps.vietsub_video._create_vietsub_video', side_effect=run):
            self.app.start_vietsub()
            self.assertTrue(self._pump_until(lambda: not self.app.busy))
        self.assertEqual(len(calls), 2)
        self.assertEqual([panel.queue.rows[task.identifier]['state'] for task in tasks],
                         ['done', 'cancelled', 'pending'])
        with patch('apps.vietsub_video._create_vietsub_video', side_effect=lambda *a, **kw: self.result()) as work:
            self.app.start_vietsub()
            self.assertTrue(self._pump_until(lambda: not self.app.busy))
            self.assertEqual(work.call_count, 2)
        self.assertFalse(panel.queue.pending())
        self.assertTrue(all(panel.queue.rows[task.identifier]['state'] == 'done' for task in tasks))

    def test_batch_video_picker_matches_srt_names_and_preserves_manual_pairing(self):
        panel, video, srt = self.inputs()
        other = self.folder / 'without_subtitles.mp4'
        other.write_bytes(b'video')
        panel.voice.set('Batch voice')
        panel.speed.set('1.25')
        with patch('apps.vietsub_panel.filedialog.askopenfilenames', return_value=[str(video), str(other)]):
            with self.assertRaisesRegex(ValueError, 'without_subtitles'):
                panel.add_videos()
        self.assertEqual(len(panel.queue.tasks), 1)
        self.assertEqual(panel.queue.tasks[0].subtitles, srt)
        self.assertEqual(panel.queue.tasks[0].voice, 'Batch voice')
        self.assertEqual(panel.queue.tasks[0].sampling.speed, 1.25)
        self.assertEqual((panel.source_video.get(), panel.source_srt.get()), ('', ''))
        self.assertEqual(panel.voice.get(), panel._default_voice)
        self.assertEqual(panel.speed.get(), '1.0')
        panel.source_video.set(str(other))
        panel.source_srt.set(str(srt))
        task = panel.add_current()
        self.assertEqual((task.video, task.subtitles), (other, srt))

    def test_queue_buttons_follow_selection_completion_and_busy_state(self):
        panel, _, _ = self.inputs()
        remove, clear = panel.queue.buttons
        self.assertTrue(remove.instate(['disabled']))
        self.assertTrue(clear.instate(['disabled']))
        task = panel.add_current()
        panel.queue.tree.selection_set(task.identifier)
        panel.queue._selected()
        self.assertTrue(remove.instate(['!disabled']))
        self.assertTrue(clear.instate(['disabled']))
        panel.queue.update_task(task.identifier, 'done', self.result())
        self.assertTrue(clear.instate(['!disabled']))
        panel.set_busy(True, True)
        self.assertTrue(remove.instate(['disabled']))
        self.assertTrue(clear.instate(['disabled']))
        panel.queue.remove_selected()
        self.assertEqual(panel.queue.tasks, [task])
        panel.set_busy(False)
        panel.queue.clear_done()
        self.assertTrue(remove.instate(['disabled']))
        self.assertTrue(clear.instate(['disabled']))
        self.assertEqual(panel.queue.tasks, [])

    def test_duplicate_task_is_rejected_without_mutating_queue(self):
        panel, _, _ = self.inputs()
        task = panel.add_current()
        with self.assertRaises(ValueError):
            panel.queue.add(task)
        self.assertEqual(panel.queue.tasks, [task])
        self.assertEqual(tuple(panel.queue.rows), (task.identifier,))
        self.assertEqual(panel.queue.tree.get_children(), (task.identifier,))

    def test_repeated_progress_does_not_rescan_queue_and_percentage_finishes(self):
        panel, _, _ = self.inputs()
        task = panel.add_current()
        first = VietsubProgress(3, 'Tạo giọng đọc', '1/3', .3)
        second = VietsubProgress(3, 'Tạo giọng đọc', '2/3', .6)
        with patch.object(panel.queue, 'refresh') as refresh:
            panel.queue.update_task(task.identifier, 'progress', first)
            panel.queue.update_task(task.identifier, 'progress', second)
            refresh.assert_not_called()
        self.assertIn('Bước 3/5', panel.queue.rows[task.identifier]['message'])
        panel.set_stage(second)
        self.assertEqual(panel.percent.get(), '60%')
        self.assertEqual(float(panel.progress['value']), 60)
        panel.set_stage(VietsubProgress(4, 'Căn tốc độ'))
        self.assertEqual(str(panel.progress['mode']), 'indeterminate')
        panel.finish(self.result())
        self.assertEqual(panel.percent.get(), '100%')
        self.assertEqual(str(panel.progress['mode']), 'determinate')
        panel.fail('Cancelled', cancelled=True)
        self.assertEqual(panel.percent.get(), '—')
        panel.set_stage(None)
        self.assertEqual(panel.percent.get(), '0%')


if __name__=='__main__':
    unittest.main()
