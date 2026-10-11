"""Automatic SRT/voice/video processing with real FFmpeg and fake TTS."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos

from apps.v3turbo_tool import Cancelled, Sampling, TurboTool
from apps.srt_speech import parse_srt
from apps.video_fast_export import _ffmpeg_binary
from apps.vietsub_queue import VietsubTask
from test_v3turbo_desktop import FakeTurbo


class AutoVietsubTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.video, self.srt = self.folder/'source.mp4', self.folder/'source.srt'
        subprocess.run([_ffmpeg_binary(), '-hide_banner', '-loglevel', 'error', '-nostdin',
                        '-f', 'lavfi', '-i', 'testsrc2=s=64x48:r=24:d=2',
                        '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2',
                        '-c:v', 'libx264', '-preset', 'ultrafast', '-bf', '2', '-c:a', 'aac', str(self.video)],
                       capture_output=True, check=True)
        self.lines = ('Xin chào, đây là một mục phụ đề dài cần xuống dòng để dễ đọc và giữ đúng mốc thời gian.',
                      'Hẹn gặp lại!')
        self.srt.write_text(f'1\n00:00:00,060 --> 00:00:01,000\n{self.lines[0]}\n\n'
                            f'2\n00:00:01,100 --> 00:00:02,000\n{self.lines[1]}\n', encoding='utf-8')
        self.factory = Mock(side_effect=FakeTurbo)
        self.tool = TurboTool(self.folder/'outputs', factory=self.factory)

    def test_full_job_publishes_silent_video_srt_and_wav_in_source_named_folder(self):
        original_video, original_srt = self.video.read_bytes(), self.srt.read_bytes()
        stages = []
        def progress(event):
            stages.append(event.step)
            with self.assertRaises(RuntimeError):
                self.tool.set_output_dir(self.folder/'other')
        result = self.tool.create_vietsub_video(self.video, self.srt, 'Mai', Sampling(speed=1.25, top_k=19),
                                               model_settings={}, progress=progress)
        self.assertEqual(set(stages), {1, 2, 3, 4, 5})
        self.assertEqual(sorted(stages), stages)
        self.assertEqual(result.video.parent, self.tool.output_dir / self.video.stem)
        self.assertEqual(set(self.tool.output_dir.iterdir()), {result.video.parent})
        self.assertEqual(set(result.video.parent.iterdir()), {result.video, result.subtitles, result.audio})
        self.assertEqual({item.path for item in self.tool.list_outputs()}, {result.video, result.subtitles, result.audio})
        self.assertEqual(self.video.read_bytes(), original_video)
        self.assertEqual(self.srt.read_bytes(), original_srt)
        cues = parse_srt(result.subtitles.read_text(encoding='utf-8'), strict_timing=True)
        self.assertEqual([cue.text for cue in cues], list(self.lines))
        self.assertEqual(cues[0].start_ms, 60)
        self.assertTrue(all(call[1]['voice']=='Mai' and call[1]['top_k']==19 for call in self.tool.tts.calls))
        info = ffmpeg_parse_infos(str(result.video))
        self.assertFalse(info['audio_found'])
        import soundfile as sf
        voice, rate = sf.read(result.audio)
        self.assertEqual(rate, 48000)
        self.assertGreater(len(voice), 0)
        self.assertAlmostEqual(info['duration'], result.duration, delta=.06)
        def hashes(path):
            result = subprocess.run([_ffmpeg_binary(), '-hide_banner', '-loglevel', 'error',
                                     '-i', str(path), '-map', '0:v:0', '-c:v', 'copy', '-f', 'framehash', '-'],
                                    capture_output=True, check=True, text=True)
            return [line.rsplit(',',1)[-1].strip() for line in result.stdout.splitlines()
                    if line.strip() and not line.startswith('#')]
        self.assertEqual(hashes(self.video), hashes(result.video))
        self.assertEqual(len(hashes(result.video)), 48)
        self.assertEqual(set(self.tool.delete_outputs([result.video])), {result.video, result.subtitles})
        self.assertTrue(result.audio.is_file())
        self.assertEqual([item.path for item in self.tool.list_outputs()], [result.audio])

    def test_invalid_timeline_is_rejected_before_model_load(self):
        self.srt.write_text('1\n00:00:03,000 --> 00:00:04,000\nToo late\n',encoding='utf-8')
        with self.assertRaises(ValueError):
            self.tool.create_vietsub_video(self.video,self.srt,model_settings={})
        self.factory.assert_not_called()
        self.assertEqual(self.tool.list_outputs(), [])

    def test_stop_during_model_load_does_not_reset_cancel_or_leave_intermediates(self):
        def factory(**kwargs):
            self.tool.stop()
            return FakeTurbo(**kwargs)
        self.tool._factory = factory
        with self.assertRaises(Cancelled):
            self.tool.create_vietsub_video(self.video,self.srt,model_settings={})
        self.assertEqual(list(self.tool.output_dir.iterdir()), [])
        self.assertTrue(self.tool.synthesize('Again').is_file())

    def test_video_failure_cleans_staged_video_audio_srt_and_releases_lock(self):
        with patch('apps.video_copy_export.render_video_copy',side_effect=OSError('video failed')):
            with self.assertRaisesRegex(OSError,'video failed'):
                self.tool.create_vietsub_video(self.video,self.srt,'Mai',model_settings={})
        self.assertEqual(list(self.tool.output_dir.iterdir()), [])
        self.assertTrue(self.tool.synthesize('Again').is_file())

    def test_stop_before_publication_leaves_no_result_folder(self):
        def progress(event):
            if event.step==5:
                self.tool.stop()
        with self.assertRaises(Cancelled):
            self.tool.create_vietsub_video(self.video,self.srt,'Mai',model_settings={},progress=progress)
        self.assertEqual(list(self.tool.output_dir.iterdir()), [])

    def task(self, subtitles=None):
        return VietsubTask(self.video, subtitles or self.srt, 'Mai', Sampling())

    def test_queue_continues_after_bad_item_reuses_model_and_never_overwrites_folder(self):
        tasks = [self.task(), self.task(self.folder / 'missing.srt'), self.task()]
        results, errors = [], []
        def notify(identifier, state, value):
            with self.assertRaises(RuntimeError):
                self.tool.set_output_dir(self.folder / 'other')
            if state == 'done':
                results.append(value)
            if state == 'error':
                errors.append(value)
        self.assertEqual(self.tool.create_vietsub_queue(tasks, model_settings={}, notify=notify), (2, 1))
        self.factory.assert_called_once()
        self.assertEqual(len(errors), 1)
        self.assertEqual([result.video.parent.name for result in results], ['source', 'source_2'])
        self.assertEqual(len(self.tool.list_outputs()), 6)
        restored = TurboTool(self.tool.output_dir)
        self.assertEqual({item.path for item in restored.list_outputs()},
                         {path for result in results for path in (result.video, result.audio, result.subtitles)})

    def test_stop_between_queue_items_keeps_completed_result_and_does_not_start_next(self):
        tasks = [self.task(), self.task()]
        events = []
        def notify(identifier, state, value):
            events.append((identifier, state))
            if state == 'done':
                self.tool.stop()
        self.assertEqual(self.tool.create_vietsub_queue(tasks, model_settings={}, notify=notify), (1, 0))
        self.assertNotIn((tasks[1].identifier, 'running'), events)
        self.assertEqual(len(self.tool.list_outputs()), 3)
        self.assertEqual([path.name for path in self.tool.output_dir.iterdir()], ['source'])

    def test_stop_inside_queued_item_cleans_only_current_item_and_does_not_reset_stop(self):
        tasks = [self.task(), self.task()]
        def notify(identifier, state, value):
            if state == 'progress' and value.step == 3:
                self.tool.stop()
        self.assertEqual(self.tool.create_vietsub_queue(tasks, model_settings={}, notify=notify), (0, 0))
        self.assertEqual(list(self.tool.output_dir.iterdir()), [])

    def test_publish_failure_preserves_existing_folder_and_cleans_staging(self):
        existing = self.tool.output_dir / 'source'
        existing.mkdir(parents=True)
        note = existing / 'keep.txt'
        note.write_text('original', encoding='utf-8')
        with patch('apps.vietsub_video._publish_folder', side_effect=OSError('publish failed')):
            with self.assertRaisesRegex(OSError, 'publish failed'):
                self.tool.create_vietsub_video(self.video, self.srt, model_settings={})
        self.assertEqual(list(self.tool.output_dir.iterdir()), [existing])
        self.assertEqual(note.read_text(encoding='utf-8'), 'original')


if __name__=='__main__':
    unittest.main()
