import unittest
import numpy as np

from apps.speech_subtitles import (
    SentenceSpan, assemble_sentence_track, render_srt, subtitle_sentences,
)


class SpeechSubtitlesTests(unittest.TestCase):
    def test_sentences_preserve_quotes_decimals_and_paragraph_breaks(self):
        text = 'Giá 3.5 triệu. Anh nói "Thật sao?", rồi đi.\nCâu cuối!'
        self.assertEqual(subtitle_sentences(text), [
            ("Giá 3.5 triệu.", "sentence"),
            ('Anh nói "Thật sao?", rồi đi.', "sentence"),
            ("Câu cuối!", "para"),
        ])

    def test_srt_formats_hours_and_readable_emotion_tokens(self):
        start = 3661123 * 48
        result = render_srt([SentenceSpan("<en>Hello</en> <|emotion_1|>.", start, start + 48000)])
        self.assertIn("01:01:01,123 --> 01:01:02,123", result)
        self.assertIn("Hello [cười].", result)
        self.assertNotIn("<|", result)

    def test_existing_silence_is_counted_without_extra_padding(self):
        a = np.concatenate([np.ones(48000) * 0.1, np.zeros(24000)]).astype(np.float32)
        b = np.concatenate([np.zeros(9600), np.ones(48000) * 0.1]).astype(np.float32)
        track, spans = assemble_sentence_track([("A.", "sentence"), ("B.", "sentence")], [a, b])
        self.assertEqual(len(track), len(a) + len(b))
        self.assertEqual(spans[0].end_sample, 48000)
        self.assertEqual(spans[1].start_sample, len(a) + 9600)
        self.assertTrue(spans[1].end_sample <= len(track))

    def test_silent_audio_has_valid_interval_and_invalid_clip_is_rejected(self):
        track, spans = assemble_sentence_track([("Silence.", "sentence")], [np.zeros(4800)])
        self.assertEqual((spans[0].start_sample, spans[0].end_sample), (0, 4800))
        self.assertIn("00:00:00,000 --> 00:00:00,100", render_srt(spans))
        for clips in ([], [np.array([])], [np.array([np.nan])]):
            with self.assertRaises(ValueError):
                assemble_sentence_track([("Text.", "sentence")], clips)

    def test_blank_text_is_rejected(self):
        with self.assertRaises(ValueError):
            subtitle_sentences(" \n\r\n ")


if __name__ == "__main__":
    unittest.main()
