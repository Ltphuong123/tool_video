"""CUDA lookahead, ordered CPU results and failure cleanup without a GPU."""
from contextlib import closing
import threading
import unittest

import numpy as np

from apps.speech_pipeline import processed_audio_batches


class SpeechPipelineTests(unittest.TestCase):
    def test_next_generation_overlaps_processing_without_advancing_three_batches(self):
        caller = threading.get_ident()
        processing_started, next_generated = threading.Event(), threading.Event()
        consumed = []

        def source():
            self.assertEqual(threading.get_ident(), caller)
            yield 0, [np.ones(16, dtype=np.float32)], .1
            self.assertTrue(processing_started.wait(3))
            next_generated.set()
            yield 1, [np.full(16, 2, dtype=np.float32)], .2
            self.assertEqual(consumed, [1])
            yield 2, [np.full(16, 3, dtype=np.float32)], .3

        def prepare(index, wav):
            self.assertNotEqual(threading.get_ident(), caller)
            if index == 0:
                processing_started.set()
                self.assertTrue(next_generated.wait(3))
            return int(wav[0])

        with closing(processed_audio_batches(source(), prepare, lambda start, wavs: 1,
                                            overlap=True)) as batches:
            for batch in batches:
                consumed.extend(batch.results)
                self.assertGreaterEqual(batch.processing_seconds, 0)
        self.assertEqual(consumed, [1, 2, 3])

    def test_sdk_owned_buffers_are_copied_before_next_generation(self):
        buffer = np.ones(16, dtype=np.float32)
        next_generated = threading.Event()

        def source():
            yield 0, [buffer], 0
            buffer[:] = 2
            next_generated.set()
            yield 1, [buffer], 0

        def prepare(index, wav):
            if index == 0:
                self.assertTrue(next_generated.wait(3))
            return wav.copy()

        results = []
        with closing(processed_audio_batches(source(), prepare, lambda start, wavs: 1,
                                            overlap=True)) as batches:
            for batch in batches:
                results.extend(batch.results)
        np.testing.assert_array_equal(results[0], np.ones(16, dtype=np.float32))
        np.testing.assert_array_equal(results[1], np.full(16, 2, dtype=np.float32))

    def test_generation_failure_waits_for_native_processing_and_closes_source(self):
        started, release, finished, source_closed = (threading.Event() for _ in range(4))

        def source():
            try:
                yield 0, [np.ones(16, dtype=np.float32)], 0
                self.assertTrue(started.wait(3))
                release.set()
                raise RuntimeError("generation failed")
            finally:
                source_closed.set()

        def prepare(index, wav):
            started.set()
            self.assertTrue(release.wait(3))
            finished.set()
            return wav

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with closing(processed_audio_batches(source(), prepare, lambda start, wavs: 1,
                                                overlap=True)) as batches:
                list(batches)
        self.assertTrue(finished.is_set())
        self.assertTrue(source_closed.is_set())

    def test_worker_timeout_is_reported_instead_of_polling_forever(self):
        def prepare(index, wav):
            raise TimeoutError("native operation failed")

        with self.assertRaisesRegex(TimeoutError, "native operation failed"):
            with closing(processed_audio_batches([(0, [np.ones(16)], 0)], prepare,
                                                lambda start, wavs: 1, overlap=True)) as batches:
                for batch in batches:
                    list(batch.results)


if __name__ == "__main__":
    unittest.main()
