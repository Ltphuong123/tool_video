"""Native audio concurrency keeps source reads, samples, and cleanup safe."""
import importlib.util
import threading
import unittest
from unittest.mock import patch

import numpy as np

from apps.video_audio_parallel import ordered_parallel_map


class Cancelled(RuntimeError):
    pass


class ParallelAudioTests(unittest.TestCase):
    def test_out_of_order_completion_still_yields_input_order(self):
        later_done = threading.Event()
        order = []

        def operation(value):
            if value == 0:
                self.assertTrue(later_done.wait(3))
            else:
                order.append(value)
                later_done.set()
            return value

        self.assertEqual(list(ordered_parallel_map(operation, range(8), workers=4)),
                         list(range(8)))
        self.assertNotEqual(order, [])

    def test_prefetch_bounds_source_reads_and_keeps_them_on_caller_thread(self):
        caller = threading.get_ident()
        reads = []
        closed = []

        def jobs():
            try:
                for value in range(100):
                    self.assertEqual(threading.get_ident(), caller)
                    reads.append(value)
                    yield value
            finally:
                closed.append(True)

        results = ordered_parallel_map(lambda value: value, jobs(), workers=4, prefetch=4)
        self.assertEqual(next(results), 0)
        self.assertEqual(reads, [0, 1, 2, 3])
        self.assertEqual(next(results), 1)
        self.assertEqual(reads, [0, 1, 2, 3, 4])
        results.close()
        self.assertEqual(closed, [True])
        self.assertEqual(len(reads), 5)

    def test_close_waits_for_active_native_calls_before_returning(self):
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def operation(value):
            if value == 1:
                entered.set()
                self.assertTrue(release.wait(3))
                finished.set()
            return value

        results = ordered_parallel_map(operation, range(4), workers=2)
        self.assertEqual(next(results), 0)
        self.assertTrue(entered.wait(3))
        closer = threading.Thread(target=results.close)
        closer.start()
        self.assertFalse(finished.is_set())
        self.assertTrue(closer.is_alive())
        release.set()
        closer.join(3)
        self.assertFalse(closer.is_alive())
        self.assertTrue(finished.is_set())

    def test_cancel_preserves_original_exception_and_closes_source_generator(self):
        closed = []
        calls = 0
        cancellation = Cancelled("stop export")

        def jobs():
            try:
                yield from range(100)
            finally:
                closed.append(True)

        def check_stop():
            nonlocal calls
            calls += 1
            if calls == 6:
                raise cancellation

        with self.assertRaises(Cancelled) as caught:
            list(ordered_parallel_map(lambda value: value, jobs(),
                                      check_stop=check_stop, workers=4))
        self.assertIs(caught.exception, cancellation)
        self.assertEqual(closed, [True])

    def test_worker_exception_is_preserved_without_consuming_entire_source(self):
        reads = []
        failure = ValueError("bad native samples")

        def jobs():
            for value in range(100):
                reads.append(value)
                yield value

        def operation(value):
            if value == 0:
                raise failure
            return value

        with self.assertRaises(ValueError) as caught:
            list(ordered_parallel_map(operation, jobs(), workers=4))
        self.assertIs(caught.exception, failure)
        self.assertEqual(reads, [0, 1, 2, 3])

    def test_worker_timeout_error_is_not_confused_with_wait_timeout(self):
        failure = TimeoutError("native worker failed")

        def operation(value):
            raise failure

        with self.assertRaises(TimeoutError) as caught:
            list(ordered_parallel_map(operation, [1, 2], workers=2))
        self.assertIs(caught.exception, failure)

    def test_zero_one_and_serial_jobs_do_not_create_executor(self):
        with patch("apps.video_audio_parallel.ThreadPoolExecutor") as pool:
            self.assertEqual(list(ordered_parallel_map(lambda value: value * 2, [])), [])
            self.assertEqual(list(ordered_parallel_map(lambda value: value * 2, [3])), [6])
            self.assertEqual(list(ordered_parallel_map(lambda value: value * 2, [1, 2, 3],
                                                       workers=1)), [2, 4, 6])
            self.assertEqual(list(ordered_parallel_map(lambda value: value * 2, [1, 2, 3],
                                                       workers=4, prefetch=1)), [2, 4, 6])
            pool.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("pedalboard"), "optional Rubber Band absent")
    def test_native_stereo_rubberband_matches_serial_samples_bit_for_bit(self):
        from pedalboard import time_stretch

        times = np.arange(12000, dtype=np.float32) / 48000
        tone = np.stack((.2 * np.sin(2*np.pi*220*times),
                         .15 * np.sin(2*np.pi*440*times)))
        jobs = [(np.ascontiguousarray(tone), factor) for factor in (.7, 1.3, 2.0, .6)]

        def operation(item):
            data, factor = item
            return time_stretch(data, 48000, stretch_factor=factor,
                                pitch_shift_in_semitones=0.0, high_quality=True,
                                retain_phase_continuity=True, preserve_formants=True)

        serial = list(ordered_parallel_map(operation, jobs, workers=1))
        parallel = list(ordered_parallel_map(operation, jobs, workers=4))
        for expected, actual in zip(serial, parallel):
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
