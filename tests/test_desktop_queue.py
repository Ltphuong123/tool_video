"""Queue ordering, error isolation and cancellation tests."""
import unittest

from apps.desktop_queue import QueueTask, run_queue
from apps.v3turbo_tool import Cancelled


class QueueTests(unittest.TestCase):
    def test_text_queue_batches_valid_inputs_and_isolates_bad_text(self):
        from apps.desktop_queue import run_text_queue
        tasks = [QueueTask("one", text="One"), QueueTask("bad", text=""),
                 QueueTask("two", text="Two"), QueueTask("three", text="Three")]
        groups, events = [], []
        def batch(texts):
            groups.append(texts)
            return [text + ".wav" for text in texts]
        result = run_text_queue(tasks, lambda task: task.text, batch, 2, lambda: False,
                                lambda *args: events.append(args))
        self.assertEqual(result, (3, 1))
        self.assertEqual(groups, [["One"], ["Two", "Three"]])
        self.assertEqual([event[2] for event in events if event[1] == "done"],
                         ["One.wav", "Two.wav", "Three.wav"])

    def test_text_batch_cancellation_stops_later_groups(self):
        from apps.desktop_queue import run_text_queue
        stopped, groups = [], []
        def batch(texts):
            groups.append(texts)
            stopped.append(True)
            return ["one.wav"]
        result = run_text_queue([QueueTask("one", text="One"), QueueTask("two", text="Two")],
                                lambda task: task.text, batch, 1, lambda: bool(stopped),
                                lambda *args: None)
        self.assertEqual(result, (1, 0))
        self.assertEqual(groups, [["One"]])

    def test_order_error_isolation_and_progress(self):
        tasks = [QueueTask("one"), QueueTask("bad"), QueueTask("three")]
        calls, events = [], []
        def process(task, progress):
            calls.append(task.label)
            if task.label == "bad":
                raise ValueError("invalid input")
            progress("1/2 câu")
            return task.label + ".wav"
        counts = run_queue(tasks, process, lambda: False, lambda *args: events.append(args))
        self.assertEqual(calls, ["one", "bad", "three"])
        self.assertEqual(counts, (2, 1))
        self.assertEqual([event[1] for event in events].count("done"), 2)
        self.assertEqual([event[1] for event in events].count("error"), 1)

    def test_cancellation_preserves_unstarted_items(self):
        tasks = [QueueTask("one"), QueueTask("two")]
        stopped, calls, events = [], [], []
        def process(task, progress):
            calls.append(task.label)
            stopped.append(True)
            return "one.wav"
        self.assertEqual(run_queue(tasks, process, lambda: bool(stopped),
                                   lambda *args: events.append(args)), (1, 0))
        self.assertEqual(calls, ["one"])
        self.assertFalse(any(event[0] == tasks[1].identifier for event in events))

    def test_cancelled_current_item_stops_queue(self):
        calls, events = [], []
        def process(task, progress):
            calls.append(task.label)
            raise Cancelled("stopped")
        self.assertEqual(run_queue([QueueTask("one"), QueueTask("two")], process,
                                   lambda: False, lambda *args: events.append(args)), (0, 0))
        self.assertEqual(calls, ["one"])
        self.assertEqual(events[-1][1], "cancelled")
