"""Queue ordering, error isolation and cancellation tests."""
import unittest
import tkinter as tk

from apps.desktop_queue import QueuePanel, QueueTask, run_queue, run_text_queue
from apps.v3turbo_tool import Cancelled


class QueueTests(unittest.TestCase):
    def test_text_queue_rejects_invalid_group_sizes_without_reading_input(self):
        reads = []
        for size in (0, -1, 1.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                run_text_queue([QueueTask("one")], lambda task: reads.append(task),
                               lambda texts: texts, size, lambda: False, lambda *args: None)
        self.assertEqual(reads, [])

    def test_cancellation_while_reading_input_does_not_synthesize_or_fail_later_items(self):
        tasks = [QueueTask("one"), QueueTask("stop"), QueueTask("three")]
        reads, batches, events = [], [], []
        def load(task):
            reads.append(task.label)
            if task.label == "stop":
                raise Cancelled("stopped while reading")
            return task.label
        result = run_text_queue(tasks, load, lambda texts: batches.append(texts), 3,
                                lambda: False, lambda *args: events.append(args))
        self.assertEqual(result, (0, 0))
        self.assertEqual(reads, ["one", "stop"])
        self.assertEqual(batches, [])
        self.assertEqual(events, [(tasks[1].identifier, "cancelled", "stopped while reading")])

    def test_delayed_progress_remains_attached_to_its_original_item(self):
        tasks = [QueueTask("one"), QueueTask("two")]
        callbacks, events = [], []
        def process(task, progress):
            callbacks.append(progress)
            return task.label + ".wav"
        run_queue(tasks, process, lambda: False, lambda *args: events.append(args))
        callbacks[0]("first background update")
        self.assertEqual(events[-1], (tasks[0].identifier, "progress", "first background update"))

    def test_queue_run_uses_a_snapshot_when_caller_changes_input_list(self):
        tasks = [QueueTask("one"), QueueTask("two")]
        calls = []
        def process(task, progress):
            calls.append(task.label)
            if task.label == "one":
                tasks.clear()
            return task.label + ".wav"
        self.assertEqual(run_queue(tasks, process, lambda: False, lambda *args: None), (2, 0))
        self.assertEqual(calls, ["one", "two"])

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


class QueuePanelTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.panel = QueuePanel(self.root)
        self.panel.pack()

    def test_native_selection_removes_only_selected_items_and_keeps_running_item(self):
        tasks = [QueueTask("one"), QueueTask("two"), QueueTask("three")]
        for task in tasks:
            self.panel.add(task)
        self.panel.update_task(tasks[0].identifier, "running", None)
        self.panel.tree.selection_set(tasks[0].identifier, tasks[1].identifier)
        self.root.update_idletasks()
        self.panel._selection_changed()
        self.panel.remove_selected()
        self.assertEqual(self.panel.tasks, [tasks[0], tasks[2]])
        self.assertEqual(self.panel.tree.get_children(), (tasks[0].identifier, tasks[2].identifier))

    def test_progress_completion_retry_and_delayed_events_preserve_result(self):
        task = QueueTask("first.srt")
        self.panel.add(task)
        self.panel.update_task(task.identifier, "running", None)
        self.panel.update_task(task.identifier, "progress", "SRT 3/10 câu")
        self.assertEqual(self.panel.rows[task.identifier]["progress"], 30)
        self.assertIn("30%", self.panel.tree.set(task.identifier, "state"))
        self.panel.update_task(task.identifier, "cancelled", "stopped")
        self.assertEqual(self.panel.pending(), (task,))
        self.panel.update_task(task.identifier, "running", None)
        self.assertEqual(self.panel.rows[task.identifier]["progress"], 0)
        self.panel.update_task(task.identifier, "done", ("result.wav", "ready"))
        self.panel.update_task(task.identifier, "progress", "late update")
        self.assertEqual(self.panel.rows[task.identifier]["progress"], 100)
        self.assertEqual(self.panel.tree.set(task.identifier, "state"), "Xong: result.wav")
        self.assertEqual(self.panel.pending(), ())
        self.assertEqual(self.panel.summary.cget("text"), "1/1 xong")

    def test_duplicate_identifiers_are_rejected_before_corrupting_queue(self):
        task = QueueTask("one")
        self.panel.add(task)
        with self.assertRaises(ValueError):
            self.panel.add(task)
        self.assertEqual(self.panel.tasks, [task])
        self.assertEqual(len(self.panel.tree.get_children()), 1)

    def test_removed_item_accepts_no_stale_worker_updates(self):
        task = QueueTask("one")
        self.panel.add(task)
        self.panel.rows[task.identifier]["selected"].set(True)
        self.panel.remove_selected()
        self.panel.update_task(task.identifier, "done", "result.wav")
        self.assertEqual(self.panel.tasks, [])
        self.assertEqual(self.panel.summary.cget("text"), "Chưa có mục nào")
