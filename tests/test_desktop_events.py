"""Progress flood, thread delivery and terminal ordering for the desktop UI."""
from queue import Empty
import threading
import unittest

from apps.desktop_events import DesktopEvents


class DesktopEventTests(unittest.TestCase):
    def drain(self, events):
        result = []
        while not events.empty():
            result.append(events.get_nowait())
        return result

    def test_progress_flood_preserves_latest_and_finished_result_in_order(self):
        events = DesktopEvents()
        events.put(("queue", ("SRT", "one", "running", None)))
        for index in range(10000):
            events.put(("queue", ("SRT", "one", "progress", index)))
        events.put(("queue", ("SRT", "one", "done", "one.wav")))
        events.put(("done", ("finished", None, None)))
        self.assertEqual(self.drain(events), [
            ("queue", ("SRT", "one", "running", None)),
            ("queue", ("SRT", "one", "progress", 9999)),
            ("queue", ("SRT", "one", "done", "one.wav")),
            ("done", ("finished", None, None)),
        ])

    def test_terminal_event_separates_progress_from_a_retry(self):
        events = DesktopEvents()
        for state, payload in (("running", None), ("progress", "first"),
                               ("error", "failed"), ("running", None), ("progress", "retry")):
            events.put(("queue", ("SRT", "one", state, payload)))
        result = self.drain(events)
        self.assertEqual(result[1][1][-1], "first")
        self.assertEqual(result[-1][1][-1], "retry")
        self.assertEqual([value[1][2] for value in result], ["running", "progress", "error", "running", "progress"])

    def test_independent_worker_progress_and_results_are_not_lost(self):
        events = DesktopEvents()

        def worker(identifier):
            events.put(("queue", ("SRT", identifier, "running", None)))
            for index in range(100):
                events.put(("queue", ("SRT", identifier, "progress", index)))
            events.put(("queue", ("SRT", identifier, "done", identifier)))

        workers = [threading.Thread(target=worker, args=(str(i),)) for i in range(4)]
        for worker_thread in workers:
            worker_thread.start()
        for worker_thread in workers:
            worker_thread.join()
        result = self.drain(events)
        for identifier in map(str, range(4)):
            states = [value[1] for value in result if value[1][1] == identifier]
            self.assertEqual(states[0][2], "running")
            self.assertEqual(states[-1][2:], ("done", identifier))
            self.assertEqual([value[3] for value in states if value[2] == "progress"][-1], 99)

    def test_empty_buffer_uses_queue_empty(self):
        with self.assertRaises(Empty):
            DesktopEvents().get_nowait()
