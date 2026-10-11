"""Thread-safe UI events that keep the latest progress without losing outcomes."""
from collections import deque
from queue import Empty
import threading


class DesktopEvents:
    def __init__(self):
        self._items = deque()
        self._progress = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(kind, value):
        if kind in ("status", "vietsub_progress"):
            return (kind,)
        if kind == "queue" and value[2] == "progress":
            return (kind, value[0], value[1])
        if kind == "vietsub_queue" and value[3] == "progress":
            return (kind, value[2].identifier)
        return None

    def put(self, event):
        kind, value = event
        key = self._key(kind, value)
        with self._lock:
            if key is None:
                # A terminal event or a new task run separates progress groups.
                # Older entries remain in order; later progress gets a new entry.
                self._progress.clear()
            elif key in self._progress:
                self._progress[key][0] = event
                return
            entry = [event, key]
            self._items.append(entry)
            if key is not None:
                self._progress[key] = entry

    def get_nowait(self):
        with self._lock:
            if not self._items:
                raise Empty
            entry = self._items.popleft()
            event, key = entry
            if key is not None and self._progress.get(key) is entry:
                del self._progress[key]
            return event

    def empty(self):
        with self._lock:
            return not self._items
