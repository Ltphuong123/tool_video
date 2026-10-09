"""Sequential desktop queues; worker events are applied only by the Tk thread."""
from dataclasses import dataclass, field
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk
import uuid

from apps.v3turbo_tool import Cancelled


@dataclass(frozen=True)
class QueueTask:
    label: str
    text: str = ""
    source: Path | None = None
    identifier: str = field(default_factory=lambda: uuid.uuid4().hex)


def run_queue(tasks, process, cancelled, notify):
    """Continue after per-item failures; stop before the next item on cancellation."""
    done = failed = 0
    for task in tasks:
        if cancelled():
            break
        notify(task.identifier, "running", None)
        try:
            result = process(task, lambda message: notify(task.identifier, "progress", message))
        except Cancelled as exc:
            notify(task.identifier, "cancelled", str(exc))
            break
        except Exception as exc:
            failed += 1
            notify(task.identifier, "error", str(exc))
        else:
            done += 1
            notify(task.identifier, "done", result)
    return done, failed


def run_text_queue(tasks, load_text, process_batch, batch_size, cancelled, notify):
    """Validate inputs separately, then synthesize one group per SDK call."""
    done = failed = 0
    for start in range(0, len(tasks), batch_size):
        if cancelled():
            break
        ready, texts = [], []
        for task in tasks[start:start + batch_size]:
            if cancelled():
                break
            try:
                text = load_text(task)
                if not text.strip():
                    raise ValueError("Văn bản trống.")
            except Exception as exc:
                failed += 1
                notify(task.identifier, "error", str(exc))
            else:
                ready.append(task)
                texts.append(text)
        if cancelled():
            break
        if not ready:
            continue
        for task in ready:
            notify(task.identifier, "running", None)
        try:
            results = process_batch(texts)
            if len(results) != len(ready):
                raise RuntimeError("Số kết quả không khớp số mục trong nhóm.")
        except Cancelled as exc:
            for task in ready:
                notify(task.identifier, "cancelled", str(exc))
            break
        except Exception as exc:
            failed += len(ready)
            for task in ready:
                notify(task.identifier, "error", str(exc))
        else:
            for task, result in zip(ready, results):
                done += 1
                notify(task.identifier, "done", result)
    return done, failed


class QueuePanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.tasks = []
        self.rows = {}
        canvas = tk.Canvas(self, background="#ffffff", highlightthickness=0, height=185)
        scroll = ttk.Scrollbar(self, command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self.body = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=self.body, anchor="nw")
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        self.body.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))

    def add(self, task):
        self.tasks.append(task)
        frame = ttk.Frame(self.body, padding=(8, 5))
        frame.pack(fill="x")
        selected = tk.BooleanVar()
        ttk.Checkbutton(frame, variable=selected).pack(side="left")
        detail = ttk.Frame(frame)
        detail.pack(side="left", fill="x", expand=True)
        ttk.Label(detail, text=task.label, wraplength=700).pack(anchor="w")
        status = tk.StringVar(value="Chờ")
        ttk.Label(detail, textvariable=status, style="Muted.TLabel", wraplength=700).pack(anchor="w")
        bar = ttk.Progressbar(detail, mode="determinate", maximum=100)
        bar.pack(fill="x", pady=3)
        self.rows[task.identifier] = dict(frame=frame, selected=selected, state="pending", status=status, bar=bar)

    def pending(self):
        return tuple(task for task in self.tasks if self.rows[task.identifier]["state"] in ("pending", "error", "cancelled"))

    def remove_selected(self):
        for task in tuple(self.tasks):
            row = self.rows[task.identifier]
            if row["selected"].get():
                row["bar"].stop()
                row["frame"].destroy()
                self.tasks.remove(task)
                del self.rows[task.identifier]

    def update_task(self, identifier, state, value):
        row = self.rows.get(identifier)
        if row is None:
            return
        bar = row["bar"]
        if state == "running":
            row["state"] = state
            row["status"].set("Đang tạo audio…")
            bar.configure(mode="indeterminate", value=0)
            bar.start(15)
        elif state == "progress":
            row["status"].set(str(value))
            match = re.search(r"(\d+)\s*/\s*(\d+)\s*(?:câu|audio)", str(value))
            if match and int(match[2]):
                bar.stop()
                bar.configure(mode="determinate", value=min(99, int(match[1]) / int(match[2]) * 100))
        else:
            row["state"] = state
            bar.stop()
            bar.configure(mode="determinate", value=100 if state == "done" else 0)
            path = value.audio if hasattr(value, "audio") else value[0] if isinstance(value, tuple) else value
            row["status"].set(f"Xong: {Path(path).name}" if state == "done" else
                              f"{'Đã dừng' if state == 'cancelled' else 'Lỗi'}: {value}")

    def stop_bars(self):
        for row in self.rows.values():
            row["bar"].stop()
