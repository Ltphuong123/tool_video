"""Sequential desktop queues; worker events are applied only by the Tk thread."""
from dataclasses import dataclass, field
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk
import uuid

from apps.v3turbo_tool import Cancelled
from apps.desktop_theme import COLORS


@dataclass(frozen=True)
class QueueTask:
    label: str
    text: str = ""
    source: Path | None = None
    identifier: str = field(default_factory=lambda: uuid.uuid4().hex)


def run_queue(tasks, process, cancelled, notify):
    """Continue after per-item failures; stop before the next item on cancellation."""
    done = failed = 0
    for task in tuple(tasks):
        if cancelled():
            break
        notify(task.identifier, "running", None)
        try:
            result = process(task, lambda message, key=task.identifier: notify(key, "progress", message))
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
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Số mục mỗi nhóm phải là số nguyên lớn hơn 0.")
    tasks = tuple(tasks)
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
            except Cancelled as exc:
                notify(task.identifier, "cancelled", str(exc))
                return done, failed
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
    """A compact native table, without a widget and timer for every queued item."""

    def __init__(self, parent):
        super().__init__(parent)
        self.tasks = []
        self.rows = {}
        self._syncing_selection = False
        self._tooltip = self._tooltip_job = self._hover_identifier = None
        self._counts = dict(pending=0, running=0, done=0, error=0, cancelled=0)
        table = ttk.Frame(self)
        table.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(table, columns=("label", "state"), show="headings",
                                 selectmode="extended", style="Queue.Treeview", height=6)
        self.tree.heading("label", text="Nội dung")
        self.tree.heading("state", text="Tiến trình")
        self.tree.column("label", width=170, minwidth=100)
        self.tree.column("state", width=175, minwidth=100)
        colors = (("pending", COLORS["muted"], "#f8f9ff"),
                  ("running", COLORS["blue"], "#eef5ff"),
                  ("done", COLORS["teal"], "#edf9f3"),
                  ("error", "#bd3445", "#fff0f2"),
                  ("cancelled", "#a46a10", "#fff6e6"))
        for state, foreground, background in colors:
            self.tree.tag_configure(state, foreground=foreground, background=background)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._selection_changed)
        self.tree.bind("<Motion>", self._hover_row)
        self.tree.bind("<Leave>", lambda event: self._hide_tooltip())
        self.tree.bind("<ButtonPress>", lambda event: self._hide_tooltip(), add="+")
        self.bind("<Destroy>", self._destroy_tooltip, add="+")
        self.summary = ttk.Label(self, text="Chưa có mục nào", style="Muted.TLabel")
        self.summary.pack(side="bottom", anchor="w", pady=(5, 0), before=table)

    def add(self, task):
        if task.identifier in self.rows:
            raise ValueError("Mục này đã có trong hàng đợi.")
        self.tasks.append(task)
        selected = tk.BooleanVar(master=self)
        status = tk.StringVar(master=self, value="Chờ")
        self.rows[task.identifier] = dict(selected=selected, state="pending", status=status, progress=0.0,
                                         label=task.label, source=task.source)
        self.tree.insert("", "end", iid=task.identifier, tags=("pending",), values=(task.label, "Chờ"))
        self.rows[task.identifier]["selection_trace"] = selected.trace_add(
            "write", lambda *args, key=task.identifier: self._select_row(key))
        self._counts["pending"] += 1
        self._refresh_summary()

    def _select_row(self, identifier):
        if self._syncing_selection:
            return
        row = self.rows.get(identifier)
        if row is not None:
            if row["selected"].get():
                self.tree.selection_add(identifier)
            else:
                self.tree.selection_remove(identifier)

    def _selection_changed(self, event=None):
        selection = set(self.tree.selection())
        self._syncing_selection = True
        try:
            for identifier, row in self.rows.items():
                row["selected"].set(identifier in selection)
        finally:
            self._syncing_selection = False

    def _refresh_summary(self):
        parts = [f"{self._counts['done']}/{len(self.tasks)} xong"]
        if self._counts["error"]:
            parts.append(f"{self._counts['error']} lỗi")
        if self._counts["cancelled"]:
            parts.append(f"{self._counts['cancelled']} đã dừng")
        self.summary.configure(text=" · ".join(parts) if self.tasks else "Chưa có mục nào")

    def _hover_row(self, event):
        identifier = self.tree.identify_row(event.y)
        if identifier == self._hover_identifier:
            return
        self._hide_tooltip()
        self._hover_identifier = identifier
        if identifier:
            self._tooltip_job = self.after(450, lambda: self._show_tooltip(identifier, event.x_root, event.y_root))

    def _show_tooltip(self, identifier, left, top):
        self._tooltip_job = None
        row = self.rows.get(identifier)
        if row is None or not self.tree.winfo_viewable():
            return
        message = row["label"] + "\n" + row["status"].get()
        if row["source"] is not None:
            message += "\n" + str(row["source"])
        self._tooltip = tk.Toplevel(self)
        self._tooltip.overrideredirect(True)
        tk.Label(self._tooltip, text=message, justify="left", wraplength=380, padx=10, pady=8,
                 background="#eef1ff", foreground=COLORS["ink"], font=("Segoe UI", 9),
                 relief="solid", borderwidth=1).pack()
        self._tooltip.update_idletasks()
        left = min(left + 14, max(0, self.winfo_screenwidth() - self._tooltip.winfo_reqwidth() - 8))
        top = min(top + 18, max(0, self.winfo_screenheight() - self._tooltip.winfo_reqheight() - 8))
        self._tooltip.geometry(f"+{max(0, left)}+{max(0, top)}")

    def _hide_tooltip(self):
        if self._tooltip_job is not None:
            self.after_cancel(self._tooltip_job)
            self._tooltip_job = None
        if self._tooltip is not None:
            self._tooltip.destroy()
            self._tooltip = None
        self._hover_identifier = None

    def _destroy_tooltip(self, event):
        if event.widget is self:
            self._hide_tooltip()

    def pending(self):
        return tuple(task for task in self.tasks if self.rows[task.identifier]["state"] in ("pending", "error", "cancelled"))

    def remove_selected(self):
        self._hide_tooltip()
        removed = set()
        for task in tuple(self.tasks):
            row = self.rows[task.identifier]
            if row["selected"].get() and row["state"] != "running":
                row["selected"].trace_remove("write", row["selection_trace"])
                self.tree.delete(task.identifier)
                self._counts[row["state"]] -= 1
                removed.add(task.identifier)
                del self.rows[task.identifier]
        self.tasks[:] = [task for task in self.tasks if task.identifier not in removed]
        self._refresh_summary()

    def update_task(self, identifier, state, value):
        row = self.rows.get(identifier)
        if row is None:
            return
        if state == "progress" and row["state"] != "running":
            # A delayed worker message cannot replace a finished result.
            return
        if state != "progress":
            if state not in self._counts:
                raise ValueError(f"Trạng thái hàng đợi không hợp lệ: {state}")
            self._counts[row["state"]] -= 1
            self._counts[state] += 1
            row["state"] = state
        if state == "running":
            row["status"].set("Đang tạo audio…")
            row["progress"] = 0.0
            self.tree.see(identifier)
        elif state == "progress":
            row["status"].set(str(value))
            match = re.search(r"(\d+)\s*/\s*(\d+)\s*(?:câu|audio)", str(value))
            if match and int(match[2]):
                row["progress"] = min(99.0, int(match[1]) / int(match[2]) * 100)
        elif state == "pending":
            row["progress"] = 0.0
            row["status"].set("Chờ")
        else:
            row["progress"] = 100.0 if state == "done" else 0.0
            path = value.audio if hasattr(value, "audio") else value[0] if isinstance(value, tuple) else value
            row["status"].set(f"Xong: {Path(path).name}" if state == "done" else
                              f"{'Đã dừng' if state == 'cancelled' else 'Lỗi'}: {value}")
        self.tree.item(identifier, tags=(row["state"],))
        message = row["status"].get()
        if row["state"] == "running" and row["progress"]:
            message = f"{row['progress']:.0f}% · {message}"
        self.tree.set(identifier, "state", message)
        if state != "progress":
            self._refresh_summary()

    def stop_bars(self):
        """The table has no per-item animation timers to stop."""
        self._hide_tooltip()
