"""A draggable source-time ruler and speed segments for the desktop editor."""
from __future__ import annotations

from dataclasses import replace
import math
import tkinter as tk

from apps.video_editor import SpeedSegment


def time_label(seconds: float) -> str:
    seconds = max(0, float(seconds))
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return (f"{hours:d}:{minutes:02d}:{remainder:05.2f}" if hours else
            f"{minutes:02d}:{remainder:05.2f}")


class VideoTimeline(tk.Canvas):
    """Source-time ruler, editable speed ranges and fixed timestamp pins."""

    def __init__(self, parent, on_select=None, on_change=None, on_seek=None, **kwargs):
        kwargs.setdefault("height", 116)
        kwargs.setdefault("background", "#f8faff")
        kwargs.setdefault("highlightthickness", 0)
        kwargs.setdefault("takefocus", True)
        super().__init__(parent, **kwargs)
        self.duration = 0.0
        self.position = 0.0
        self.segments = []
        self.selected = None
        self.selection = None
        self.enabled = True
        self.editable = True
        self.markers = ()
        self.target_markers = ()
        self._pin_items = {}
        self._tooltip = None
        self._tooltip_job = None
        self._hover_marker = None
        self.on_select = on_select or (lambda start, end, index: None)
        self.on_change = on_change or (lambda segments, index: None)
        self.on_seek = on_seek or (lambda seconds: None)
        self._drag = None
        self.bind("<Configure>", lambda event: self.redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Motion>", self._hover)
        self.bind("<Leave>", lambda event: self._hide_tooltip())
        self.bind("<Destroy>", self._destroy_tooltip, add="+")
        self.bind("<Left>", lambda event: self._step(-1))
        self.bind("<Right>", lambda event: self._step(1))

    def set_video(self, duration):
        self.duration = max(0, float(duration))
        self.position = 0.0
        self.segments = []
        self.markers = self.target_markers = ()
        self.selected = self.selection = self._drag = None
        self.redraw()

    def set_markers(self, old_markers, new_markers=()):
        """Pin marker objects with ``index`` and ``time_ms`` to source time.

        Destination timestamps describe the correspondence in the tooltip;
        they never move a pin away from its source frame.
        """
        self.markers = tuple(old_markers)
        self.target_markers = tuple(new_markers)
        self._drag = None
        self.redraw()

    def set_editable(self, editable):
        """Lock range edits while retaining seeking and fixed marker clicks."""
        self.editable = bool(editable)
        if not self.editable and self._drag and self._drag[0] not in ("seek", "pin"):
            self._drag = None

    def set_segments(self, segments, selected=None):
        self.segments = list(segments)
        self.selected = selected if selected is not None and 0 <= selected < len(self.segments) else None
        if self.selected is not None:
            segment = self.segments[self.selected]
            self.selection = (segment.start, segment.end)
        else:
            self.selection = None
        self.redraw()

    def set_selection(self, start, end):
        if self.duration and 0 <= start < end <= self.duration:
            self.selection = (float(start), float(end))
        else:
            self.selection = None
        self.redraw()

    def set_position(self, seconds):
        self.position = max(0, min(self.duration, float(seconds)))
        self.delete("playhead")
        if self.duration:
            self._draw_playhead()

    def _bounds(self):
        return 16.0, max(17.0, float(self.winfo_width()) - 16)

    def time_to_x(self, seconds):
        left, right = self._bounds()
        return left + (right - left) * float(seconds) / max(self.duration, 1e-9)

    def x_to_time(self, x):
        left, right = self._bounds()
        return self.duration * max(0, min(1, (float(x) - left) / (right - left)))

    def redraw(self):
        self._hide_tooltip()
        self.delete("all")
        self._pin_items.clear()
        left, right = self._bounds()
        if not self.duration:
            self.create_text((left + right) / 2, 45, text="Mở video để chọn đoạn trên thanh thời gian",
                             fill="#64718a", font=("Segoe UI", 9))
            return
        self.create_rectangle(left, 38, right, 76, fill="#e7edf6", outline="")
        target = self.duration / max(2, (right - left) / 75)
        exponent = 10 ** math.floor(math.log10(max(target, 0.001)))
        step = next(multiplier * exponent for multiplier in (1, 2, 5, 10)
                    if multiplier * exponent >= target)
        tick = 0.0
        while tick <= self.duration + 1e-9:
            x = self.time_to_x(tick)
            self.create_line(x, 25, x, 34, fill="#9facbf")
            self.create_text(x, 15, text=time_label(tick), fill="#64718a", font=("Segoe UI", 8))
            tick += step
        for index, segment in enumerate(self.segments):
            x1, x2 = self.time_to_x(segment.start), self.time_to_x(segment.end)
            active = index == self.selected
            self.create_rectangle(x1, 39, x2, 75, fill="#5b48ef" if active else "#b9b1fa",
                                  outline="#392b98" if active else "#9589ee", width=2 if active else 1)
            if x2 - x1 > 35:
                self.create_text((x1 + x2) / 2, 57, text=f"{segment.speed:g}x",
                                 fill="#ffffff" if active else "#392b98", font=("Segoe UI", 9, "bold"))
            for x in (x1, x2):
                self.create_line(x, 42, x, 72, width=5, fill="#392b98" if active else "#9589ee")
        if self.selected is None and self.selection:
            start, end = self.selection
            self.create_rectangle(self.time_to_x(start), 38, self.time_to_x(end), 76,
                                  fill="#d9e9ff", outline="#3984d5", dash=(3, 2), width=2)
        self._draw_markers()
        self._draw_playhead()

    def _draw_markers(self):
        visible = [marker for marker in self.markers if 0 <= marker.time_ms / 1000 <= self.duration]
        positions = [self.time_to_x(marker.time_ms / 1000) for marker in visible]
        for position, (marker, x) in enumerate(zip(visible, positions)):
            tags = ("marker", f"marker:{marker.index}")
            self.create_line(x, 35, x, 88, fill="#c48716", dash=(2, 3), tags=tags)
            pin = self.create_polygon(x, 83, x - 5, 90, x - 5, 96,
                                      x + 5, 96, x + 5, 90, fill="#d99319",
                                      outline="#875608", tags=(*tags, f"marker-pin:{marker.index}"))
            self._pin_items[pin] = marker
            # Dense pins remain visible; their IDs and exact times appear on hover.
            previous = positions[position - 1] if position else -math.inf
            following = positions[position + 1] if position + 1 < len(positions) else math.inf
            label_width = 8 * len(str(marker.index)) + 12
            if min(x - previous, following - x) >= label_width:
                label = self.create_text(x, 106, text=f"#{marker.index}", fill="#875608",
                                         font=("Segoe UI", 8, "bold"), tags=tags)
                self._pin_items[label] = marker

    def _marker_at(self, event):
        candidates = {}
        for item in self.find_overlapping(event.x - 1, event.y - 1,
                                          event.x + 1, event.y + 1):
            marker = self._pin_items.get(item)
            if marker is not None:
                candidates[marker.index] = marker
        # Pins can overlap at this zoom level. Pick the closest source position,
        # so an exact click on an earlier pin is not intercepted by a later one.
        return min(candidates.values(),
                   key=lambda marker: abs(event.x - self.time_to_x(marker.time_ms / 1000)),
                   default=None)

    @staticmethod
    def _marker_timestamp(milliseconds):
        seconds, milliseconds = divmod(int(milliseconds), 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _hover(self, event):
        marker = self._marker_at(event) if self.enabled and not self._drag else None
        if marker is self._hover_marker:
            return
        self._hide_tooltip()
        self._hover_marker = marker
        if marker is not None:
            self._tooltip_job = self.after(300, lambda: self._show_tooltip(marker))

    def _marker_description(self, marker):
        target = next((item for item in self.target_markers if item.index == marker.index), None)
        text = f"#{marker.index}  {self._marker_timestamp(marker.time_ms)}"
        if target is not None:
            text += f" → {self._marker_timestamp(target.time_ms)}"
        return text

    def _show_tooltip(self, marker):
        self._tooltip_job = None
        if marker is not self._hover_marker or not self.winfo_exists():
            return
        self._tooltip = tk.Toplevel(self)
        self._tooltip.overrideredirect(True)
        tk.Label(self._tooltip, text=self._marker_description(marker), background="#fff7df", foreground="#61400b",
                 relief="solid", borderwidth=1, padx=7, pady=4,
                 font=("Segoe UI", 9)).pack()
        self._tooltip.update_idletasks()
        width = self._tooltip.winfo_reqwidth()
        left = self.winfo_rootx() + round(self.time_to_x(marker.time_ms / 1000)) - width // 2
        left = max(0, min(left, self.winfo_screenwidth() - width))
        top = self.winfo_rooty() + 115
        self._tooltip.geometry(f"+{left}+{top}")

    def _hide_tooltip(self):
        if self._tooltip_job is not None:
            self.after_cancel(self._tooltip_job)
            self._tooltip_job = None
        if self._tooltip is not None:
            self._tooltip.destroy()
            self._tooltip = None
        self._hover_marker = None

    def _destroy_tooltip(self, event):
        if event.widget is self:
            self._hide_tooltip()

    def _draw_playhead(self):
        x = self.time_to_x(self.position)
        self.create_line(x, 28, x, 83, fill="#e34b63", width=2, tags="playhead")
        self.create_polygon(x - 5, 26, x + 5, 26, x, 33, fill="#e34b63", tags="playhead")

    def _seek(self, x):
        self.set_position(self.x_to_time(x))
        self.on_seek(self.position)

    def _press(self, event):
        if not self.enabled or not self.duration:
            return
        self.focus_set()
        self._hide_tooltip()
        marker = self._marker_at(event)
        if marker is not None:
            self._drag = ("pin",)
            self.set_position(marker.time_ms / 1000)
            self.on_seek(self.position)
            return
        if not self.editable or event.y < 38 or event.y > 76:
            self._drag = ("seek",)
            self._seek(event.x)
            return
        # Prefer the selected segment at a shared boundary.
        indices = list(range(len(self.segments)))
        if self.selected in indices:
            indices.remove(self.selected)
            indices.insert(0, self.selected)
        for index in indices:
            segment = self.segments[index]
            x1, x2 = self.time_to_x(segment.start), self.time_to_x(segment.end)
            if x1 - 6 <= event.x <= x2 + 6:
                self.selected = index
                self.selection = (segment.start, segment.end)
                mode = "start" if abs(event.x - x1) <= 7 else "end" if abs(event.x - x2) <= 7 else "move"
                self._drag = (mode, index, self.x_to_time(event.x), segment)
                self.on_select(segment.start, segment.end, index)
                self.redraw()
                return
        anchor = self.x_to_time(event.x)
        # A draft is restricted to the empty interval where the drag started.
        low, high = 0.0, self.duration
        for segment in self.segments:
            if segment.end <= anchor:
                low = max(low, segment.end)
            elif segment.start >= anchor:
                high = min(high, segment.start)
        self.selected = None
        self.selection = None
        self._drag = ("select", anchor, low, high)
        self.redraw()

    def _motion(self, event):
        if not self._drag or not self.enabled:
            return
        mode = self._drag[0]
        if mode == "pin":
            return
        if mode == "seek":
            self._seek(event.x)
            return
        if not self.editable:
            return
        value = self.x_to_time(event.x)
        if mode == "select":
            _, anchor, low, high = self._drag
            value = max(low, min(high, value))
            self.selection = (min(anchor, value), max(anchor, value))
            self.redraw()
            return
        _, index, anchor, original = self._drag
        low = max((segment.end for i, segment in enumerate(self.segments)
                   if i != index and segment.end <= original.start), default=0.0)
        high = min((segment.start for i, segment in enumerate(self.segments)
                    if i != index and segment.start >= original.end), default=self.duration)
        minimum = min(0.05, (high - low) / 2)
        if mode == "start":
            start, end = max(low, min(original.end - minimum, value)), original.end
        elif mode == "end":
            start, end = original.start, min(high, max(original.start + minimum, value))
        else:
            delta = max(low - original.start, min(high - original.end, value - anchor))
            start, end = original.start + delta, original.end + delta
        segment = replace(original, start=start, end=end,
                          ramp_seconds=min(original.ramp_seconds, (end - start) / 2))
        self.segments[index] = segment
        self.selection = (start, end)
        self.on_select(start, end, index)
        self.redraw()

    def _release(self, event):
        if not self._drag:
            return
        if not self.enabled or (not self.editable and self._drag[0] not in ("seek", "pin")):
            self._drag = None
            return
        self._motion(event)
        mode = self._drag[0]
        self._drag = None
        if mode == "select":
            if self.selection and self.selection[1] - self.selection[0] >= 0.001:
                self.on_select(*self.selection, None)
            else:
                self.selection = None
                self._seek(event.x)
                self.redraw()
        elif mode in ("start", "end", "move"):
            self.on_change(list(self.segments), self.selected)

    def _step(self, direction):
        if self.enabled and self.duration:
            self.set_position(self.position + direction)
            self.on_seek(self.position)
        return "break"
