"""Fixed timestamp pins and edit locking on the real Tk source timeline."""
from dataclasses import dataclass
from types import SimpleNamespace
import tkinter as tk
import unittest

from apps.video_editor import SpeedSegment
from apps.video_timeline import VideoTimeline


@dataclass(frozen=True)
class Marker:
    index: int
    time_ms: int


class VideoMarkerTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root = tk.Tk()
        except tk.TclError as exc:
            raise unittest.SkipTest(f"Tk display is unavailable: {exc}") from exc
        cls.root.attributes("-alpha", 0.0)
        cls.root.geometry("660x180")

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.seeks, self.selections, self.changes = [], [], []
        self.timeline = VideoTimeline(
            self.root,
            on_seek=self.seeks.append,
            on_select=lambda *args: self.selections.append(args),
            on_change=lambda *args: self.changes.append(args),
        )
        self.timeline.pack(fill="x")
        self.root.update()
        self.addCleanup(self.timeline.destroy)
        self.timeline.set_video(12)
        self.old = (Marker(1, 60), Marker(2, 3130), Marker(3, 6290), Marker(4, 8950))
        self.new = (Marker(1, 133), Marker(2, 2995), Marker(3, 5666), Marker(4, 7666))

    def event(self, seconds, y=92):
        return SimpleNamespace(x=self.timeline.time_to_x(seconds), y=y)

    def pin_x(self, index):
        pin, = self.timeline.find_withtag(f"marker-pin:{index}")
        return self.timeline.coords(pin)[0]

    def test_pins_use_source_duration_and_stay_at_source_times_after_alignment(self):
        self.timeline.set_markers(self.old, self.new)
        left, right = self.timeline._bounds()
        for marker in self.old:
            with self.subTest(marker=marker.index):
                expected = left + (right - left) * (marker.time_ms / 1000) / 12
                self.assertAlmostEqual(self.pin_x(marker.index), expected)
        self.assertNotEqual(self.pin_x(2), self.timeline.time_to_x(self.new[1].time_ms / 1000))

    def test_segment_and_position_redraws_keep_markers_and_video_change_clears_them(self):
        self.timeline.set_markers(self.old, self.new)
        before = [self.pin_x(marker.index) for marker in self.old]
        self.timeline.set_segments([SpeedSegment(1, 3, 2, 0)], selected=0)
        self.timeline.set_position(4)
        self.timeline.redraw()
        self.assertEqual(self.timeline.markers, self.old)
        self.assertEqual(self.timeline.target_markers, self.new)
        self.assertEqual([self.pin_x(marker.index) for marker in self.old], before)
        self.timeline.set_video(20)
        self.assertEqual(self.timeline.markers, ())
        self.assertEqual(self.timeline.target_markers, ())
        self.assertEqual(self.timeline.find_withtag("marker"), ())

    def test_pin_click_seeks_exact_source_milliseconds_and_drag_cannot_move_pin(self):
        segments = [SpeedSegment(1, 4, 2, 0)]
        self.timeline.set_segments(segments)
        self.timeline.set_markers(self.old, self.new)
        original_x = self.pin_x(2)
        self.timeline._press(self.event(3.13))
        self.timeline._motion(self.event(7, y=50))
        self.timeline._release(self.event(9, y=50))
        self.assertEqual(self.seeks, [3.13])
        self.assertEqual(self.timeline.position, 3.13)
        self.assertEqual(self.timeline.markers, self.old)
        self.assertEqual(self.timeline.segments, segments)
        self.assertEqual(self.pin_x(2), original_x)
        self.assertEqual(self.selections, [])
        self.assertEqual(self.changes, [])

    def test_edit_lock_keeps_pins_ruler_keyboard_and_track_seeking_available(self):
        segments = [SpeedSegment(1, 4, 2, 0)]
        self.timeline.set_segments(segments)
        self.timeline.set_markers(self.old, self.new)
        self.timeline.set_editable(False)
        self.timeline._press(self.event(2, y=50))
        self.timeline._motion(self.event(5, y=50))
        self.timeline._release(self.event(5, y=50))
        self.timeline._press(self.event(6, y=15))
        self.timeline._release(self.event(6, y=15))
        self.timeline._step(1)
        self.assertAlmostEqual(self.timeline.position, 7)
        self.timeline._press(self.event(6.29))
        self.timeline._release(self.event(6.29))
        self.assertEqual(self.seeks[-1], 6.29)
        self.assertEqual(self.timeline.segments, segments)
        self.assertEqual(self.selections, [])
        self.assertEqual(self.changes, [])

    def test_locking_during_drag_cancels_the_change_callback(self):
        segments = [SpeedSegment(1, 4, 2, 0)]
        self.timeline.set_segments(segments)
        self.timeline._press(self.event(2, y=50))
        self.assertEqual(self.timeline._drag[0], "move")
        self.timeline.set_editable(False)
        self.timeline._motion(self.event(6, y=50))
        self.timeline._release(self.event(6, y=50))
        self.assertEqual(self.timeline.segments, segments)
        self.assertEqual(self.changes, [])

    def test_disabled_timeline_does_not_seek_or_edit(self):
        self.timeline.set_markers(self.old, self.new)
        self.timeline.enabled = False
        self.timeline._press(self.event(3.13))
        self.timeline._press(self.event(5, y=15))
        self.timeline._step(1)
        self.assertEqual(self.seeks, [])
        self.assertEqual(self.timeline.position, 0)

    def test_hover_description_pairs_ids_and_keeps_millisecond_precision(self):
        self.timeline.set_markers(self.old, tuple(reversed(self.new)))
        self.assertEqual(self.timeline._marker_description(self.old[1]),
                         "#2  00:00:03,130 → 00:00:02,995")
        self.timeline.set_markers(self.old)
        self.assertEqual(self.timeline._marker_description(self.old[0]), "#1  00:00:00,060")

    def test_dense_markers_keep_individually_seekable_pins(self):
        markers = (Marker(12, 1000), Marker(13, 1120))
        self.timeline.set_markers(markers)
        for marker in markers:
            self.assertTrue(self.timeline.find_withtag(f"marker-pin:{marker.index}"))
        self.timeline._press(self.event(1))
        self.timeline._release(self.event(1))
        self.assertEqual(self.seeks, [1])
        self.timeline._press(self.event(1.12))
        self.timeline._release(self.event(1.12))
        self.assertEqual(self.seeks, [1, 1.12])
        self.assertEqual(self.timeline.markers, markers)


if __name__ == "__main__":
    unittest.main()
