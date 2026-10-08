"""Read numbered time anchors and align a video's source and output timelines."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from apps.video_editor import SpeedSegment


@dataclass(frozen=True)
class TimeMarker:
    """A numbered anchor, stored in integer milliseconds without rounding."""

    index: int
    time_ms: int


@dataclass(frozen=True)
class MarkerTimeMap:
    """Piecewise constant speeds that land exactly on the paired anchors."""

    source_knots: np.ndarray
    output_knots: np.ndarray
    source_duration: float
    segments: tuple[SpeedSegment, ...]
    old_markers: tuple[TimeMarker, ...]
    new_markers: tuple[TimeMarker, ...]
    tail_mode: str = "keep"

    @property
    def output_duration(self) -> float:
        return float(self.output_knots[-1])

    def source_time(self, output_t):
        return np.interp(output_t, self.output_knots, self.source_knots)

    def output_time(self, source_t):
        return np.interp(source_t, self.source_knots, self.output_knots)


_TIMESTAMP = re.compile(r"^([0-9]{2,}):([0-5][0-9]):([0-5][0-9]),([0-9]{3})$")
_INDEX = re.compile(r"^[0-9]+$")


def _validate_markers(markers: Iterable[TimeMarker], name: str) -> tuple[TimeMarker, ...]:
    try:
        values = tuple(markers)
    except TypeError as exc:
        raise ValueError(f"{name} phải là danh sách các mốc thời gian.") from exc
    if not values:
        raise ValueError(f"{name} chưa có mốc thời gian nào.")
    seen = set()
    previous = -1
    for marker in values:
        if not isinstance(marker, TimeMarker):
            raise ValueError(f"{name}: mỗi mốc phải là một TimeMarker.")
        if (not isinstance(marker.index, int) or isinstance(marker.index, bool) or
                marker.index <= 0):
            raise ValueError(f"{name}: số thứ tự mốc phải là số nguyên lớn hơn 0.")
        if (not isinstance(marker.time_ms, int) or isinstance(marker.time_ms, bool) or
                marker.time_ms < 0):
            raise ValueError(f"{name}: thời gian mốc phải là số mili giây nguyên không âm.")
        if marker.index in seen:
            raise ValueError(f"{name}: số thứ tự {marker.index} bị trùng.")
        if marker.time_ms <= previous:
            raise ValueError(f"{name}: thời gian mốc {marker.index} phải tăng dần, không được trùng.")
        seen.add(marker.index)
        previous = marker.time_ms
    return values


def parse_marker_text(text: str) -> tuple[TimeMarker, ...]:
    """Parse ``index\nHH:MM:SS,mmm`` pairs, allowing blank lines and a BOM.

    File order is the chronological order. IDs need not be consecutive, but
    every ID must be unique and every timestamp must be strictly increasing.
    """
    if not isinstance(text, str):
        raise ValueError("Nội dung file mốc phải là văn bản.")
    lines = [(number, line.strip()) for number, line in
             enumerate(text.lstrip("\ufeff").splitlines(), start=1) if line.strip()]
    if not lines:
        raise ValueError("File mốc chưa có mốc thời gian nào.")
    if len(lines) % 2:
        raise ValueError(f"Dòng {lines[-1][0]}: mỗi số thứ tự cần một dòng HH:MM:SS,mmm phía dưới.")
    markers = []
    for offset in range(0, len(lines), 2):
        index_line, index_text = lines[offset]
        time_line, time_text = lines[offset + 1]
        if not _INDEX.fullmatch(index_text) or int(index_text) <= 0:
            raise ValueError(f"Dòng {index_line}: số thứ tự mốc phải là số nguyên lớn hơn 0.")
        match = _TIMESTAMP.fullmatch(time_text)
        if match is None:
            raise ValueError(f"Dòng {time_line}: mốc phải có định dạng HH:MM:SS,mmm "
                             "(phút và giây từ 00 đến 59).")
        hours, minutes, seconds, milliseconds = (int(part) for part in match.groups())
        markers.append(TimeMarker(int(index_text),
                                  ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds))
    return _validate_markers(markers, "File mốc")


def read_marker_file(path) -> tuple[TimeMarker, ...]:
    """Read a UTF-8 marker file, including the common Windows UTF-8 BOM."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise ValueError("Không tìm thấy file mốc thời gian.")
    try:
        return parse_marker_text(source.read_text(encoding="utf-8-sig"))
    except UnicodeError as exc:
        raise ValueError("File mốc cần được lưu ở mã hóa UTF-8.") from exc
    except OSError as exc:
        raise ValueError(f"Không đọc được file mốc thời gian: {exc}") from exc


def validate_marker_pairs(old_markers: Iterable[TimeMarker],
                          new_markers: Iterable[TimeMarker]
                          ) -> tuple[tuple[TimeMarker, ...], tuple[TimeMarker, ...]]:
    """Validate matching anchors without probing a video or assuming a duration."""
    old = _validate_markers(old_markers, "Mốc cũ")
    new = _validate_markers(new_markers, "Mốc mới")
    old_ids = {marker.index for marker in old}
    new_by_id = {marker.index: marker for marker in new}
    if old_ids != set(new_by_id):
        missing = sorted(old_ids - set(new_by_id))
        extra = sorted(set(new_by_id) - old_ids)
        detail = []
        if missing:
            detail.append("thiếu mốc mới " + ", ".join(str(index) for index in missing))
        if extra:
            detail.append("thừa mốc mới " + ", ".join(str(index) for index in extra))
        raise ValueError("Hai file phải có cùng số thứ tự mốc: " + "; ".join(detail) + ".")
    paired_new = tuple(new_by_id[marker.index] for marker in old)
    _validate_markers(paired_new, "Mốc mới theo thứ tự mốc cũ")
    if (old[0].time_ms == 0) != (paired_new[0].time_ms == 0):
        raise ValueError("Mốc đầu bằng 00:00:00,000 phải bằng 0 trong cả hai file. "
                         "Không thể co giãn một đoạn có độ dài bằng 0.")
    return old, paired_new


def build_marker_time_map(duration: float, old_markers: Iterable[TimeMarker],
                          new_markers: Iterable[TimeMarker],
                          tail_mode: str = "keep") -> MarkerTimeMap:
    """Align matching IDs exactly and keep everything after the last anchor at 1x.

    Adjacent anchors determine their interval's speed. A shared origin
    ``(0, 0)`` is added when both first anchors occur after zero. No ramp or
    speed clamp is applied, since either would move the requested anchors.
    """
    try:
        duration = float(duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Video phải có thời lượng hữu hạn lớn hơn 0.") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video phải có thời lượng hữu hạn lớn hơn 0.")
    if tail_mode != "keep":
        raise ValueError("Phần video sau mốc cuối được giữ ở tốc độ 1x.")
    old, paired_new = validate_marker_pairs(old_markers, new_markers)
    source_ms = [marker.time_ms for marker in old]
    output_ms = [marker.time_ms for marker in paired_new]
    try:
        source = np.array([value / 1000 for value in source_ms], dtype=np.float64)
        output = np.array([value / 1000 for value in output_ms], dtype=np.float64)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Mốc thời gian quá lớn để xử lý video.") from exc
    if not np.isfinite(source).all() or not np.isfinite(output).all():
        raise ValueError("Mốc thời gian phải hữu hạn.")
    if source[-1] > duration:
        raise ValueError(f"Mốc cũ {old[-1].index} ({source[-1]:.3f}s) vượt thời lượng video "
                         f"({duration:.3f}s).")
    if source[0] > 0:
        source = np.concatenate(([0.0], source))
        output = np.concatenate(([0.0], output))
        source_ms.insert(0, 0)
        output_ms.insert(0, 0)
    keep_tail = source[-1] < duration
    if keep_tail:
        with np.errstate(over="ignore"):
            output = np.concatenate((output, [output[-1] + (duration - source[-1])]))
        source = np.concatenate((source, [duration]))
    if not np.isfinite(output).all():
        raise ValueError("Thời lượng video đầu ra phải hữu hạn.")
    if (len(source) < 2 or not np.all(np.diff(source) > 0) or
            not np.all(np.diff(output) > 0)):
        raise ValueError("Các mốc thời gian quá gần nhau để tạo đường thời gian hợp lệ.")
    # Millisecond differences preserve exact 1x speeds even when an interval
    # has a large absolute offset. The unchanged tail is explicitly 1x.
    try:
        speeds = [(end - begin) / (output_end - output_begin)
                  for begin, end, output_begin, output_end in
                  zip(source_ms[:-1], source_ms[1:], output_ms[:-1], output_ms[1:])]
    except OverflowError as exc:
        raise ValueError("Các đoạn video phải có tốc độ hữu hạn lớn hơn 0.") from exc
    if keep_tail:
        speeds.append(1.0)
    speeds = np.asarray(speeds, dtype=np.float64)
    if not np.isfinite(speeds).all() or not np.all(speeds > 0):
        raise ValueError("Các đoạn video phải có tốc độ hữu hạn lớn hơn 0.")
    segments = tuple(SpeedSegment(float(begin), float(end), float(speed), 0.0)
                     for begin, end, speed in zip(source[:-1], source[1:], speeds))
    source.setflags(write=False)
    output.setflags(write=False)
    return MarkerTimeMap(source, output, duration, segments, old, paired_new, tail_mode)
