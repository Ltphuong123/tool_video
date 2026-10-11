"""Prepare one plain-text dialogue paragraph and matching SRT start markers."""
from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from pathlib import Path
import re
from typing import Callable

from apps.srt_speech import parse_srt


_STAMP = r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
_HEADER = re.compile(rf"^\s*{_STAMP}\s*-->\s*{_STAMP}(?:\s+.*)?$")
_BAD_TEXT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")


@dataclass(frozen=True)
class VietsubPreparation:
    document: Path
    markers: Path
    lines: tuple[str, ...]
    starts_ms: tuple[int, ...]


def _milliseconds(parts):
    hours, minutes, seconds, fraction = parts
    if int(minutes) > 59 or int(seconds) > 59:
        raise ValueError("Phút và giây trong SRT phải từ 00 đến 59.")
    return ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int(fraction.ljust(3, "0"))


def read_vietsub_srt(source: Path, check_stop: Callable = lambda: None):
    """Reuse the speech parser after validating headers and source ordering."""
    check_stop()
    data = source.read_bytes()
    try:
        text = data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
    except UnicodeError as exc:
        raise ValueError("Hãy lưu SRT dưới dạng UTF-8 hoặc UTF-16 có BOM để giữ đúng dấu tiếng Việt.") from exc
    if _BAD_TEXT.search(text):
        raise ValueError("SRT chứa ký tự điều khiển không hợp lệ cho file văn bản.")
    starts = []
    for line in text.splitlines():
        check_stop()
        if "-->" not in line:
            continue
        match = _HEADER.fullmatch(line)
        if match is None:
            raise ValueError(f"Timeline SRT không hợp lệ: {line.strip()}")
        start, end = _milliseconds(match.groups()[:4]), _milliseconds(match.groups()[4:])
        if end <= start:
            raise ValueError("Mốc kết thúc SRT phải sau mốc bắt đầu.")
        if starts and start < starts[-1]:
            raise ValueError("Các câu SRT phải theo thứ tự mốc bắt đầu tăng dần.")
        starts.append(start)
    cues = parse_srt(text, strict_timing=True)
    if not cues:
        raise ValueError("File SRT không có câu phụ đề để xuất.")
    check_stop()
    return tuple(unescape(cue.text) for cue in cues), tuple(cue.start_ms for cue in cues)


def marker_text(starts_ms):
    parts = []
    for index, milliseconds in enumerate(starts_ms, 1):
        seconds, fraction = divmod(milliseconds, 1000)
        minutes, second = divmod(seconds, 60)
        hour, minute = divmod(minutes, 60)
        parts.append(f"{index}\n{hour:02}:{minute:02}:{second:02},{fraction:03}\n")
    return "\n".join(parts)


def dialogue_text(lines):
    """Keep dialogue order, using spaces instead of paragraph/line breaks."""
    return " ".join(" ".join(line.split()) for line in lines)


def prepare_vietsub_files(source, document, markers, *, check_stop=lambda: None,
                          progress=lambda message: None) -> VietsubPreparation:
    source, document, markers = (Path(path).expanduser().resolve() for path in (source, document, markers))
    if not source.is_file() or source.suffix.lower() != ".srt":
        raise ValueError("Hãy chọn một file SRT có sẵn.")
    if document.suffix.lower() != ".txt" or markers.suffix.lower() != ".txt":
        raise ValueError("Cần xuất hai file .txt: lời thoại và mốc bắt đầu.")
    if len({source, document, markers}) != 3 or document.exists() or markers.exists():
        raise ValueError("Chọn hai tên file mới; không ghi đè file đã có.")
    progress("Đang đọc các câu và mốc bắt đầu trong SRT…")
    lines, starts = read_vietsub_srt(source, check_stop)
    owned = []
    try:
        document.parent.mkdir(parents=True, exist_ok=True)
        markers.parent.mkdir(parents=True, exist_ok=True)
        check_stop()
        with document.open("x", encoding="utf-8", newline="\n") as stream:
            owned.append(document)
            progress(f"Đang xuất lời thoại TXT · {len(lines)} câu…")
            check_stop()
            stream.write(dialogue_text(lines))
        check_stop()
        with markers.open("x", encoding="utf-8", newline="\n") as stream:
            owned.append(markers)
            stream.write(marker_text(starts))
        check_stop()
        return VietsubPreparation(document, markers, lines, starts)
    except BaseException:
        for path in owned:
            path.unlink(missing_ok=True)
        raise
