"""Sentence subtitles timed from the audio clips used in the exported track."""
from __future__ import annotations

from dataclasses import dataclass
import re
import numpy as np

from vieneu_utils.core_utils import (
    V3_GAP_SILENCE, edge_silence, pause_pad_samples, split_into_sentences,
)


@dataclass(frozen=True)
class SentenceSpan:
    text: str
    start_sample: int
    end_sample: int


def subtitle_sentences(text: str) -> list[tuple[str, str]]:
    """Split raw paragraphs before normalization, preserving quoted sentences.

    The second value describes the gap BEFORE this sentence.
    """
    result = []
    for paragraph in re.split(r"[\r\n]+", text):
        for index, sentence in enumerate(split_into_sentences(paragraph.strip())):
            result.append((sentence, "para" if index == 0 and result else "sentence"))
    if not result:
        raise ValueError("Hãy nhập văn bản để tạo audio và phụ đề.")
    return result


def assemble_sentence_track(sentences: list[tuple[str, str]], clips: list[np.ndarray],
                            sample_rate: int = 48000) -> tuple[np.ndarray, list[SentenceSpan]]:
    if len(sentences) != len(clips) or not clips:
        raise ValueError("Số câu và số đoạn audio phải khớp nhau.")
    parts, spans = [], []
    cursor = 0
    previous = None
    for (text, gap), audio in zip(sentences, clips):
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not wav.size or not np.isfinite(wav).all():
            raise ValueError(f"Câu '{text}' không có audio hợp lệ.")
        if previous is not None:
            pad = pause_pad_samples(previous, wav, sample_rate, V3_GAP_SILENCE[gap])
            if pad:
                parts.append(np.zeros(pad, dtype=np.float32))
                cursor += pad
        lead, tail = edge_silence(wav, sample_rate)
        if lead >= len(wav):
            # Silent clips still get a valid sentence interval rather than a
            # reversed/empty cue. Never remove samples from the exported audio.
            lead, tail = 0, 0
        spans.append(SentenceSpan(text, cursor + lead, cursor + len(wav) - tail))
        parts.append(wav)
        cursor += len(wav)
        previous = wav
    return np.concatenate(parts), spans


def _timestamp(milliseconds: int) -> str:
    seconds, ms = divmod(milliseconds, 1000)
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02}:{minute:02}:{sec:02},{ms:03}"


def render_srt(spans: list[SentenceSpan], sample_rate: int = 48000) -> str:
    lines = []
    for index, span in enumerate(spans, 1):
        start = (span.start_sample * 1000 + sample_rate // 2) // sample_rate
        end = max(start + 1, (span.end_sample * 1000 + sample_rate // 2) // sample_rate)
        text = re.sub(r"</?en>", "", span.text, flags=re.IGNORECASE)
        for token, label in ((1, "cười"), (2, "thở dài"), (3, "hắng giọng")):
            text = text.replace(f"<|emotion_{token}|>", f"[{label}]")
        text = re.sub(r"\s+", " ", text).strip()
        lines.append(f"{index}\n{_timestamp(start)} --> {_timestamp(end)}\n{text}\n")
    return "\n".join(lines)
