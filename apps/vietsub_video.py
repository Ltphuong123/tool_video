"""Generate independent video/SRT/WAV results and publish one folder atomically."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tempfile
from time import perf_counter
import uuid

from apps.video_markers import TimeMarker, build_marker_time_map, validate_marker_pairs
from apps.vietsub_prepare import read_vietsub_srt
from apps.output_bundles import bundle_folder_name as _folder_name, publish_output_folder as _publish_folder


@dataclass(frozen=True)
class VietsubProgress:
    step: int
    title: str
    detail: str = ""
    fraction: float | None = None


@dataclass(frozen=True)
class VietsubVideo:
    video: Path
    subtitles: Path
    audio: Path
    sentences: int
    duration: float
    elapsed: float


def create_vietsub_video(tool, video, subtitles, voice=None, sampling=None, *,
                         model_settings=None, progress=lambda event: None):
    with tool.operation(require_model=False):
        return _create_vietsub_video(tool, video, subtitles, voice, sampling,
                                    model_settings=model_settings, progress=progress)


def _create_vietsub_video(tool, video, subtitles, voice=None, sampling=None, *,
                          model_settings=None, progress=lambda event: None):
    from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos
    from apps.srt_speech import parse_srt
    from apps.video_copy_export import render_video_copy

    started = perf_counter()
    video, subtitles = (Path(path).expanduser().resolve() for path in (video, subtitles))
    tool.check_stop()
    progress(VietsubProgress(1, "Kiểm tra video và phụ đề"))
    if not video.is_file():
        raise ValueError("Không tìm thấy video gốc.")
    if not subtitles.is_file() or subtitles.suffix.lower() != '.srt':
        raise ValueError("Hãy chọn file SRT gốc còn tồn tại trên máy.")
    lines, starts = read_vietsub_srt(subtitles, tool.check_stop)
    old = tuple(TimeMarker(index, stamp) for index, stamp in enumerate(starts, 1))
    validate_marker_pairs(old, old)
    tool.check_stop()
    info = ffmpeg_parse_infos(str(video))
    if not info.get('video_found'):
        raise ValueError("File đầu vào không có hình video.")
    build_marker_time_map(float(info['duration']), old, old)
    tool.check_stop()
    tool.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.vietsub_', dir=tool.output_dir) as directory:
        working = Path(directory)
        ready = working / 'ready'
        ready.mkdir()
        progress(VietsubProgress(2, "Chuẩn bị giọng đọc"))
        tool.check_stop()
        if tool.tts is None:
            if model_settings is None:
                raise ValueError("Hãy tải model ở Cấu hình trước khi tạo giọng đọc.")
            tool._load_model(**model_settings)
        tool.check_stop()
        progress(VietsubProgress(2, "Giọng đọc đã sẵn sàng", fraction=1))
        progress(VietsubProgress(3, "Tạo giọng đọc và SRT mới", f'{len(lines)} mục phụ đề'))
        tool.check_stop()

        def speech_progress(message):
            tool.check_stop()
            match = re.search(r'(\d+)/(\d+) câu', message)
            if match:
                done, total = map(int, match.groups())
                progress(VietsubProgress(3, "Tạo giọng đọc và SRT mới",
                                         f'Đã đọc {done}/{total} mục', min(.99, done / total)))

        # Use the same speech batching/speed logic directly. Parse source once;
        # no TXT or marker files need to be written and read back.
        speech = tool._synthesize_sentences_with_subtitles(
            [(line, 'sentence') for line in lines], voice, None, sampling, 'wav', speech_progress,
            source_origin_ms=starts[0], subtitle_line_width=70, output_dir=working)
        tool.check_stop()
        cues = parse_srt(speech.subtitles.read_text(encoding='utf-8'), strict_timing=True)
        if len(cues) != len(old):
            raise RuntimeError("Số mục SRT mới không khớp SRT gốc.")
        new = tuple(TimeMarker(index, cue.start_ms) for index, cue in enumerate(cues, 1))
        mapping = build_marker_time_map(float(info['duration']), old, new)
        token = uuid.uuid4().hex[:12]
        name = f'video_{token}'
        destination = ready / f'{name}.mp4'
        progress(VietsubProgress(4, "Căn tốc độ video", f'{len(old)} mốc thời gian', 0))
        render_video_copy(video, destination, mapping, check_stop=tool.check_stop,
                          progress=lambda fraction: progress(VietsubProgress(
                              4, "Căn tốc độ video", f'{len(old)} mốc thời gian', fraction)))
        tool.check_stop()
        progress(VietsubProgress(5, "Lưu video, SRT và giọng đọc", fraction=0))
        speech.audio.rename(ready / f'{name}.wav')
        speech.subtitles.rename(ready / f'{name}.srt')
        folder = _publish_folder(ready, tool.output_dir, _folder_name(video), tool.check_stop)
        # Publication is the commit point: a later stop keeps this finished item.
    progress(VietsubProgress(5, "Hoàn tất", f'{len(lines)} mục phụ đề', 1))
    return VietsubVideo(folder / f'{name}.mp4', folder / f'{name}.srt', folder / f'{name}.wav',
                        len(lines), mapping.output_duration, perf_counter() - started)
