"""Reusable operations for the v3 Turbo desktop app; no GUI or model at import."""
from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterator
import json
import math
import os
import re
import tempfile
from time import perf_counter
import threading
import uuid

import numpy as np
import soundfile as sf

from apps.speech_speed import require_rubberband, stretch_rubberband
from apps.simple_tts_engine import generate_speech
from apps.srt_speech import concatenate, lay_on_timeline, parse_srt, wrap_srt_text
from apps.srt_timing import (
    cue_sample_windows, prepare_srt_clip,
)
from apps.speech_subtitles import assemble_sentence_track, render_srt, subtitle_sentences
from apps.user_voices import (
    _entry_from_json,
)
from apps.vietsub_prepare import VietsubPreparation
from apps.output_bundles import bundle_folder_name, publish_output_folder
from apps.speech_pipeline import processed_audio_batches

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
STREAM_BUFFER_BYTES = 8 * 1024 * 1024
OUTPUT_NAME = re.compile(
    r"^(speech(?:_subtitled)?|stream|conversation|srt|denoised|microphone|"
    r"batch(?:_\d{4,})?|voices|reference|video|vietsub_text|vietsub_marks|vietsub_new_marks)_[0-9a-f]{12}\.([a-z0-9]+)$", re.IGNORECASE,
)
OUTPUT_KINDS = {"wav": "Audio", "flac": "Audio", "mp3": "Audio", "srt": "Phụ đề",
                "zip": "Batch ZIP", "json": "Giọng đã xuất", "npz": "Embedding / codes", "mp4": "Video",
                "docx": "Tài liệu Word", "txt": "Mốc thời gian"}


class Cancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class SubtitledSpeech:
    audio: Path
    subtitles: Path
    sentences: int
    duration: float
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class VietsubSpeech:
    preparation: VietsubPreparation
    speech: SubtitledSpeech
    markers: Path
    starts_ms: tuple[int, ...]


@dataclass(frozen=True)
class OutputFile:
    path: Path
    kind: str
    size: int
    modified: float


@dataclass
class Sampling:
    speed: float = 1.0
    temperature: float = 0.8
    top_k: int = 25
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    repetition_window: int = 64
    max_new_frames: int = 300
    max_chars: int = 256
    batch_size: int = 8
    use_ref_codes: bool = True

    def kwargs(self, streaming: bool = False) -> dict:
        ranges = {
            "speed": (0.5, 2),
            "temperature": (0, 2), "top_k": (1, 1024), "top_p": (0.00001, 1),
            "repetition_penalty": (1, 2), "repetition_window": (1, 4096),
            "max_new_frames": (1, 512), "max_chars": (64, 512), "batch_size": (1, 32),
        }
        values = asdict(self)
        integer_fields = {"top_k", "repetition_window", "max_new_frames", "max_chars", "batch_size"}
        for key, (lo, hi) in ranges.items():
            value = values[key]
            if not math.isfinite(float(value)) or not lo <= value <= hi:
                raise ValueError(f"{key} phải nằm trong [{lo}, {hi}].")
            if key in integer_fields and (not isinstance(value, (int, np.integer)) or isinstance(value, bool)):
                raise ValueError(f"{key} phải là số nguyên.")
        if streaming:
            values.pop("batch_size")
        # v3 Turbo ignores speed; the desktop tool stretches the resulting audio.
        values.pop("speed")
        values.update(denoise=False, apply_watermark=False)
        return values


def read_document(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        try:
            import fitz
        except ImportError as exc:
            raise ValueError("Đọc PDF cần PyMuPDF: uv sync --extra pdf") from exc
        with fitz.open(path) as doc:
            text = "\n\n".join(page.get_text() for page in doc)
    else:
        text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError("File không có văn bản; PDF scan cần OCR trước.")
    return text


class TurboTool:
    def __init__(self, output_dir: str | Path | None = None, factory=None,
                 settings_path: str | Path | None = None):
        self.settings_path = Path(settings_path).expanduser().resolve() if settings_path else None
        self.settings_error = None
        settings = self._read_settings()
        initial = Path(output_dir or settings.get("output_dir") or
                       ROOT / "outputs" / "v3turbo_desktop").expanduser().resolve()
        self.output_dir = initial
        # Changing the destination must not lose the user's saved cloned voices.
        self.voices_path = Path(settings.get("voices_path") or
                                initial / "user_voices.json").expanduser().resolve()
        previous = [Path(path).expanduser().resolve() for path in settings.get("output_dirs", [])]
        self.output_dirs = tuple(dict.fromkeys([*previous, initial]))
        self.tts = None
        self._factory = factory
        self._model_settings = None
        self._lock = threading.Lock()
        self.stop_event = threading.Event()

    def _read_settings(self) -> dict:
        if self.settings_path is None:
            return {}
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8-sig"))
            if (not isinstance(data, dict) or data.get("version") != 1 or
                    not isinstance(data.get("output_dir"), str) or not data["output_dir"].strip() or
                    not isinstance(data.get("output_dirs"), list) or
                    any(not isinstance(path, str) or not path.strip() for path in data["output_dirs"]) or
                    not isinstance(data.get("voices_path"), str) or
                    Path(data["voices_path"]).name != "user_voices.json"):
                raise ValueError("Cấu hình thư mục lưu không hợp lệ.")
            return data
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            self.settings_error = f"Không đọc được cài đặt nơi lưu; dùng thư mục mặc định: {exc}"
            return {}

    def _write_settings(self, output_dir: Path, output_dirs: tuple[Path, ...]):
        if self.settings_path is None:
            return
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "output_dir": str(output_dir),
                "output_dirs": [str(path) for path in output_dirs],
                "voices_path": str(self.voices_path)}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                             dir=self.settings_path.parent, delete=False) as file:
                temporary = Path(file.name)
                json.dump(data, file, ensure_ascii=False, indent=2)
                file.write("\n")
            temporary.replace(self.settings_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def set_output_dir(self, path: str | Path) -> Path:
        """Use a writable destination for future results; keep voices and old files."""
        if not str(path).strip():
            raise ValueError("Hãy chọn thư mục lưu kết quả.")
        with self.operation(require_model=False):
            directory = Path(path).expanduser().resolve()
            try:
                directory.mkdir(parents=True, exist_ok=True)
                # Probe the actual permission rather than relying on os.access.
                with tempfile.TemporaryFile(dir=directory):
                    pass
            except OSError as exc:
                raise ValueError(f"Không thể ghi vào thư mục đã chọn: {directory}") from exc
            directories = tuple(dict.fromkeys([*self.output_dirs, directory]))
            self._write_settings(directory, directories)
            self.output_dir, self.output_dirs = directory, directories
            self.settings_error = None
            return directory

    @staticmethod
    def _output_kind(path: Path, *, bundle=False) -> str | None:
        match = OUTPUT_NAME.fullmatch(path.name)
        if match is None:
            return None
        label, suffix = (part.lower() for part in match.groups())
        if bundle and label == 'video' and suffix == 'wav':
            return 'Audio'
        if suffix in ("wav", "flac", "mp3"):
            return "Audio" if label not in ("voices", "reference", "video", "vietsub_text", "vietsub_marks", "vietsub_new_marks") else None
        if suffix == "txt" and label == "vietsub_text":
            return "Lời thoại"
        if suffix == "txt" and label == "vietsub_new_marks":
            return "Mốc thời gian mới"
        if suffix == "srt" and label in ("video", "srt"):
            return "Phụ đề"
        expected = {"srt": "speech_subtitled", "zip": "batch",
                    "json": "voices", "npz": "reference", "mp4": "video",
                    "docx": "vietsub_text", "txt": "vietsub_marks"}
        return OUTPUT_KINDS.get(suffix) if expected.get(suffix) == label else None

    def list_outputs(self) -> list[OutputFile]:
        """Read history from generated files, including previously used folders."""
        results = []
        seen = set()
        for directory in self.output_dirs:
            try:
                files = list(directory.iterdir())
            except OSError:
                continue
            # Video/SRT bundles are exactly one level below the output root.
            # Ignore hidden staging folders, links and unrelated nested files.
            nested = []
            for folder in files:
                try:
                    if folder.name.startswith('.') or folder.is_symlink() or not folder.is_dir():
                        continue
                    if folder.resolve().parent != directory:
                        continue
                    nested.extend(path for path in folder.iterdir()
                                  if path.name.startswith(('video_', 'srt_')) and self._output_kind(path, bundle=True))
                except OSError:
                    continue
            files.extend(nested)
            for path in files:
                kind = self._output_kind(path, bundle=path.parent != directory)
                if kind is None or path.is_symlink():
                    continue
                try:
                    canonical = path.resolve()
                    parent = canonical.parent
                    if canonical in seen:
                        continue
                    if not path.is_file() or (parent != directory and
                                             (path.parent.is_symlink() or parent.parent != directory)):
                        continue
                    info = path.stat()
                except OSError:
                    continue
                seen.add(canonical)
                results.append(OutputFile(path, kind, info.st_size, info.st_mtime))
        return sorted(results, key=lambda item: (item.modified, item.path.name), reverse=True)

    def delete_outputs(self, paths: list[str | Path], include_subtitles: bool = True) -> list[Path]:
        """Delete only selected generated files after validating the entire selection."""
        if not paths:
            raise ValueError("Hãy chọn file muốn xóa trong lịch sử.")
        with self.operation(require_model=False):
            allowed = {item.path.resolve(): item.path for item in self.list_outputs()}
            selected = []
            for value in paths:
                path = Path(value).expanduser()
                canonical = path.resolve()
                if path.is_symlink() or canonical not in allowed:
                    raise ValueError(f"Chỉ xóa được file do tool tạo trong lịch sử: {path.name}")
                selected.append(allowed[canonical])
                if include_subtitles and path.suffix.lower() in (".wav", ".flac", ".mp3", ".mp4"):
                    subtitle = canonical.with_suffix(".srt")
                    if subtitle in allowed:
                        selected.append(allowed[subtitle])
            selected = list(dict.fromkeys(selected))
            # Check again before the first mutation, protecting changed directories.
            for path in selected:
                parent = path.resolve().parent
                if (path.is_symlink() or path.parent.is_symlink() or not path.is_file() or
                        (parent not in self.output_dirs and parent.parent not in self.output_dirs)):
                    raise ValueError(f"File đã thay đổi; hãy làm mới lịch sử: {path.name}")
            for path in selected:
                path.unlink()
            return selected

    @contextmanager
    def operation(self, require_model=True):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Đang có tác vụ khác chạy. Hãy dừng hoặc chờ hoàn tất.")
        try:
            self.stop_event.clear()
            if require_model and self.tts is None:
                raise ValueError("Hãy tải model ở tab Cấu hình trước.")
            yield
        finally:
            self._lock.release()

    def stop(self):
        self.stop_event.set()

    def check_stop(self):
        if self.stop_event.is_set():
            raise Cancelled("Đã dừng tác vụ.")

    def load(self, **settings) -> str:
        with self.operation(require_model=False):
            return self._load_model(**settings)

    def _load_model(self, **settings) -> str:
        if self.tts is not None and self._model_settings == settings:
            return f"v3 Turbo ready: {self.tts.backend} (cached)"
        if self.tts is not None:
            self.tts.close()
            self.tts = None
        factory = self._factory
        if factory is None:
            from vieneu import Vieneu
            factory = Vieneu
        model = factory(mode="v3turbo", **settings)
        try:
            if self.voices_path.is_file():
                data = json.loads(self.voices_path.read_text(encoding="utf-8"))
                for name, entry in data.get("presets", {}).items():
                    if model.resolve_voice_name(name) is None:
                        value = _entry_from_json(model, entry)
                        if value is not None:
                            model._preset_voices[name] = value
        except Exception:
            model.close()
            raise
        self.tts = model
        self._model_settings = dict(settings)
        return f"Đã tải v3 Turbo · {model.backend} · 48 kHz · {len(model._preset_voices)} giọng"

    def unload(self):
        with self.operation(require_model=False):
            if self.tts is not None:
                self.tts.close()
                self.tts = None

    def voice_names(self) -> list[str]:
        return [name for _, name in self.tts.list_preset_voices()] if self.tts else []

    def _reference(self, voice: str | None, ref_audio: str | None):
        if ref_audio:
            if not Path(ref_audio).is_file():
                raise ValueError("Không tìm thấy audio tham chiếu.")
            return {"ref_audio": ref_audio}
        if voice and self.tts.resolve_voice_name(voice) is None:
            raise ValueError(f"Giọng '{voice}' không tồn tại.")
        return {"voice": voice or None}

    def _sampling(self, sampling: Sampling, streaming=False):
        kw = sampling.kwargs(streaming)
        if sampling.speed != 1.0:
            require_rubberband()
        return kw

    def _adjust_speed(self, audio, sampling: Sampling) -> np.ndarray:
        self.check_stop()
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sampling.speed != 1.0 and wav.size:
            wav = stretch_rubberband(wav, sampling.speed, check_stop=self.check_stop)
            self.check_stop()
        return wav

    def _path(self, label: str, suffix: str, directory=None) -> Path:
        directory = Path(directory) if directory is not None else self.output_dir
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{label}_{uuid.uuid4().hex[:12]}{suffix}"

    def _save(self, audio, fmt="wav", rate=48000, label="speech", output_dir=None) -> Path:
        fmt = fmt.lower()
        if fmt not in ("wav", "flac", "mp3"):
            raise ValueError("Định dạng phải là WAV, FLAC hoặc MP3.")
        if fmt == "mp3" and "MP3" not in sf.available_formats():
            raise ValueError("libsndfile trên máy này chưa hỗ trợ MP3. Hãy chọn WAV hoặc FLAC.")
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not wav.size or not np.isfinite(wav).all():
            raise ValueError("Model không trả về audio hợp lệ.")
        path = self._path(label, f".{fmt}", directory=output_dir)
        try:
            sf.write(path, wav, rate, format=fmt.upper(), **({"subtype": "PCM_16"} if fmt != "mp3" else {}))
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path

    def _generate_audio(self, text, reference, kwargs, *, batch=False):
        self.check_stop()
        result = generate_speech(text, model=self.tts, save=False, batch=batch,
                                 inference_kwargs=kwargs, **reference)
        self.check_stop()
        results = result if batch else [result]
        if any(item.sample_rate != 48000 for item in results):
            raise ValueError("Tool desktop cần audio Turbo 48 kHz.")
        return [item.audio for item in results] if batch else result.audio

    def _generated_audio_batches(self, texts, reference, kwargs, window):
        for start in range(0, len(texts), window):
            self.check_stop()
            group = texts[start:start + window]
            started = perf_counter()
            wavs = self._generate_audio(group, reference, kwargs, batch=True)
            if len(wavs) != len(group):
                raise RuntimeError("Số audio trả về không khớp số câu.")
            yield start, wavs, perf_counter() - started

    def _cuda_tempo_overlap(self):
        device = getattr(getattr(self.tts, "engine", None), "device", None)
        return self.tts.backend == "pytorch" and getattr(device, "type", None) == "cuda"

    def synthesize(self, text: str, voice=None, ref_audio=None, sampling=None, fmt="wav") -> Path:
        if not text.strip():
            raise ValueError("Hãy nhập văn bản.")
        sampling = sampling or Sampling()
        with self.operation():
            kw = self._sampling(sampling)
            ref = self._reference(voice, ref_audio)
            audio = self._generate_audio(text, ref, kw)
            self.check_stop()
            return self._save(self._adjust_speed(audio, sampling), fmt)


    def synthesize_many(self, texts, voice=None, sampling=None, fmt="wav"):
        """Queue batch: use one SDK batch, exporting ordinary speech files."""
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Mỗi mục phải có văn bản.")
        sampling = sampling or Sampling()
        with self.operation():
            audio = self._generate_audio(texts, self._reference(voice, None), self._sampling(sampling), batch=True)
            paths = []
            for wav in audio:
                self.check_stop()
                paths.append(self._save(self._adjust_speed(wav, sampling), fmt))
            return paths

    def synthesize_with_subtitles(self, text: str, voice=None, ref_audio=None, sampling=None,
                                  fmt="wav", progress: Callable = lambda message: None) -> SubtitledSpeech:
        with self.operation():
            return self._synthesize_with_subtitles(text, voice, ref_audio, sampling, fmt, progress)

    def _synthesize_with_subtitles(self, text: str, voice=None, ref_audio=None, sampling=None,
                                  fmt="wav", progress: Callable = lambda message: None) -> SubtitledSpeech:
        return self._synthesize_sentences_with_subtitles(subtitle_sentences(text), voice, ref_audio,
                                                        sampling, fmt, progress)

    def _synthesize_sentences_with_subtitles(self, sentences, voice=None, ref_audio=None, sampling=None,
                                            fmt="wav", progress: Callable = lambda message: None,
                                            *, source_origin_ms=None, subtitle_line_width=None,
                                            output_dir=None) -> SubtitledSpeech:
        started = perf_counter()
        model_seconds = 0.0
        tempo_seconds = 0.0
        if not sentences or any(not text.strip() for text, gap in sentences):
            raise ValueError("Mỗi mục phụ đề phải có lời thoại để tạo audio và SRT.")
        sampling = sampling or Sampling()
        kw = self._sampling(sampling)
        ref = self._reference(voice, ref_audio)
        clips = []
        window = min(128, sampling.batch_size * 4) if self.tts.backend == "pytorch" else sampling.batch_size
        texts = [sentence for sentence, gap in sentences]
        overlap = (sampling.speed != 1.0 and len(sentences) > window and self._cuda_tempo_overlap())
        def workers(start, wavs):
            return min(4, len(wavs), os.cpu_count() or 1) if sampling.speed != 1.0 else 1
        with closing(processed_audio_batches(
                self._generated_audio_batches(texts, ref, kw, window),
                lambda index, wav: self._adjust_speed(wav, sampling), workers,
                overlap=overlap, check_stop=self.check_stop)) as batches:
            for batch in batches:
                model_seconds += batch.model_seconds
                clips.extend(batch.results)
                tempo_seconds += batch.processing_seconds
                progress(f"Audio + SRT: đã đọc {len(clips)}/{len(sentences)} câu")
        self.check_stop()
        assembly_started = perf_counter()
        track, spans = assemble_sentence_track(sentences, clips)
        if source_origin_ms is not None:
            first_ms = (spans[0].start_sample * 1000 + 24000) // 48000
            if source_origin_ms > 0 and first_ms == 0:
                # An interval cannot shrink to zero without losing video frames.
                # Preserve the source intro and shift both audio and subtitles.
                pad = source_origin_ms * 48
                track = np.pad(track, (pad, 0))
                spans = [replace(span, start_sample=span.start_sample + pad,
                                 end_sample=span.end_sample + pad) for span in spans]
            elif source_origin_ms == 0 and first_ms > 0:
                # Include model lead-in silence in the first cue at the shared
                # zero origin; do not cut or modify the generated waveform.
                spans[0] = replace(spans[0], start_sample=0)
        assembly_seconds = perf_counter() - assembly_started
        del clips
        self.check_stop()
        srt_text = render_srt(spans, line_width=subtitle_line_width)
        audio_path = subtitle_path = None
        export_started = perf_counter()
        try:
            audio_path = self._save(track, fmt, label="speech_subtitled", **({"output_dir": output_dir} if output_dir is not None else {}))
            subtitle_path = audio_path.with_suffix(".srt")
            subtitle_path.write_text(srt_text, encoding="utf-8")
            self.check_stop()
        except BaseException:
            if audio_path is not None:
                audio_path.unlink(missing_ok=True)
            if subtitle_path is not None:
                subtitle_path.unlink(missing_ok=True)
            raise
        export_seconds = perf_counter() - export_started
        elapsed = perf_counter() - started
        return SubtitledSpeech(audio_path, subtitle_path, len(sentences), len(track) / 48000,
                               {"total": elapsed, "model": model_seconds,
                                "tempo": tempo_seconds, "assembly": assembly_seconds,
                                "export": export_seconds,
                                "other": max(0.0, elapsed - model_seconds)})

    def stream(self, text: str, voice=None, ref_audio=None, sampling=None,
               on_audio: Callable | None = None) -> Iterator[tuple[Path | None, str]]:
        if not text.strip():
            raise ValueError("Hãy nhập văn bản.")
        sampling = sampling or Sampling()
        with self.operation():
            kw = self._sampling(sampling, streaming=True)
            ref = self._reference(voice, ref_audio)
            path = self._path("stream", ".wav")
            iterator = self.tts.infer_stream(text, **ref, **kw)
            frames = 0
            buffered = sampling.speed != 1.0
            mode = "Đang tạo audio để chỉnh tốc độ" if buffered else "Đang streaming"
            try:
                # Raw float bytes avoid encoding/decoding a temporary WAV. Large
                # sources spill automatically, so buffering does not grow forever.
                destination = (tempfile.SpooledTemporaryFile(max_size=STREAM_BUFFER_BYTES, mode="w+b")
                               if buffered else sf.SoundFile(path, mode="w", samplerate=48000,
                                                              channels=1, subtype="PCM_16"))
                with destination as file:
                    for chunk in iterator:
                        self.check_stop()
                        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
                        if not chunk.size:
                            continue
                        if not np.isfinite(chunk).all():
                            raise ValueError("Model trả về audio không hữu hạn.")
                        file.write(chunk.tobytes() if buffered else chunk)
                        frames += len(chunk)
                        if on_audio and not buffered:
                            on_audio(chunk, 48000)
                        yield None, f"{mode} · đã sinh {frames / 48000:.1f} giây audio"
                    if buffered and frames:
                        file.seek(0)
                        original = np.frombuffer(file.read(), dtype=np.float32)
                if not frames:
                    raise ValueError("Model không trả về audio.")
                self.check_stop()
                if buffered:
                    yield None, f"Đang chỉnh tốc độ {sampling.speed:g}x, giữ cao độ…"
                    adjusted = self._adjust_speed(original, sampling)
                    del original
                    sf.write(path, adjusted, 48000, subtype="PCM_16")
                    frames = len(adjusted)
                    if on_audio:
                        for start in range(0, frames, 9600):
                            self.check_stop()
                            on_audio(adjusted[start:start + 9600], 48000)
                            yield None, f"Đang phát audio ở tốc độ {sampling.speed:g}x"
                self.check_stop()
                yield path, f"Hoàn tất · {frames / 48000:.1f} giây audio · 48 kHz · {sampling.speed:g}x"
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            finally:
                close = getattr(iterator, "close", None)
                if close:
                    close()


    def srt(self, path, voice=None, sampling=None, keep_timing=True, fmt="wav",
            progress: Callable = lambda message: None, fit_to_timing=None, min_speed=None):
        started = perf_counter()
        model_seconds = 0.0
        processing_seconds = 0.0
        if fit_to_timing is None:
            fit_to_timing = bool(keep_timing)
        if fit_to_timing and not keep_timing:
            raise ValueError("Tự tăng tốc theo SRT cần bật giữ mốc bắt đầu phụ đề.")
        source_path = Path(path).expanduser().resolve()
        source_srt = source_path.read_text(encoding="utf-8-sig")
        cues = parse_srt(source_srt, strict_timing=fit_to_timing)
        if not cues:
            raise ValueError("File SRT không chứa lời thoại hợp lệ.")
        windows = cue_sample_windows(cues) if fit_to_timing else []
        if fit_to_timing:
            progress(f"SRT: đã đọc {len(cues)} câu; giữ mốc tuyệt đối và mọi khoảng trống, "
                     f"tổng thời lượng {max(cue.end_ms for cue in cues) / 1000:.3f}s.")
        sampling = sampling or Sampling()
        if fit_to_timing:
            require_rubberband()
        if min_speed is not None:
            if not math.isfinite(float(min_speed)) or not 0.5 <= min_speed <= 2:
                raise ValueError("Tốc độ ban đầu / tối thiểu SRT phải nằm trong 0.5–2.0x.")
            sampling = replace(sampling, speed=float(min_speed))
        with self.operation():
            kw = self._sampling(sampling)
            if fit_to_timing:
                progress(f"Bộ chỉnh tốc độ SRT: rubberband · tốc độ ban đầu {sampling.speed:.2f}x.")
            ref = self._reference(voice, None)
            clips = []
            track = (np.zeros(max(cue.end_ms for cue in cues) * 48000 // 1000, dtype=np.float32)
                     if fit_to_timing else None)
            accelerated, max_speed = 0, sampling.speed
            window = min(128, sampling.batch_size * 4) if self.tts.backend == "pytorch" else sampling.batch_size
            overlap = (len(cues) > window and self._cuda_tempo_overlap() and
                       (fit_to_timing or sampling.speed != 1.0))

            def prepare(index, wav):
                if fit_to_timing:
                    begin, end = windows[index]
                    return prepare_srt_clip(np.asarray(wav, dtype=np.float32).reshape(-1),
                                            end - begin, sampling.speed,
                                            check_stop=self.check_stop)
                return self._adjust_speed(wav, sampling), sampling.speed

            def workers(start, wavs):
                # CUDA lookahead uses a fixed CPU pool; ordinary CPU/ONNX
                # requests avoid a pool when all clips already fit at 1x.
                needs_tempo = sampling.speed != 1.0 or (fit_to_timing and any(
                    np.asarray(wav).size > windows[start + offset][1] - windows[start + offset][0]
                    for offset, wav in enumerate(wavs)))
                return min(4, len(wavs), os.cpu_count() or 1) if overlap or needs_tempo else 1

            with closing(processed_audio_batches(
                    self._generated_audio_batches([cue.text for cue in cues], ref, kw, window),
                    prepare, workers, overlap=overlap, check_stop=self.check_stop)) as batches:
                for batch in batches:
                    start = batch.start
                    group = cues[start:start + window]
                    model_seconds += batch.model_seconds
                    for offset, (audio, speed) in enumerate(batch.results):
                        if fit_to_timing:
                            cue = group[offset]
                            begin, end = windows[start + offset]
                            if speed > sampling.speed:
                                accelerated += 1
                            max_speed = max(max_speed, speed)
                            progress(f"SRT câu {cue.index}: bắt đầu {begin / 48000:.3f}s, "
                                     f"kết thúc {(begin + len(audio)) / 48000:.3f}s / giới hạn {end / 48000:.3f}s, "
                                     f"tốc độ {speed:.2f}x")
                            if len(audio) > end - begin:
                                raise ValueError(f"Câu SRT {cue.index}: audio chưa vừa khung thời gian.")
                            track[begin:begin + len(audio)] = audio
                        else:
                            clips.append(audio)
                    processing_seconds += batch.processing_seconds
                    progress(f"SRT {start + len(group)}/{len(cues)} câu")
            self.check_stop()
            if fit_to_timing:
                note = (f"{accelerated} câu tăng tốc để vừa khung SRT; tốc độ ban đầu {sampling.speed:.2f}x, "
                        f"cao nhất {max_speed:.2f}x. Giữ mốc bắt đầu, không lùi câu.")
            elif keep_timing:
                track, pushed, shift = lay_on_timeline(clips, cues, 48000)
                note = f"{pushed} câu lùi mốc; mức lùi lớn nhất {shift:.2f}s."
            else:
                track = concatenate(clips, 48000)
                note = "Các câu nối tiếp, nghỉ 0,5 giây."
            if not np.any(np.abs(track) >= 1 / 32768):
                raise ValueError("Audio SRT không có tín hiệu nghe được; không xuất file im lặng.")
            note += f" Câu đầu ở {cues[0].start_ms / 1000:.3f}s; phần trước là khoảng lặng theo mốc SRT."
            export_started = perf_counter()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.srt_', dir=self.output_dir) as directory:
                ready = Path(directory) / 'ready'
                ready.mkdir()
                output = self._save(track, fmt, label='srt', output_dir=ready)
                self.check_stop()
                subtitle = output.with_suffix('.srt')
                with subtitle.open('x', encoding='utf-8', newline='\n') as stream:
                    stream.write(wrap_srt_text(source_srt, width=70))
                self.check_stop()
                folder = publish_output_folder(ready, self.output_dir,
                                               bundle_folder_name(source_path, fallback='SRT'), self.check_stop)
                output, subtitle = folder / output.name, folder / subtitle.name
            note += f" SRT mới: {subtitle.name}; mỗi dòng tối đa 70 ký tự, giữ timeline gốc. Thư mục: {folder.name}."
            note += (f" Total={perf_counter() - started:.2f}s; model={model_seconds:.2f}s; "
                     f"tempo={processing_seconds:.2f}s; export={perf_counter() - export_started:.2f}s.")
            return output, note


    def prepare_vietsub(self, path, progress=lambda message: None):
        with self.operation(require_model=False):
            return self._prepare_vietsub(path, progress)

    def _prepare_vietsub(self, path, progress=lambda message: None, output_dir=None):
        """Export a dialogue paragraph and matching cue starts without TTS."""
        from apps.vietsub_prepare import prepare_vietsub_files

        document = self._path("vietsub_text", ".txt", directory=output_dir)
        markers = document.with_name(document.name.replace("vietsub_text_", "vietsub_marks_", 1)).with_suffix(".txt")
        result = prepare_vietsub_files(path, document, markers,
                                      check_stop=self.check_stop, progress=progress)
        try:
            self.check_stop()
        except BaseException:
            document.unlink(missing_ok=True)
            markers.unlink(missing_ok=True)
            raise
        return result

    def prepare_vietsub_with_speech(self, path, voice=None, sampling=None, fmt="wav", *,
                                   model_settings=None, progress=lambda message: None,
                                   on_prepared=lambda result: None) -> VietsubSpeech:
        with self.operation(require_model=False):
            return self._prepare_vietsub_with_speech(path, voice, sampling, fmt, model_settings=model_settings,
                                                     progress=progress, on_prepared=on_prepared)

    def _prepare_vietsub_with_speech(self, path, voice=None, sampling=None, fmt="wav", *,
                                   model_settings=None, progress=lambda message: None,
                                   on_prepared=lambda result: None, output_dir=None) -> VietsubSpeech:
        """Keep preparation, model load, cue speech and new markers in one job.

        Completed TXT/markers survive a later TTS error or cancellation.
        Each source cue maps to one new cue, using the text speech workflow.
        """
        from apps.vietsub_prepare import marker_text

        prepared = self._prepare_vietsub(path, progress, output_dir)
        on_prepared(prepared)
        self.check_stop()
        if self.tts is None:
            if model_settings is None:
                raise ValueError("Đã xuất TXT và mốc. Hãy tải model ở tab Cấu hình trước khi đọc văn bản.")
            progress("Đã xuất TXT và mốc; đang tải model đọc văn bản…")
            self._load_model(**model_settings)
            self.check_stop()
        progress("Đã xuất mốc gốc; đang đọc từng mục SRT để tạo audio + SRT mới…")
        sentences = [(line, "sentence") for line in prepared.lines]
        speech = self._synthesize_sentences_with_subtitles(sentences, voice, None, sampling, fmt, progress,
                                                          source_origin_ms=prepared.starts_ms[0],
                                                          subtitle_line_width=70, output_dir=output_dir)
        markers = prepared.markers.with_name(prepared.markers.name.replace("vietsub_marks_", "vietsub_new_marks_", 1))
        owned_markers = False
        try:
            self.check_stop()
            # Read rounded timestamps from the actual exported SRT so its
            # marker list agrees exactly with that file, one cue per input.
            cues = parse_srt(speech.subtitles.read_text(encoding="utf-8"), strict_timing=True)
            if len(cues) != len(prepared.lines):
                raise RuntimeError("Số mục SRT mới không khớp SRT gốc.")
            starts = tuple(cue.start_ms for cue in cues)
            progress("Đang xuất danh sách mốc bắt đầu của SRT mới…")
            self.check_stop()
            with markers.open("x", encoding="utf-8", newline="\n") as stream:
                owned_markers = True
                stream.write(marker_text(starts))
            self.check_stop()
        except BaseException:
            speech.audio.unlink(missing_ok=True)
            speech.subtitles.unlink(missing_ok=True)
            if owned_markers:
                markers.unlink(missing_ok=True)
            raise
        return VietsubSpeech(prepared, speech, markers, starts)

    def edit_video(self, path, start, end, speed=1.5, ramp_seconds=0.5, keep_audio=False,
                   quality=20, preset="fast", progress: Callable = lambda message: None,
                   export_mode="encode"):
        values = (start, end, speed, ramp_seconds)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Mốc thời gian và tốc độ video phải là số hữu hạn.")
        if start < 0 or end <= start:
            raise ValueError("Mốc kết thúc phải sau mốc bắt đầu và mốc bắt đầu không âm.")
        if not 0.25 <= speed <= 4 or not 0 <= ramp_seconds <= (end - start) / 2:
            raise ValueError("Tốc độ video cần 0.25–4.0x; thời gian chuyển không vượt nửa đoạn đã chọn.")
        if export_mode not in ("copy", "encode"):
            raise ValueError("Chế độ xuất video không hợp lệ.")
        if export_mode == "encode" and (not isinstance(quality, int) or isinstance(quality, bool) or not 18 <= quality <= 28):
            raise ValueError("Chất lượng video CRF phải là số nguyên 18–28.")
        if export_mode == "encode" and preset not in ("ultrafast", "veryfast", "faster", "fast", "medium"):
            raise ValueError("Cấu hình xuất video không hợp lệ.")
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ValueError("Không tìm thấy video đầu vào.")
        with self.operation(require_model=False):
            from apps.video_editor import render_speed_segment
            destination = self._path("video", ".mp4")
            try:
                result = render_speed_segment(source, destination, start, end, speed=speed,
                                              ramp_seconds=ramp_seconds, keep_audio=keep_audio,
                                              quality=quality, preset=preset, check_stop=self.check_stop,
                                              progress=progress, export_mode=export_mode)
                self.check_stop()
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
            note = f"{start:.2f}–{end:.2f}s · {speed:g}x · chuyển tốc độ {ramp_seconds:g}s."
            note = ("Giữ nguyên hình · " if export_mode == "copy" else "Mã hóa lại · ") + note
            if result is not None:
                note += f" Video xuất {result.output_duration:.2f}s."
            return destination, note

    def create_vietsub_video(self, video, subtitles, voice=None, sampling=None, *,
                             model_settings=None, progress=lambda event: None):
        from apps.vietsub_video import create_vietsub_video
        return create_vietsub_video(self, video, subtitles, voice, sampling,
                                    model_settings=model_settings, progress=progress)

    def create_vietsub_queue(self, tasks, *, model_settings=None, cancelled=lambda: False,
                             notify=lambda identifier, state, value: None):
        from apps.vietsub_queue import run_vietsub_queue
        return run_vietsub_queue(self, tasks, model_settings=model_settings,
                                 cancelled=cancelled, notify=notify)

    def edit_video_segments(self, path, segments, keep_audio=False, quality=20,
                            preset="fast", progress: Callable = lambda message: None,
                            export_mode="encode"):
        """Render a snapshot of several source-time ranges without loading TTS."""
        from apps.video_editor import SpeedSegment, render_speed_segments

        segments = tuple(SpeedSegment(float(item.start), float(item.end), float(item.speed),
                                      float(item.ramp_seconds)) for item in segments)
        if not segments:
            raise ValueError("Hãy thêm ít nhất một đoạn tốc độ trước khi xuất video.")
        for segment in segments:
            if not all(math.isfinite(value) for value in
                       (segment.start, segment.end, segment.speed, segment.ramp_seconds)):
                raise ValueError("Mốc thời gian và tốc độ video phải là số hữu hạn.")
            if segment.start < 0 or segment.end <= segment.start:
                raise ValueError("Mốc kết thúc phải sau mốc bắt đầu và mốc bắt đầu không âm.")
            if not 0.25 <= segment.speed <= 4 or not 0 <= segment.ramp_seconds <= (segment.end - segment.start) / 2:
                raise ValueError("Tốc độ video cần 0.25–4.0x; thời gian chuyển không vượt nửa đoạn đã chọn.")
        ordered = sorted(segments, key=lambda item: item.start)
        if any(current.end > following.start for current, following in zip(ordered, ordered[1:])):
            raise ValueError("Các đoạn tốc độ không được chồng lên nhau.")
        if export_mode not in ("copy", "encode"):
            raise ValueError("Chế độ xuất video không hợp lệ.")
        if export_mode == "encode" and (not isinstance(quality, int) or isinstance(quality, bool) or not 18 <= quality <= 28):
            raise ValueError("Chất lượng video CRF phải là số nguyên 18–28.")
        if export_mode == "encode" and preset not in ("ultrafast", "veryfast", "faster", "fast", "medium"):
            raise ValueError("Cấu hình xuất video không hợp lệ.")
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ValueError("Không tìm thấy video đầu vào.")
        with self.operation(require_model=False):
            destination = self._path("video", ".mp4")
            try:
                result = render_speed_segments(source, destination, segments, keep_audio=keep_audio,
                                               quality=quality, preset=preset,
                                               check_stop=self.check_stop, progress=progress, export_mode=export_mode)
                self.check_stop()
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
            note = f"{len(segments)} đoạn tốc độ."
            note = ("Giữ nguyên hình · " if export_mode == "copy" else "Mã hóa lại · ") + note
            if result is not None:
                note += f" Video xuất {result.output_duration:.2f}s."
            return destination, note

    def edit_video_markers(self, path, old_markers, new_markers, keep_audio=False,
                           quality=20, preset="fast", progress: Callable = lambda message: None,
                           export_mode="encode"):
        """Align a snapshot of numbered source and target anchors in a new MP4."""
        from apps.video_markers import validate_marker_pairs
        from apps.video_editor import render_marker_alignment

        old_markers, new_markers = validate_marker_pairs(tuple(old_markers), tuple(new_markers))
        if export_mode not in ("copy", "encode"):
            raise ValueError("Chế độ xuất video không hợp lệ.")
        if export_mode == "encode" and (not isinstance(quality, int) or isinstance(quality, bool) or not 18 <= quality <= 28):
            raise ValueError("Chất lượng video CRF phải là số nguyên 18–28.")
        if export_mode == "encode" and preset not in ("ultrafast", "veryfast", "faster", "fast", "medium"):
            raise ValueError("Cấu hình xuất video không hợp lệ.")
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ValueError("Không tìm thấy video đầu vào.")
        with self.operation(require_model=False):
            destination = self._path("video", ".mp4")
            try:
                result = render_marker_alignment(source, destination, old_markers, new_markers,
                                                 keep_audio=keep_audio, quality=quality, preset=preset,
                                                 check_stop=self.check_stop, progress=progress, export_mode=export_mode)
                self.check_stop()
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
            note = f"Đã căn {len(old_markers)} mốc cũ theo mốc mới · phần đuôi giữ 1x."
            note = ("Giữ nguyên hình · " if export_mode == "copy" else "Mã hóa lại · ") + note
            if result is not None:
                note += f" Video xuất {result.output_duration:.3f}s."
            return destination, note
