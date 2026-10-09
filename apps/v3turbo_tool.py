"""Reusable operations for the v3 Turbo desktop app; no GUI or model at import."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterator
import json
import math
import re
import tempfile
from time import perf_counter
import threading
import uuid

import numpy as np
import soundfile as sf

from apps.speech_speed import require_rubberband, stretch_rubberband
from apps.simple_tts_engine import generate_speech
from apps.simple_tts_audio import AudioOptions, build_effect_chain, process_audio
from apps.srt_speech import concatenate, lay_on_timeline, parse_srt
from apps.srt_timing import (
    cue_sample_windows, prepare_srt_clip,
)
from apps.speech_subtitles import assemble_sentence_track, render_srt, subtitle_sentences
from apps.user_voices import (
    _entry_from_json,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
STREAM_BUFFER_BYTES = 8 * 1024 * 1024
OUTPUT_NAME = re.compile(
    r"^(speech(?:_subtitled)?|stream|conversation|srt|denoised|microphone|"
    r"batch(?:_\d{4,})?|voices|reference|video)_[0-9a-f]{12}\.([a-z0-9]+)$", re.IGNORECASE,
)
OUTPUT_KINDS = {"wav": "Audio", "flac": "Audio", "mp3": "Audio", "srt": "Phụ đề",
                "zip": "Batch ZIP", "json": "Giọng đã xuất", "npz": "Embedding / codes", "mp4": "Video"}


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
    bright: bool = False
    compress: bool = False
    peak_guard: bool = False

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
        for key in ("speed", "bright", "compress", "peak_guard"):
            values.pop(key)
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
    def _output_kind(path: Path) -> str | None:
        match = OUTPUT_NAME.fullmatch(path.name)
        if match is None:
            return None
        label, suffix = (part.lower() for part in match.groups())
        if suffix in ("wav", "flac", "mp3"):
            return "Audio" if label not in ("voices", "reference", "video") else None
        expected = {"srt": "speech_subtitled", "zip": "batch",
                    "json": "voices", "npz": "reference", "mp4": "video"}
        return OUTPUT_KINDS.get(suffix) if expected.get(suffix) == label else None

    def list_outputs(self) -> list[OutputFile]:
        """Read history from generated files, including previously used folders."""
        results = []
        for directory in self.output_dirs:
            try:
                files = list(directory.iterdir())
            except OSError:
                continue
            for path in files:
                kind = self._output_kind(path)
                if kind is None or path.is_symlink():
                    continue
                try:
                    if not path.is_file() or path.resolve().parent != directory:
                        continue
                    info = path.stat()
                except OSError:
                    continue
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
                if include_subtitles and path.suffix.lower() in (".wav", ".flac", ".mp3"):
                    subtitle = canonical.with_suffix(".srt")
                    if subtitle in allowed:
                        selected.append(allowed[subtitle])
            selected = list(dict.fromkeys(selected))
            # Check again before the first mutation, protecting changed directories.
            for path in selected:
                if (path.is_symlink() or not path.is_file() or
                        path.resolve().parent not in self.output_dirs):
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
        if sampling.bright or sampling.compress:
            from apps.simple_tts_audio import pedalboard_module
            pedalboard_module()
        return kw

    def _effects(self, audio, sampling: Sampling, effects_chain=None):
        if not (sampling.bright or sampling.compress or sampling.peak_guard):
            return audio
        self.check_stop()
        wav, _ = process_audio(audio, 48000, AudioOptions(
            bright=sampling.bright, compress=sampling.compress, peak_guard=sampling.peak_guard),
                               effects_chain=effects_chain)
        return wav

    def _adjust_speed(self, audio, sampling: Sampling, effects_chain=None) -> np.ndarray:
        self.check_stop()
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sampling.speed != 1.0 and wav.size:
            wav = stretch_rubberband(wav, sampling.speed, check_stop=self.check_stop)
            self.check_stop()
        return self._effects(wav, sampling, effects_chain)

    def _path(self, label: str, suffix: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / f"{label}_{uuid.uuid4().hex[:12]}{suffix}"

    def _save(self, audio, fmt="wav", rate=48000, label="speech") -> Path:
        fmt = fmt.lower()
        if fmt not in ("wav", "flac", "mp3"):
            raise ValueError("Định dạng phải là WAV, FLAC hoặc MP3.")
        if fmt == "mp3" and "MP3" not in sf.available_formats():
            raise ValueError("libsndfile trên máy này chưa hỗ trợ MP3. Hãy chọn WAV hoặc FLAC.")
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not wav.size or not np.isfinite(wav).all():
            raise ValueError("Model không trả về audio hợp lệ.")
        path = self._path(label, f".{fmt}")
        sf.write(path, wav, rate, format=fmt.upper(), **({"subtype": "PCM_16"} if fmt != "mp3" else {}))
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
        started = perf_counter()
        model_seconds = 0.0
        sentences = subtitle_sentences(text)
        sampling = sampling or Sampling()
        with self.operation():
            kw = self._sampling(sampling)
            ref = self._reference(voice, ref_audio)
            speed_sampling = replace(sampling, bright=False, compress=False, peak_guard=False)
            clips = []
            window = min(128, sampling.batch_size * 4) if self.tts.backend == "pytorch" else sampling.batch_size
            for start in range(0, len(sentences), window):
                self.check_stop()
                group = [sentence for sentence, gap in sentences[start:start + window]]
                batch_started = perf_counter()
                wavs = self._generate_audio(group, ref, kw, batch=True)
                model_seconds += perf_counter() - batch_started
                if len(wavs) != len(group):
                    raise RuntimeError("Số audio trả về không khớp số câu để tạo phụ đề.")
                clips.extend(self._adjust_speed(wav, speed_sampling) for wav in wavs)
                progress(f"Audio + SRT: đã đọc {len(clips)}/{len(sentences)} câu")
            self.check_stop()
            track, spans = assemble_sentence_track(sentences, clips)
            del clips, wavs
            track = self._effects(track, sampling)
            self.check_stop()
            srt_text = render_srt(spans)
            audio_path = subtitle_path = None
            try:
                audio_path = self._save(track, fmt, label="speech_subtitled")
                subtitle_path = audio_path.with_suffix(".srt")
                subtitle_path.write_text(srt_text, encoding="utf-8")
                self.check_stop()
            except BaseException:
                if audio_path is not None:
                    audio_path.unlink(missing_ok=True)
                if subtitle_path is not None:
                    subtitle_path.unlink(missing_ok=True)
                raise
            elapsed = perf_counter() - started
            return SubtitledSpeech(audio_path, subtitle_path, len(sentences), len(track) / 48000,
                                   {"total": elapsed, "model": model_seconds,
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
            buffered = sampling.speed != 1.0 or sampling.bright or sampling.compress or sampling.peak_guard
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
        cues = parse_srt(Path(path).read_text(encoding="utf-8-sig"), strict_timing=fit_to_timing)
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
            effects_chain = build_effect_chain(AudioOptions(bright=sampling.bright, compress=sampling.compress))
            clips = []
            track = (np.zeros(max(cue.end_ms for cue in cues) * 48000 // 1000, dtype=np.float32)
                     if fit_to_timing else None)
            accelerated, max_speed = 0, sampling.speed
            window = min(128, sampling.batch_size * 4) if self.tts.backend == "pytorch" else sampling.batch_size
            for start in range(0, len(cues), window):
                self.check_stop()
                group = cues[start:start + window]
                batch_started = perf_counter()
                wavs = self._generate_audio([cue.text for cue in group], ref, kw, batch=True)
                model_seconds += perf_counter() - batch_started
                processing_started = perf_counter()
                if len(wavs) != len(group):
                    raise RuntimeError("Số audio không khớp số câu SRT.")
                for offset, wav in enumerate(wavs):
                    if fit_to_timing:
                        cue = group[offset]
                        begin, end = windows[start + offset]
                        audio, speed = prepare_srt_clip(np.asarray(wav, dtype=np.float32).reshape(-1),
                                                        end - begin, sampling.speed,
                                                        check_stop=self.check_stop)
                        if speed > sampling.speed:
                            accelerated += 1
                        max_speed = max(max_speed, speed)
                        progress(f"SRT câu {cue.index}: bắt đầu {begin / 48000:.3f}s, "
                                 f"kết thúc {(begin + len(audio)) / 48000:.3f}s / giới hạn {end / 48000:.3f}s, "
                                 f"tốc độ {speed:.2f}x")
                        audio = self._effects(audio, sampling, effects_chain)
                        if len(audio) > end - begin:
                            raise ValueError(f"Câu SRT {cue.index}: audio chưa vừa khung thời gian.")
                        track[begin:begin + len(audio)] = audio
                    else:
                        audio = self._adjust_speed(wav, sampling, effects_chain)
                        clips.append(audio)
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
            output = self._save(track, fmt, label="srt")
            note += (f" Total={perf_counter() - started:.2f}s; model={model_seconds:.2f}s; "
                     f"tempo/effects={processing_seconds:.2f}s; export={perf_counter() - export_started:.2f}s.")
            return output, note

    def edit_video(self, path, start, end, speed=1.5, ramp_seconds=0.5, keep_audio=False,
                   quality=20, preset="fast", progress: Callable = lambda message: None):
        values = (start, end, speed, ramp_seconds)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Mốc thời gian và tốc độ video phải là số hữu hạn.")
        if start < 0 or end <= start:
            raise ValueError("Mốc kết thúc phải sau mốc bắt đầu và mốc bắt đầu không âm.")
        if not 0.25 <= speed <= 4 or not 0 <= ramp_seconds <= (end - start) / 2:
            raise ValueError("Tốc độ video cần 0.25–4.0x; thời gian chuyển không vượt nửa đoạn đã chọn.")
        if not isinstance(quality, int) or isinstance(quality, bool) or not 18 <= quality <= 28:
            raise ValueError("Chất lượng video CRF phải là số nguyên 18–28.")
        if preset not in ("ultrafast", "veryfast", "faster", "fast", "medium"):
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
                                              progress=progress)
                self.check_stop()
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
            note = f"{start:.2f}–{end:.2f}s · {speed:g}x · chuyển tốc độ {ramp_seconds:g}s."
            if result is not None:
                note += f" Video xuất {result.output_duration:.2f}s."
            return destination, note

    def edit_video_segments(self, path, segments, keep_audio=False, quality=20,
                            preset="fast", progress: Callable = lambda message: None):
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
        if not isinstance(quality, int) or isinstance(quality, bool) or not 18 <= quality <= 28:
            raise ValueError("Chất lượng video CRF phải là số nguyên 18–28.")
        if preset not in ("ultrafast", "veryfast", "faster", "fast", "medium"):
            raise ValueError("Cấu hình xuất video không hợp lệ.")
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ValueError("Không tìm thấy video đầu vào.")
        with self.operation(require_model=False):
            destination = self._path("video", ".mp4")
            try:
                result = render_speed_segments(source, destination, segments, keep_audio=keep_audio,
                                               quality=quality, preset=preset,
                                               check_stop=self.check_stop, progress=progress)
                self.check_stop()
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
            note = f"{len(segments)} đoạn tốc độ."
            if result is not None:
                note += f" Video xuất {result.output_duration:.2f}s."
            return destination, note

    def edit_video_markers(self, path, old_markers, new_markers, keep_audio=False,
                           quality=20, preset="fast", progress: Callable = lambda message: None):
        """Align a snapshot of numbered source and target anchors in a new MP4."""
        from apps.video_markers import validate_marker_pairs
        from apps.video_editor import render_marker_alignment

        old_markers, new_markers = validate_marker_pairs(tuple(old_markers), tuple(new_markers))
        if not isinstance(quality, int) or isinstance(quality, bool) or not 18 <= quality <= 28:
            raise ValueError("Chất lượng video CRF phải là số nguyên 18–28.")
        if preset not in ("ultrafast", "veryfast", "faster", "fast", "medium"):
            raise ValueError("Cấu hình xuất video không hợp lệ.")
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ValueError("Không tìm thấy video đầu vào.")
        with self.operation(require_model=False):
            destination = self._path("video", ".mp4")
            try:
                result = render_marker_alignment(source, destination, old_markers, new_markers,
                                                 keep_audio=keep_audio, quality=quality, preset=preset,
                                                 check_stop=self.check_stop, progress=progress)
                self.check_stop()
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
            note = f"Đã căn {len(old_markers)} mốc cũ theo mốc mới · phần đuôi giữ 1x."
            if result is not None:
                note += f" Video xuất {result.output_duration:.3f}s."
            return destination, note
