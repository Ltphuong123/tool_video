"""Reusable operations for the v3 Turbo desktop app; no GUI or model at import."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterator
import json
import math
import re
import tempfile
import threading
import uuid
import zipfile

import numpy as np
import soundfile as sf

from apps.speech_speed import require_rubberband, stretch_rubberband
from apps.srt_speech import concatenate, lay_on_timeline, parse_srt
from apps.srt_timing import (
    cue_sample_windows, prepare_srt_clip, soften_srt_tail,
)
from apps.speech_subtitles import assemble_sentence_track, render_srt, subtitle_sentences
from apps.user_voices import (
    USER_MARK, _entry_from_json, _entry_to_json,
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
    denoise: bool = True
    use_ref_codes: bool = True
    apply_watermark: bool = False

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


def parse_conversation(script: str, mapping: dict[str, str]) -> list[tuple[str, str]]:
    lines = []
    for index, raw in enumerate(script.splitlines(), 1):
        if not raw.strip():
            continue
        speaker, sep, text = raw.partition(":")
        if not sep or not text.strip():
            raise ValueError(f"Dòng {index}: dùng định dạng Nhân vật: lời thoại.")
        voice = mapping.get(speaker.strip())
        if not voice:
            raise ValueError(f"Chưa chọn giọng cho '{speaker.strip()}'.")
        lines.append((voice, text.strip()))
    if not lines:
        raise ValueError("Hãy nhập kịch bản hội thoại.")
    return lines


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
            watermark = "sẵn sàng" if model.watermarker else "chưa cài"
            return f"Đã tải v3 Turbo · {model.backend} · 48 kHz · {len(model._preset_voices)} giọng · watermark {watermark}"

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
        if sampling.apply_watermark and not self.tts.watermarker:
            raise ValueError("Watermark chưa khả dụng. Cài: uv sync --extra watermark")
        if sampling.speed != 1.0:
            require_rubberband()
            # Apply the watermark after time stretching so it marks the final audio.
            kw["apply_watermark"] = False
        return kw

    def _adjust_speed(self, audio, sampling: Sampling) -> np.ndarray:
        self.check_stop()
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sampling.speed == 1.0 or not wav.size:
            return wav
        wav = stretch_rubberband(wav, sampling.speed, check_stop=self.check_stop)
        self.check_stop()
        if sampling.apply_watermark:
            wav = self.tts._apply_watermark(wav)
        return np.asarray(wav, dtype=np.float32)

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

    def synthesize(self, text: str, voice=None, ref_audio=None, sampling=None, fmt="wav") -> Path:
        if not text.strip():
            raise ValueError("Hãy nhập văn bản.")
        sampling = sampling or Sampling()
        with self.operation():
            kw = self._sampling(sampling)
            ref = self._reference(voice, ref_audio)
            audio = self.tts.infer(text, **ref, **kw)
            self.check_stop()
            return self._save(self._adjust_speed(audio, sampling), fmt)


    def synthesize_with_subtitles(self, text: str, voice=None, ref_audio=None, sampling=None,
                                  fmt="wav", progress: Callable = lambda message: None) -> SubtitledSpeech:
        sentences = subtitle_sentences(text)
        sampling = sampling or Sampling()
        with self.operation():
            kw = self._sampling(sampling)
            ref = self._reference(voice, ref_audio)
            clips = []
            for start in range(0, len(sentences), sampling.batch_size):
                self.check_stop()
                group = [sentence for sentence, gap in sentences[start:start + sampling.batch_size]]
                wavs = self.tts.infer_batch(group, **ref, **kw)
                if len(wavs) != len(group):
                    raise RuntimeError("Số audio trả về không khớp số câu để tạo phụ đề.")
                clips.extend(self._adjust_speed(wav, sampling) for wav in wavs)
                progress(f"Audio + SRT: đã đọc {len(clips)}/{len(sentences)} câu")
            self.check_stop()
            track, spans = assemble_sentence_track(sentences, clips)
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
            return SubtitledSpeech(audio_path, subtitle_path, len(sentences), len(track) / 48000)

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

    def batch(self, texts: list[str], voice=None, ref_audio=None, sampling=None, fmt="wav",
              progress: Callable = lambda message: None) -> Path:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Mỗi mục trong batch phải có văn bản.")
        sampling = sampling or Sampling()
        with self.operation():
            kw = self._sampling(sampling)
            ref = self._reference(voice, ref_audio)
            paths = []
            manifest = []
            for start in range(0, len(texts), sampling.batch_size):
                self.check_stop()
                group = texts[start:start + sampling.batch_size]
                wavs = self.tts.infer_batch(group, **ref, **kw)
                if len(wavs) != len(group):
                    raise RuntimeError("Số audio trả về không khớp số văn bản.")
                self.check_stop()
                for index, (text, wav) in enumerate(zip(group, wavs), start + 1):
                    path = self._save(self._adjust_speed(wav, sampling), fmt, label=f"batch_{index:04}")
                    paths.append(path)
                    manifest.append({"index": index, "text": text, "audio": f"{index:04}.{fmt}",
                                     "speed": sampling.speed, "speed_method": "rubberband"})
                progress(f"Đã tạo {len(paths)}/{len(texts)} audio")
            archive = self._path("batch", ".zip")
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
                for index, path in enumerate(paths, 1):
                    out.write(path, f"{index:04}.{fmt}")
                out.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            return archive

    def conversation(self, script, mapping, sampling=None, gap=0.4, fmt="wav",
                     progress: Callable = lambda message: None):
        lines = parse_conversation(script, mapping)
        if not math.isfinite(gap) or not 0 <= gap <= 10:
            raise ValueError("Khoảng nghỉ phải nằm trong 0–10 giây.")
        sampling = sampling or Sampling()
        with self.operation():
            kw = self._sampling(sampling)
            for voice in dict.fromkeys(voice for voice, _ in lines):
                self._reference(voice, None)
            clips = []
            start = 0
            while start < len(lines):
                self.check_stop()
                voice = lines[start][0]
                end = start + 1
                while end < min(len(lines), start + sampling.batch_size) and lines[end][0] == voice:
                    end += 1
                wavs = self.tts.infer_batch([text for _, text in lines[start:end]], voice=voice, **kw)
                if len(wavs) != end - start:
                    raise RuntimeError("Số audio trả về không khớp số lượt hội thoại.")
                for index, wav in enumerate(wavs, start + 1):
                    clips.append(self._adjust_speed(wav, sampling))
                    progress(f"Hội thoại {index}/{len(lines)} · {voice}")
                start = end
            self.check_stop()
            return self._save(concatenate(clips, 48000, gap), fmt, label="conversation")

    def srt(self, path, voice=None, sampling=None, keep_timing=True, fmt="wav",
            progress: Callable = lambda message: None, fit_to_timing=None, min_speed=None):
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
                # All speed adjustments finish before marking the final clip.
                kw["apply_watermark"] = False
                progress(f"Bộ chỉnh tốc độ SRT: rubberband · tốc độ ban đầu {sampling.speed:.2f}x.")
            ref = self._reference(voice, None)
            clips = []
            track = (np.zeros(max(cue.end_ms for cue in cues) * 48000 // 1000, dtype=np.float32)
                     if fit_to_timing else None)
            accelerated, max_speed = 0, sampling.speed
            for start in range(0, len(cues), sampling.batch_size):
                self.check_stop()
                group = cues[start:start + sampling.batch_size]
                wavs = self.tts.infer_batch([cue.text for cue in group], **ref, **kw)
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
                        if sampling.apply_watermark:
                            expected = len(audio)
                            audio = np.asarray(self.tts._apply_watermark(audio), dtype=np.float32).reshape(-1)
                            if len(audio) != expected:
                                raise ValueError("Watermark thay đổi độ dài audio; hãy tắt watermark khi căn thời gian SRT.")
                        audio = soften_srt_tail(audio)
                        if len(audio) > end - begin:
                            raise ValueError(f"Câu SRT {cue.index}: audio chưa vừa khung thời gian.")
                        track[begin:begin + len(audio)] = audio
                    else:
                        audio = self._adjust_speed(wav, sampling)
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
            return self._save(track, fmt, label="srt"), note

    def edit_video(self, path, start, end, speed=1.5, ramp_seconds=0.5, keep_audio=True,
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
        if preset not in ("fast", "medium", "faster"):
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
            note = f"MoviePy · {start:.2f}–{end:.2f}s · {speed:g}x · chuyển tốc độ {ramp_seconds:g}s."
            if result is not None:
                note += f" Video xuất {result.output_duration:.2f}s."
            return destination, note

    def _persist_voices(self):
        self.voices_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"presets": {name: _entry_to_json(v) for name, v in self.tts._preset_voices.items()
                            if v.get(USER_MARK)}}
        temp = self.voices_path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.voices_path)

    def add_voice(self, name, audio, description="", denoise=True, use_ref_codes=True):
        name = name.strip()
        if not name or len(name) > 40:
            raise ValueError("Tên giọng phải có 1–40 ký tự.")
        if not audio:
            raise ValueError("Hãy chọn audio mẫu.")
        with self.operation():
            self._reference(None, audio)
            canonical = self.tts.resolve_voice_name(name)
            if canonical and not self.tts._preset_voices[canonical].get(USER_MARK):
                raise ValueError("Tên này trùng giọng có sẵn hoặc bí danh của giọng có sẵn.")
            self.tts.add_voice(name, audio, description=description, denoise=denoise,
                               use_ref_codes=use_ref_codes)
            self.tts._preset_voices[name][USER_MARK] = True
            self._persist_voices()

    def delete_voice(self, name):
        with self.operation():
            if not self.tts._preset_voices.get(name, {}).get(USER_MARK):
                raise ValueError("Chỉ xóa được giọng do bạn lưu.")
            self.tts.remove_voice(name)
            self._persist_voices()

    def export_voices(self):
        with self.operation():
            path = self._path("voices", ".json")
            self.tts.save_voices(path)
            return path

    def denoise(self, audio):
        if not audio:
            raise ValueError("Hãy chọn audio mẫu.")
        with self.operation():
            self._reference(None, audio)
            wav, rate = self.tts.denoise(audio)
            self.check_stop()
            return self._save(wav, "wav", rate, label="denoised")

    def export_reference(self, audio, denoise=True):
        if not audio:
            raise ValueError("Hãy chọn audio mẫu.")
        with self.operation():
            self._reference(None, audio)
            speaker, codes = self.tts.encode_reference(audio, denoise=denoise)
            self.check_stop()
            path = self._path("reference", ".npz")
            np.savez_compressed(path, speaker_emb=np.asarray(speaker), ref_codes=np.asarray(codes))
            return path

    def import_voices(self, path):
        with self.operation():
            data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            presets = data.get("presets")
            if not isinstance(presets, dict) or not presets:
                raise ValueError("JSON phải chứa đối tượng presets có ít nhất một giọng.")
            entries, skipped = {}, 0
            for name, raw in presets.items():
                if not isinstance(name, str) or not name.strip() or len(name) > 40:
                    raise ValueError("Tên giọng trong JSON phải có 1–40 ký tự.")
                existing = self.tts.resolve_voice_name(name)
                if existing and not self.tts._preset_voices[existing].get(USER_MARK):
                    skipped += 1
                    continue
                if not isinstance(raw, dict):
                    raise ValueError(f"Giọng '{name}' có dữ liệu không hợp lệ.")
                embedding = np.asarray(raw.get("speaker_emb"), dtype=np.float32)
                if embedding.shape != (192,) or not np.isfinite(embedding).all() or not embedding.any():
                    raise ValueError(f"Giọng '{name}' cần speaker_emb gồm 192 số hữu hạn, khác zero.")
                codes = raw.get("codes")
                if codes is not None:
                    codes = np.asarray(codes)
                    if codes.ndim != 2 or codes.shape[1] != 16 or not codes.size:
                        raise ValueError(f"Giọng '{name}': codes phải có kích thước [frames, 16].")
                    if not np.isfinite(codes).all() or np.any(codes < 0) or np.any(codes > 1023) or np.any(codes != codes.astype(np.int64)):
                        raise ValueError(f"Giọng '{name}': codes phải là số nguyên 0–1023.")
                entries[name] = _entry_from_json(self.tts, raw)
            previous = dict(self.tts._preset_voices)
            self.tts._preset_voices.update(entries)
            try:
                self._persist_voices()
            except Exception:
                self.tts._preset_voices = previous
                raise
            return f"Đã nhập {len(entries)} giọng riêng; bỏ qua {skipped} giọng có sẵn."
