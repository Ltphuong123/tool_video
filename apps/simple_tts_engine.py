"""Reusable Turbo speech generation without a dependency on the desktop UI."""
import atexit
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import threading
import uuid

import numpy as np
import soundfile as sf

from apps.simple_tts_audio import process_audio

_DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "simple_tts"
_default_model = None
_default_voices = None
_default_lock = threading.Lock()


@dataclass(frozen=True)
class SpeechResult:
    path: Path | None
    audio: np.ndarray
    sample_rate: int


def generate_speech(text, voice=None, output_dir=None, *, options=None, model=None,
                    preset_voices=None, ref_audio=None, inference_kwargs=None, save=True, batch=False):
    """Generate Turbo audio; return SpeechResult or a list of results for batch=True.

    save=False returns audio in RAM without creating a temporary WAV.
    Pass an existing model to reuse its backend, presets and clone references.
    inference_kwargs forwards SDK sampling settings; denoise/watermark stay off.
    Without a model, load Turbo lazily and reuse it until process exit.
    """
    global _default_model, _default_voices
    texts = text if batch else [text]
    if batch and not isinstance(texts, (list, tuple)):
        raise ValueError("Batch cần danh sách văn bản.")
    if not texts or any(not isinstance(item, str) or not item.strip() for item in texts):
        raise ValueError("Hãy nhập văn bản cần đọc.")
    if options is not None:
        options.validate()
    kwargs = {"apply_watermark": False, **(inference_kwargs or {})}
    if kwargs.get("apply_watermark") or kwargs.get("denoise"):
        raise ValueError("Only Rubber Band, EQ, compressor and peak control are supported.")
    kwargs["denoise"] = False
    if "voice" in kwargs or "ref_audio" in kwargs:
        raise ValueError("Truyền voice/ref_audio qua tham số riêng.")
    if ref_audio is not None and not Path(ref_audio).is_file():
        raise ValueError("Không tìm thấy audio tham chiếu.")
    with _default_lock if model is None else nullcontext():
        if model is None:
            if _default_model is None:
                from vieneu import Vieneu
                engine = Vieneu(mode="v3turbo", backend="auto", device="auto")
                try:
                    voices = list(engine.list_preset_voices())
                except Exception:
                    engine.close()
                    raise
                _default_model, _default_voices = engine, voices
            model, preset_voices = _default_model, _default_voices
        if ref_audio is not None:
            kwargs["ref_audio"] = ref_audio
        else:
            if voice is not None and isinstance(voice, str):
                if preset_voices is not None:
                    valid = voice in {identifier for _, identifier in preset_voices}
                elif hasattr(model, "resolve_voice_name"):
                    valid = model.resolve_voice_name(voice) is not None
                else:
                    valid = voice in {identifier for _, identifier in model.list_preset_voices()}
                if not valid:
                    raise ValueError("Giọng đã chọn không có trong v3 Turbo.")
            kwargs["voice"] = voice
        rate = model.sample_rate
        if isinstance(rate, (bool, np.bool_)) or not isinstance(rate, (int, np.integer)) or rate <= 0:
            raise ValueError("Tần số audio của mô hình không hợp lệ.")
        inputs = [item.strip() for item in texts]
        audios = model.infer_batch(inputs, **kwargs) if batch else [model.infer(inputs[0], **kwargs)]
        if len(audios) != len(inputs):
            raise RuntimeError("Số audio trả về không khớp số văn bản.")
        # Validate all results before writing any batch output.
        validated = []
        for audio in audios:
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
                raise ValueError("Mô hình không trả về audio mono hợp lệ.")
            validated.append(audio)
        results = []
        for audio in validated:
            if options is not None:
                audio, _ = process_audio(audio, int(rate), options)
            path = None
            if save:
                folder = Path(output_dir or _DEFAULT_OUTPUT).expanduser().resolve()
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"speech_v3turbo_{uuid.uuid4().hex[:12]}.wav"
                temporary = path.with_suffix(".tmp")
                try:
                    sf.write(temporary, audio, int(rate), format="WAV", subtype="FLOAT")
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            # Saved results can be cached independently of SDK-owned buffers.
            results.append(SpeechResult(path, audio.copy() if save else audio, int(rate)))
        return results if batch else results[0]


def _close_default_model():
    global _default_model, _default_voices
    with _default_lock:
        model, _default_model, _default_voices = _default_model, None, None
        if model is not None:
            model.close()


atexit.register(_close_default_model)
