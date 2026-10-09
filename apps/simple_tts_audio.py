"""Rubber Band and lightweight voice effects for the separate desktop app."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import uuid

import numpy as np
import soundfile as sf

TEMPO = {"Không đổi tốc độ": "none", "Rubber Band chất lượng cao": "rubberband"}


@dataclass(frozen=True)
class AudioOptions:
    tempo: str = "none"
    speed: float = 1.0
    bright: bool = False
    compress: bool = False
    peak_guard: bool = False

    def validate(self):
        if self.tempo not in TEMPO.values():
            raise ValueError("Công nghệ xử lý không hợp lệ.")
        if not math.isfinite(self.speed) or not 0.5 <= self.speed <= 2.0:
            raise ValueError("Tốc độ phải từ 0.5 đến 2.0x.")
        if self.tempo == "none" and self.speed != 1:
            raise ValueError("Chọn Rubber Band hoặc đặt tốc độ 1.0x.")


def mono(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Bộ xử lý trả về audio không hợp lệ.")
    return np.ascontiguousarray(audio)


def pedalboard_module():
    try:
        import pedalboard
        return pedalboard
    except ImportError as exc:
        raise RuntimeError("Chưa có Pedalboard. Chạy: uv pip install --python .venv/Scripts/python.exe -r requirements-simple-tts-audio.txt") from exc


def build_effect_chain(options):
    """Build once per job; reset the same effects before each independent cue."""
    if not (options.bright or options.compress):
        return None
    pb = pedalboard_module()
    effects = []
    if options.bright:
        effects.extend([pb.HighpassFilter(cutoff_frequency_hz=65),
                        pb.HighShelfFilter(cutoff_frequency_hz=3500, gain_db=1.5, q=0.707)])
    if options.compress:
        effects.append(pb.Compressor(threshold_db=-18, ratio=2, attack_ms=15, release_ms=120))
    return pb.Pedalboard(effects)


def process_audio(audio, rate, options, *, effects_chain=None):
    options.validate()
    audio = mono(audio)
    # No copy or engine setup for bypass. All processing keeps source untouched.
    if options.speed != 1:
        audio = pedalboard_module().time_stretch(
            audio, rate, stretch_factor=options.speed, pitch_shift_in_semitones=0,
            high_quality=True, retain_phase_continuity=True)
    if options.bright or options.compress:
        chain = effects_chain if effects_chain is not None else build_effect_chain(options)
        audio = chain.process(audio, rate, reset=True)
    if options.speed != 1 or options.bright or options.compress:
        audio = mono(audio)
    if options.peak_guard:
        peak = float(np.max(np.abs(audio)))
        ceiling = 10 ** (-1 / 20)
        if peak > ceiling:
            audio = audio * (ceiling / peak)
    return audio, rate


def save_variant(source, output_dir, options, audio=None, rate=None):
    options.validate()
    source = Path(source).resolve()
    if audio is None:
        audio, rate = sf.read(source, dtype="float32")
    audio, rate = process_audio(audio, rate, options)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{source.stem}_{options.tempo}_{uuid.uuid4().hex[:8]}.wav"
    temporary = path.with_suffix(".tmp")
    try:
        sf.write(temporary, audio, rate, format="WAV", subtype="FLOAT")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    report = {"source": str(source), "result": str(path), "options": asdict(options),
              "sample_rate": rate, "duration_seconds": len(audio) / rate,
              "peak": float(np.max(np.abs(audio)))}
    path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
