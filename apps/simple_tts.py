"""Independent, minimal desktop TTS app using only the VieNeu SDK."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import queue
import subprocess
import sys
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from apps.simple_tts_audio import AudioOptions, TEMPO, mono, save_variant
from apps.simple_tts_engine import generate_speech

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOICE = "Giọng mặc định"


class SimpleTTS:
    """Keep one model loaded; use a separate destination from the Studio."""
    def __init__(self, output_dir=None, factory=None):
        self.output_dir = Path(output_dir or ROOT / "outputs" / "simple_tts").expanduser().resolve()
        self.factory = factory
        self.model = None
        self.lock = threading.Lock()
        self._voices = []
        self._source_key = None
        self._source_audio = None
        self._source_rate = None

    @contextmanager
    def operation(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Đang có tác vụ tạo giọng chạy.")
        try:
            yield
        finally:
            self.lock.release()

    def _ensure_model(self):
        if self.model is not None:
            return
        factory = self.factory
        if factory is None:
            from vieneu import Vieneu
            factory = Vieneu
        model = factory(mode="v3turbo", backend="auto", device="auto")
        try:
            voices = list(model.list_preset_voices())
        except Exception:
            model.close()
            raise
        self.model, self._voices = model, voices

    def voices(self):
        return list(self._voices) if self.model is not None else []

    def load(self):
        with self.operation():
            self._ensure_model()
            return self.voices()

    def generate(self, text, voice=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Hãy nhập văn bản cần đọc.")
        with self.operation():
            self._ensure_model()
            result = generate_speech(text, voice, self.output_dir,
                                     model=self.model, preset_voices=self._voices)
            self._cache_source(result.path, result.audio, result.sample_rate)
            return result.path

    @staticmethod
    def _file_key(path):
        stat = path.stat()
        return path.resolve(), stat.st_mtime_ns, stat.st_size

    def _cache_source(self, path, audio, rate):
        # Retain just the latest source; never accumulate audio from history.
        self._source_key = self._file_key(path)
        self._source_audio = audio
        self._source_rate = rate

    def variant(self, source, options):
        options.validate()
        with self.operation():
            source = Path(source).resolve()
            key = self._file_key(source)
            if key != self._source_key:
                audio, rate = sf.read(source, dtype="float32")
                self._cache_source(source, mono(audio), rate)
            return save_variant(source, self.output_dir, options,
                                self._source_audio, self._source_rate)

    def close(self):
        with self.operation():
            previous, self.model = self.model, None
            if previous is not None:
                previous.close()
            self._voices = []
            self._source_key = self._source_audio = self._source_rate = None


class SimpleTTSApp:
    def __init__(self, root, service=None):
        self.root = root
        self.service = service or SimpleTTS()
        self.events = queue.Queue()
        self.busy = False
        self.closing = False
        self.last_audio = None
        self.original_audio = None
        self.results = {}
        self.extra_controls = []
        self.voice_ids = {DEFAULT_VOICE: None}
        root.title("VieNeu — Đọc văn bản")
        root.geometry("900x760")
        root.minsize(820, 700)
        root.configure(background="#eef2f8")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Simple.TFrame", background="#ffffff")
        style.configure("Simple.TLabel", background="#ffffff", foreground="#25324b", font=("Segoe UI", 10))
        style.configure("SimpleTitle.TLabel", background="#ffffff", foreground="#17233c",
                        font=("Segoe UI", 21, "bold"))
        style.configure("Simple.TButton", font=("Segoe UI", 10), padding=(12, 8))
        style.configure("SimplePrimary.TButton", background="#5b48ef", foreground="#ffffff",
                        font=("Segoe UI", 10, "bold"), padding=(18, 9))
        style.map("SimplePrimary.TButton", background=[("disabled", "#ccc5f8"), ("active", "#4836d2")])
        frame = ttk.Frame(root, style="Simple.TFrame", padding=22)
        frame.pack(fill="both", expand=True, padx=16, pady=16)
        ttk.Label(frame, text="Văn bản thành giọng nói", style="SimpleTitle.TLabel").pack(anchor="w")
        ttk.Label(frame, text="Chọn mô hình, nhập nội dung và tạo file WAV.",
                  style="Simple.TLabel").pack(anchor="w", pady=(4, 16))

        row = ttk.Frame(frame, style="Simple.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text="Mô hình", style="Simple.TLabel", width=10).pack(side="left")
        ttk.Label(row, text="VieNeu v3 Turbo", style="Simple.TLabel").pack(side="left", fill="x", expand=True)
        self.load_button = ttk.Button(row, text="Tải giọng", style="Simple.TButton",
                                      command=lambda: self.guard(self.load_voices))
        self.load_button.pack(side="left", padx=(10, 0))
        row = ttk.Frame(frame, style="Simple.TFrame")
        row.pack(fill="x", pady=(10, 0))
        ttk.Label(row, text="Giọng đọc", style="Simple.TLabel", width=10).pack(side="left")
        self.voice_name = tk.StringVar(value=DEFAULT_VOICE)
        self.voice_box = ttk.Combobox(row, textvariable=self.voice_name, values=[DEFAULT_VOICE], state="readonly")
        self.voice_box.pack(side="left", fill="x", expand=True)
        ttk.Label(frame, text="Turbo tự chọn CPU/CUDA. Lần tải đầu cần Internet.",
                  style="Simple.TLabel").pack(anchor="w", pady=(10, 12))

        self.tabs = ttk.Notebook(frame)
        self.tabs.pack(fill="both", expand=True)
        editor = ttk.Frame(self.tabs, style="Simple.TFrame", padding=8)
        self.tabs.add(editor, text="1. Văn bản")
        settings_page, settings = self.scroll_page(self.tabs)
        self.tabs.add(settings_page, text="2. Thử công nghệ")
        self.tempo_name = tk.StringVar(value=next(iter(TEMPO)))
        self.speed = tk.StringVar(value="1.0")
        self.bright = tk.BooleanVar(value=False)
        self.compress = tk.BooleanVar(value=False)
        self.peak_guard = tk.BooleanVar(value=False)
        ttk.Label(settings, text="Đổi tốc độ", style="Simple.TLabel").pack(anchor="w")
        box = ttk.Combobox(settings, textvariable=self.tempo_name, values=list(TEMPO), state="readonly")
        box.pack(fill="x", pady=(4, 12))
        self.extra_controls.append((box, "readonly"))
        row = ttk.Frame(settings, style="Simple.TFrame")
        row.pack(fill="x", pady=(0, 12))
        ttk.Label(row, text="Tốc độ (0.5–2.0x)", style="Simple.TLabel").pack(side="left")
        speed_box = ttk.Spinbox(row, textvariable=self.speed, from_=0.5, to=2.0, increment=0.05, width=8)
        speed_box.pack(side="left", padx=12)
        self.extra_controls.append((speed_box, "normal"))
        for label, var in [("EQ: giọng sáng nhẹ", self.bright),
                           ("Compressor: âm lượng đều hơn", self.compress),
                           ("Giảm đỉnh vượt -1 dBFS (chỉ giảm âm lượng)", self.peak_guard)]:
            box = ttk.Checkbutton(settings, text=label, variable=var)
            box.pack(anchor="w", pady=3)
            self.extra_controls.append((box, "normal"))
        ttk.Label(settings, text="Tạo bản gốc một lần, đổi thiết lập rồi bấm Thử trên bản gốc.\n"
                  "Mỗi lần thử lưu WAV + JSON riêng, không tạo lại lời đọc.\n"
                  "EQ/giảm đỉnh không phục hồi tiếng vỡ đã có trong âm gốc.",
                  style="Simple.TLabel", wraplength=750).pack(anchor="w", pady=12)
        self.text = tk.Text(editor, wrap="word", font=("Segoe UI", 12), height=8,
                            undo=True, relief="flat", background="#f8faff", padx=12, pady=12,
                            foreground="#17233c", insertbackground="#5b48ef",
                            highlightthickness=1, highlightbackground="#dce3ef")
        scrollbar = ttk.Scrollbar(editor, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.text.pack(fill="both", expand=True)

        actions = ttk.Frame(frame, style="Simple.TFrame")
        actions.pack(side="bottom", fill="x", pady=(12, 0), before=self.tabs)
        self.generate_button = ttk.Button(actions, text="Tạo bản gốc", style="SimplePrimary.TButton",
                                         command=lambda: self.guard(self.generate))
        self.generate_button.pack(side="left")
        self.play_button = ttk.Button(actions, text="Nghe file", style="Simple.TButton",
                                      state="disabled", command=lambda: self.guard(self.play))
        self.play_button.pack(side="left", padx=8)
        self.apply_button = ttk.Button(actions, text="Thử trên bản gốc", style="Simple.TButton",
                                       state="disabled", command=lambda: self.guard(self.apply_settings))
        self.apply_button.pack(side="left", padx=(0, 8))
        self.original_button = ttk.Button(actions, text="Nghe bản gốc", style="Simple.TButton",
                                          state="disabled", command=lambda: self.guard(self.play_original))
        self.original_button.pack(side="left", padx=(0, 8))
        self.extra_controls.extend([(self.apply_button, "normal"), (self.original_button, "normal")])
        self.folder_button = ttk.Button(actions, text="Mở nơi lưu", style="Simple.TButton",
                                        command=lambda: self.guard(self.open_output))
        self.folder_button.pack(side="left")
        comparison = ttk.Frame(frame, style="Simple.TFrame")
        comparison.pack(side="bottom", fill="x", pady=(8, 0), before=actions)
        self.result_name = tk.StringVar()
        self.result_box = ttk.Combobox(comparison, textvariable=self.result_name, state="readonly")
        self.result_box.pack(side="left", fill="x", expand=True)
        self.result_box.bind("<<ComboboxSelected>>", self.select_result)
        self.import_button = ttk.Button(comparison, text="Mở WAV để thử",
                                        command=lambda: self.guard(self.import_audio))
        self.import_button.pack(side="left", padx=(8, 0))
        self.extra_controls.extend([(self.result_box, "readonly"), (self.import_button, "normal")])
        self.status = tk.StringVar(value="Có thể tạo ngay bằng giọng mặc định; Tải giọng để chọn giọng khác.")
        self.status_label = ttk.Label(frame, textvariable=self.status, style="Simple.TLabel", wraplength=680)
        self.status_label.pack(side="bottom", anchor="w", pady=(12, 0), before=actions)
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.pack(side="bottom", fill="x", pady=(8, 0), before=self.status_label)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.after_id = root.after(80, self.drain)

    @staticmethod
    def scroll_page(parent):
        page = ttk.Frame(parent)
        canvas = tk.Canvas(page, background="#ffffff", highlightthickness=0)
        scrollbar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        body = ttk.Frame(canvas, style="Simple.TFrame", padding=12)
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        return page, body

    def guard(self, function):
        try:
            function()
        except Exception as exc:
            self.status.set(str(exc))
            messagebox.showerror("Đọc văn bản", str(exc), parent=self.root)

    def update_voices(self, pairs):
        current = self.voice_name.get()
        self.voice_ids = {DEFAULT_VOICE: None, **dict(pairs)}
        self.voice_box.configure(values=list(self.voice_ids))
        self.voice_name.set(current if current in self.voice_ids else DEFAULT_VOICE)

    def submit(self, function, callback):
        if self.busy or self.closing:
            raise RuntimeError("Hãy chờ tác vụ hiện tại hoàn tất.")
        self.busy = True
        self.voice_box.configure(state="disabled")
        self.load_button.configure(state="disabled")
        self.generate_button.configure(state="disabled")
        self.play_button.configure(state="disabled")
        for widget, _ in self.extra_controls:
            widget.configure(state="disabled")
        self.progress.start(12)
        self.status.set("Đang tải / tạo giọng / xử lý âm thanh…")
        def work():
            try:
                self.events.put((callback, function(), None))
            except Exception as exc:
                self.events.put((callback, None, exc))
        threading.Thread(target=work, daemon=True).start()

    def drain(self):
        try:
            callback, result, error = self.events.get_nowait()
        except queue.Empty:
            self.after_id = self.root.after(80, self.drain)
            return
        self.busy = False
        self.progress.stop()
        if self.closing:
            self.finish_close()
            return
        self.voice_box.configure(state="readonly")
        self.load_button.configure(state="normal")
        self.generate_button.configure(state="normal")
        self.play_button.configure(state="normal" if self.last_audio else "disabled")
        for widget, state in self.extra_controls:
            widget.configure(state=state)
        for widget in (self.apply_button, self.original_button):
            widget.configure(state="normal" if self.original_audio else "disabled")
        if error:
            self.guard(lambda: self.raise_error(error))
        else:
            self.guard(lambda: callback(result))
        self.after_id = self.root.after(80, self.drain)

    @staticmethod
    def raise_error(error):
        raise error

    def load_voices(self):
        def done(pairs):
            self.update_voices(pairs)
            self.status.set(f"Đã tải v3 Turbo · {len(pairs)} giọng.")
        self.submit(self.service.load, done)

    def generate(self):
        text = self.text.get("1.0", "end").strip()
        if not text:
            raise ValueError("Hãy nhập văn bản cần đọc.")
        voice = self.voice_ids[self.voice_name.get()]
        def done(path):
            self.original_audio = path
            self.add_result(path, "Gốc")
            self.apply_button.configure(state="normal")
            self.original_button.configure(state="normal")
            self.update_voices(self.service.voices())
            self.play_button.configure(state="normal")
            self.status.set(f"Đã lưu: {path}")
        self.submit(lambda: self.service.generate(text, voice), done)

    def options(self):
        return AudioOptions(
            tempo=TEMPO[self.tempo_name.get()], speed=float(self.speed.get()),
            bright=self.bright.get(), compress=self.compress.get(), peak_guard=self.peak_guard.get())

    def add_result(self, path, label):
        self.last_audio = Path(path)
        key = f"{label} · {self.last_audio.name}"
        self.results[key] = self.last_audio
        self.result_box.configure(values=list(self.results))
        self.result_name.set(key)
        self.play_button.configure(state="normal")

    def select_result(self, event=None):
        self.last_audio = self.results.get(self.result_name.get())

    def import_audio(self):
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("WAV", "*.wav")])
        if path:
            info = sf.info(path)
            if info.channels != 1 or info.frames == 0:
                raise ValueError("Chọn WAV mono có âm thanh.")
            self.original_audio = Path(path)
            self.add_result(self.original_audio, "WAV gốc")
            self.apply_button.configure(state="normal")
            self.original_button.configure(state="normal")
            self.status.set("Đã mở WAV. Chọn công nghệ rồi bấm Thử trên bản gốc.")

    def apply_settings(self):
        if not self.original_audio:
            raise ValueError("Hãy tạo hoặc mở bản gốc trước.")
        options = self.options()
        options.validate()
        source = self.original_audio
        def done(path):
            self.add_result(path, f"{options.tempo} / {options.speed:g}x")
            self.status.set(f"Đã lưu WAV + JSON: {path.name}")
        self.submit(lambda: self.service.variant(source, options), done)

    def play_original(self):
        if not self.original_audio or not self.original_audio.is_file():
            raise ValueError("File gốc không còn tồn tại.")
        self.open_file(self.original_audio)

    @staticmethod
    def open_file(path):
        if os.name == "nt":
            os.startfile(path)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])

    def play(self):
        if not self.last_audio or not self.last_audio.is_file():
            raise ValueError("Chưa có file audio để nghe.")
        self.open_file(self.last_audio)

    def open_output(self):
        self.service.output_dir.mkdir(parents=True, exist_ok=True)
        self.open_file(self.service.output_dir)

    def close(self):
        if self.closing:
            return
        self.closing = True
        if self.busy:
            # Let the worker release the SDK safely before ending the app.
            self.root.withdraw()
        else:
            self.finish_close()

    def finish_close(self):
        self.root.after_cancel(self.after_id)
        self.service.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Ứng dụng TTS tối giản riêng: v3 Turbo")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="Dựng giao diện ẩn, không tải model")
    args = parser.parse_args()
    root = tk.Tk()
    if args.check:
        root.withdraw()
    app = SimpleTTSApp(root, SimpleTTS(args.output_dir))
    if args.check:
        root.update_idletasks()
        print("Simple TTS ready: v3 Turbo; no model loaded")
        app.close()
    else:
        root.mainloop()


if __name__ == "__main__":
    main()
