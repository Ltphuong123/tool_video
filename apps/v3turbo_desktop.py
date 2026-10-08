"""Tkinter desktop UI for the v3 Turbo SDK."""
from __future__ import annotations

import argparse
import atexit
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from apps.desktop_audio import MicrophoneRecorder, PlaybackStopped, WavePlayer
from apps.v3turbo_tool import Cancelled, DEFAULT_MODEL, OUTPUT_KINDS, ROOT, Sampling, TurboTool, read_document


class DesktopApp:
    def __init__(self, root: tk.Tk, tool: TurboTool | None = None):
        self.root = root
        self.tool = tool or TurboTool(settings_path=ROOT / "outputs" / "v3turbo_desktop_settings.json")
        self.events = queue.Queue()
        self.busy = False
        self.closed = False
        self.player = None
        self.recorder = None
        self.job_process = None
        self.api_process = None
        self.audio_files = {}
        self.action_buttons = []
        self.voice_boxes = []
        self.vars = {}
        self.pages = {}
        self.nav_buttons = {}
        self.history_records = []
        root.title("VieNeu v3 Turbo — Studio giọng nói")
        root.geometry(f"{min(1280, root.winfo_screenwidth() - 60)}x{min(850, root.winfo_screenheight() - 100)}")
        root.minsize(980, 620)
        root.configure(background="#eef2f8")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#ffffff")
        style.configure("TLabelframe", background="#ffffff", bordercolor="#e1e7f0", borderwidth=1)
        style.configure("TLabelframe.Label", background="#ffffff", foreground="#19233f", font=("Segoe UI", 11, "bold"))
        style.configure("Shell.TFrame", background="#eef2f8")
        style.configure("Sidebar.TFrame", background="#101c32")
        style.configure("TLabel", background="#ffffff", foreground="#19233f", font=("Segoe UI", 10))
        style.configure("Muted.TLabel", foreground="#64718a")
        style.configure("Title.TLabel", font=("Segoe UI", 21, "bold"), foreground="#19233f")
        style.configure("Section.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Sidebar.TLabel", background="#101c32", foreground="#ffffff", font=("Segoe UI", 18, "bold"))
        style.configure("SidebarMuted.TLabel", background="#101c32", foreground="#9faccc", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 8), background="#f0f3fa",
                        foreground="#26324b", borderwidth=0)
        style.map("TButton", background=[("active", "#e4e9f4")])
        style.configure("Primary.TButton", background="#5b48ef", foreground="#ffffff", font=("Segoe UI", 10, "bold"))
        style.map("Primary.TButton", background=[("disabled", "#d8d2ff"), ("active", "#4836d2")],
                  foreground=[("disabled", "#7c719f")])
        style.configure("Danger.TButton", background="#fff0f0", foreground="#bd3445")
        style.map("Danger.TButton", background=[("active", "#ffe0e4")])
        style.configure("Nav.TButton", background="#101c32", foreground="#b5c1dc", anchor="w",
                        padding=(16, 12), font=("Segoe UI", 10))
        style.map("Nav.TButton", background=[("selected", "#293a5c"), ("active", "#1d2c47")],
                  foreground=[("selected", "#ffffff"), ("active", "#ffffff")])
        style.configure("TEntry", fieldbackground="#f8faff", bordercolor="#dce3ef", padding=7)
        style.configure("TCombobox", fieldbackground="#f8faff", padding=6, arrowsize=14)
        style.map("TCombobox", fieldbackground=[("readonly", "#f8faff")], foreground=[("readonly", "#26324b")])
        style.configure("TCheckbutton", background="#ffffff", font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", "#ffffff")])
        style.configure("Treeview", background="#ffffff", fieldbackground="#ffffff", foreground="#26324b",
                        rowheight=34, font=("Segoe UI", 10), borderwidth=0)
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), background="#f0f3fa", padding=9)
        style.map("Treeview", background=[("selected", "#ece9ff")], foreground=[("selected", "#392b98")])
        style.configure("Studio.TNotebook", background="#eef2f8", borderwidth=0, tabmargins=0)
        style.layout("Studio.TNotebook.Tab", [])
        style.configure("TNotebook.Tab", font=("Segoe UI", 10), padding=(15, 6))
        style.configure("Horizontal.TProgressbar", background="#5b48ef", troughcolor="#edf0f8", borderwidth=0, thickness=4)
        self.status = tk.StringVar(value="Chưa tải model. Chọn cấu hình rồi bấm Tải model.")
        shell = ttk.Frame(root, style="Shell.TFrame")
        shell.pack(fill="both", expand=True)
        self.sidebar = ttk.Frame(shell, width=190, style="Sidebar.TFrame", padding=(10, 22))
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        ttk.Label(self.sidebar, text="VieNeu", style="Sidebar.TLabel").pack(anchor="w", padx=12)
        ttk.Label(self.sidebar, text="v3 TURBO · VOICE STUDIO", style="SidebarMuted.TLabel").pack(anchor="w", padx=12, pady=(4, 25))
        self.navigation = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        self.navigation.pack(fill="x")
        ttk.Label(self.sidebar, text="48 kHz · Rubber Band\nTạo giọng trên máy của bạn", style="SidebarMuted.TLabel").pack(side="bottom", anchor="w", padx=12, pady=8)
        body = ttk.Frame(shell, style="Shell.TFrame", padding=(20, 12))
        body.pack(side="left", fill="both", expand=True)
        header = ttk.Frame(body, padding=(20, 10))
        header.pack(fill="x")
        headings = ttk.Frame(header)
        headings.pack(side="left", fill="x", expand=True)
        self.page_title = tk.StringVar(value="Văn bản")
        self.page_caption = tk.StringVar()
        ttk.Label(headings, textvariable=self.page_title, style="Title.TLabel").pack(anchor="w")
        ttk.Label(headings, textvariable=self.page_caption, style="Muted.TLabel").pack(anchor="w", pady=(4, 0))
        controls = ttk.Frame(header)
        controls.pack(side="right")
        self.model_state = tk.StringVar(value="Model chưa tải")
        ttk.Label(controls, textvariable=self.model_state, style="Muted.TLabel").pack(anchor="e", pady=(0, 7))
        ttk.Button(controls, text="Dừng tác vụ / phát", style="Danger.TButton", command=self.stop).pack(anchor="e")
        storage = ttk.Frame(body, padding=(16, 8))
        storage.pack(fill="x", pady=(10, 12))
        ttk.Label(storage, text="Nơi lưu", style="Muted.TLabel").pack(side="left", padx=(0, 12))
        self.output_location = tk.StringVar(value=str(self.tool.output_dir))
        ttk.Label(storage, textvariable=self.output_location, width=40).pack(side="left", fill="x", expand=True)
        self._button(storage, "Đổi thư mục…", self.choose_output_dir).pack_configure(pady=0)
        ttk.Button(storage, text="Mở", command=lambda: self._guard(self.open_output)).pack(side="left")
        self.book = ttk.Notebook(body, style="Studio.TNotebook")
        self.book.pack(fill="both", expand=True)
        self.book.bind("<<NotebookTabChanged>>", self._page_changed)
        self._config_tab()
        self._speech_tab()
        self._voices_tab()
        self._batch_tab()
        self._conversation_tab()
        self._srt_tab()
        self._video_tab()
        self._tools_tab()
        self._history_tab()
        footer = ttk.Frame(body, padding=(15, 10))
        footer.pack(side="bottom", fill="x", pady=(12, 0), before=self.book)
        ttk.Label(footer, textvariable=self.status, wraplength=900, style="Muted.TLabel").pack(anchor="w")
        self.progress = ttk.Progressbar(footer, mode="indeterminate")
        self.progress.pack(fill="x", pady=(6, 0))
        self.book.select(self.pages["Văn bản"])
        self.refresh_history()
        if getattr(self.tool, "settings_error", None):
            self._log(self.tool.settings_error)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._drain_handle = root.after(80, self._drain)
        atexit.register(self._terminate_children)

    def _tab(self, title, scroll=False):
        frame = ttk.Frame(self.book)
        self.book.add(frame, text=title)
        self.pages[title] = frame
        labels = {"Văn bản": "01   Đọc văn bản", "SRT": "02   Đọc phụ đề SRT", "Video": "03   Chỉnh tốc độ video",
                  "Hàng loạt": "04   Tạo hàng loạt", "Hội thoại": "05   Hội thoại", "Clone / Giọng": "06   Thư viện giọng",
                  "Kết quả / Log": "07   Lịch sử & file", "Cấu hình": "08   Cấu hình", "Fine-tune / API": "09   Fine-tune & API"}
        button = ttk.Button(self.navigation, text=labels[title], style="Nav.TButton", command=lambda: self.book.select(frame))
        self.nav_buttons[title] = button
        # The visual order follows the workflow, independent of widget creation.
        order = ["Văn bản", "SRT", "Video", "Hàng loạt", "Hội thoại", "Clone / Giọng", "Kết quả / Log", "Cấu hình", "Fine-tune / API"]
        for key in order:
            if key in self.nav_buttons:
                self.nav_buttons[key].pack_forget()
                self.nav_buttons[key].pack(fill="x", pady=2)
        if not scroll:
            content = ttk.Frame(frame, padding=16)
            content.pack(fill="both", expand=True)
            return content
        canvas = tk.Canvas(frame, highlightthickness=0, background="#ffffff")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        content = ttk.Frame(canvas, padding=20)
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        def wheel(event):
            canvas.yview_scroll(-1 if event.num == 4 else 1 if event.num == 5 else -int(event.delta / 120), "units")
            return "break"
        def bind_wheel(widget):
            widget.bind("<MouseWheel>", wheel, add="+")
            widget.bind("<Button-4>", wheel, add="+")
            widget.bind("<Button-5>", wheel, add="+")
            for child in widget.winfo_children():
                bind_wheel(child)
        self.root.after_idle(lambda: bind_wheel(content))
        return content

    def _page_changed(self, event=None):
        current = self.book.select()
        captions = {"Văn bản": "Viết nội dung, chọn giọng và tạo audio.", "SRT": "Giữ mốc phụ đề, tự căn tốc độ từng câu.",
                    "Video": "Đổi tốc độ một đoạn, chuyển êm ở hai đầu và giữ đồng bộ âm thanh.",
                    "Hàng loạt": "Tạo nhiều audio trong một lượt và xuất ZIP.", "Hội thoại": "Gán giọng theo nhân vật và điều chỉnh khoảng nghỉ.",
                    "Clone / Giọng": "Tạo và quản lý thư viện giọng riêng.", "Kết quả / Log": "Tìm, nghe, lưu bản sao và quản lý file đã tạo.",
                    "Cấu hình": "Model, thiết bị, tham số đọc và nơi lưu kết quả.", "Fine-tune / API": "Huấn luyện giọng và chạy API trên máy."}
        for title, frame in self.pages.items():
            selected = str(frame) == current
            self.nav_buttons[title].state(["selected" if selected else "!selected"])
            if selected:
                self.page_title.set("Lịch sử & file" if title == "Kết quả / Log" else title)
                self.page_caption.set(captions[title])

    def _field(self, parent, label, default="", choices=None, width=20):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=24).pack(side="left", padx=(0, 8))
        var = tk.StringVar(value=str(default))
        if choices:
            widget = ttk.Combobox(row, textvariable=var, values=choices, state="readonly", width=width)
        else:
            widget = ttk.Entry(row, textvariable=var, width=width)
        widget.pack(side="left", fill="x", expand=True)
        return var, widget, row

    def _button(self, parent, label, command, primary=False):
        button = ttk.Button(parent, text=label, style="Primary.TButton" if primary else "TButton", command=lambda: self._guard(command))
        button.pack(side="left", padx=(0, 7), pady=6)
        self.action_buttons.append(button)
        return button

    def _guard(self, command):
        try:
            command()
        except Exception as exc:
            self.status.set(str(exc))
            messagebox.showerror("VieNeu v3 Turbo", str(exc), parent=self.root)

    def _text(self, parent, height=12):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, pady=6)
        text = tk.Text(frame, height=height, wrap="word", font=("Segoe UI", 11), undo=True,
                       relief="flat", padx=14, pady=12, background="#f8faff", foreground="#19233f",
                       insertbackground="#5b48ef", selectbackground="#ded8ff", highlightthickness=1,
                       highlightbackground="#e1e7f0", highlightcolor="#9b8fff")
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)
        return text

    def _choose(self, variable, kind="file", types=None):
        if kind == "directory":
            value = filedialog.askdirectory(parent=self.root)
        else:
            value = filedialog.askopenfilename(parent=self.root, filetypes=types or [("Tất cả", "*.*")])
        if value:
            variable.set(value)

    def _path_field(self, parent, label, default="", kind="file", types=None):
        var, widget, row = self._field(parent, label, default, width=45)
        ttk.Button(row, text="Chọn…", command=lambda: self._choose(var, kind, types)).pack(side="left", padx=4)
        return var

    def _voice_field(self, parent, label="Giọng đọc"):
        var, box, row = self._field(parent, label, "", choices=[""], width=25)
        self.voice_boxes.append((var, box))
        return var

    def _config_tab(self):
        tab = self._tab("Cấu hình", scroll=True)
        destination = ttk.LabelFrame(tab, text="Lưu trữ kết quả", padding=14)
        destination.pack(fill="x", pady=(0, 18))
        self.output_dir_var = self._path_field(destination, "Thư mục lưu", str(self.tool.output_dir), kind="directory")
        actions = ttk.Frame(destination)
        actions.pack(fill="x")
        self._button(actions, "Áp dụng nơi lưu", self.apply_output_dir, primary=True)
        ttk.Label(destination, text="Áp dụng cho file tạo tiếp theo. Lịch sử giữ cả các thư mục đã dùng.",
                  style="Muted.TLabel").pack(anchor="w", pady=(5, 0))
        columns = ttk.Frame(tab)
        columns.pack(fill="x")
        left = ttk.LabelFrame(columns, text="Model và thiết bị", padding=14)
        right = ttk.LabelFrame(columns, text="Giọng đọc và đầu ra", padding=14)
        left.pack(side="left", fill="both", expand=True, padx=(0, 22))
        right.pack(side="left", fill="both", expand=True)
        orientation = [None]
        def arrange(event):
            horizontal = event.width >= 920
            if horizontal == orientation[0]:
                return
            orientation[0] = horizontal
            left.pack_forget()
            right.pack_forget()
            left.pack(side="left" if horizontal else "top", fill="both" if horizontal else "x",
                      expand=horizontal, padx=(0, 18) if horizontal else 0, pady=(0, 0) if horizontal else (0, 16))
            right.pack(side="left" if horizontal else "top", fill="both" if horizontal else "x", expand=horizontal)
        columns.bind("<Configure>", arrange)
        fields = [
            ("backbone_repo", "Model / thư mục đã merge", DEFAULT_MODEL, None),
            ("backend", "Backend", "auto", ["auto", "onnx", "pytorch"]),
            ("device", "Thiết bị", "auto", ["auto", "cpu", "cuda"]),
            ("dtype", "Kiểu số GPU", "auto", ["auto", "float32", "bfloat16"]),
            ("precision", "Độ chính xác ONNX", "fp32", ["fp32", "int8"]),
            ("model_subfolder", "Thư mục trọng số", "update", None),
            ("threads", "Số luồng CPU (0 = tự động)", 0, None),
            ("max_streams", "Số luồng streaming CUDA", 16, None),
            ("babble_retries", "Lượt thử lại câu ngắn", 2, None),
        ]
        for key, label, default, choices in fields:
            self.vars[key], _, _ = self._field(left, label, default, choices)
        self.vars["onnx_dir"] = self._path_field(left, "Thư mục ONNX (tùy chọn)", kind="directory")
        self.vars["onnx_repo"], _, _ = self._field(left, "Repo ONNX (tùy chọn)")
        self.vars["moss_tokenizer"], _, _ = self._field(left, "Codec MOSS", "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano")
        row = ttk.Frame(left)
        row.pack(fill="x")
        self._button(row, "Tải model", self.load_model, primary=True)
        self._button(row, "Giải phóng model", lambda: self._job(self.tool.unload, lambda _: self._unloaded()))
        ttk.Label(left, text="Lần tải đầu cần Internet để lấy trọng số. CPU dùng ONNX; GPU dùng CUDA.\n"
                  "Model fine-tune đã merge: chọn PyTorch/CUDA.\nINT8 cần CPU tương thích để tránh méo tiếng.",
                  wraplength=480).pack(anchor="w", pady=12)
        labels = {"speed": "Tốc độ đọc (0.5–2.0x)", "temperature": "Temperature", "top_k": "Top K", "top_p": "Top P",
                  "repetition_penalty": "Phạt lặp", "repetition_window": "Cửa sổ phạt lặp",
                  "max_new_frames": "Giới hạn frame / đoạn", "max_chars": "Ký tự / đoạn",
                  "batch_size": "Số mục / batch"}
        defaults = Sampling()
        for key, label in labels.items():
            self.vars[key], _, _ = self._field(right, label, getattr(defaults, key))
        ttk.Label(right, text="Chỉnh tốc độ bằng Rubber Band, giữ cao độ.").pack(anchor="w", pady=6)
        for key, label, value in [("denoise", "Khử nhiễu audio mẫu", True),
                                  ("use_ref_codes", "Dùng mã audio tham chiếu", True),
                                  ("apply_watermark", "Gắn watermark (cần cài thêm)", False)]:
            self.vars[key] = tk.BooleanVar(value=value)
            ttk.Checkbutton(right, text=label, variable=self.vars[key]).pack(anchor="w", pady=6)
        self.format, _, _ = self._field(right, "Định dạng xuất", "wav", ["wav", "flac", "mp3"])
        ttk.Label(right, text="Cấu hình áp dụng cho văn bản, audio + SRT từng câu, batch, hội thoại và streaming.\n"
                  "Tab SRT có tốc độ ban đầu riêng.\n"
                  "Streaming luôn lưu WAV. MP3 phụ thuộc libsndfile của máy.\n"
                  "Tốc độ: 0.8x chậm, 1.0x gốc, 1.2x nhanh; giữ cao độ.\n"
                  "Ưu tiên 0.85–1.15x để hạn chế biến đổi chất giọng.\n"
                  "Streaming khác 1.0x chờ tạo xong rồi chỉnh tốc độ và phát.\n"
                  "Phong cách đọc theo giọng tham chiếu; style/instructions chưa được điều khiển.",
                  wraplength=460).pack(anchor="w", pady=12)

    def _speech_tab(self):
        tab = self._tab("Văn bản")
        self.voice = self._voice_field(tab)
        self.reference = self._path_field(tab, "Audio tham chiếu (clone)", types=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("Tất cả", "*.*")])
        ttk.Button(tab, text="Bỏ audio clone", command=lambda: self.reference.set("")).pack(anchor="w")
        self.speech = self._text(tab)
        self.speech.insert("1.0", "Xin chào! Đây là công cụ chuyển văn bản thành giọng nói bằng VieNeu v3 Turbo.")
        row = ttk.Frame(tab)
        row.pack(side="bottom", fill="x", before=self.speech.master)
        ttk.Button(row, text="Nhập TXT / PDF", command=lambda: self._guard(self.import_document)).pack(side="left", padx=(0, 7))
        self._button(row, "Tạo audio", self.synthesize, primary=True)
        self._button(row, "Tạo audio + SRT từng câu", self.synthesize_with_subtitles)
        self._button(row, "Streaming và nghe", self.stream)
        ttk.Label(tab, text="Tag thử nghiệm: [cười]  [thở dài]  [hắng giọng]. Audio gốc 48 kHz; hỗ trợ Việt / Anh / Anh–Việt.",
                  wraplength=700, style="Muted.TLabel").pack(side="bottom", anchor="w", pady=5, before=row)

    def _voices_tab(self):
        tab = self._tab("Clone / Giọng", scroll=True)
        ttk.Label(tab, text="Dùng audio sạch khoảng 3–8 giây của giọng muốn clone; không cần chép lại lời thoại.",
                  wraplength=1000).pack(anchor="w", pady=8)
        self.clone_path = self._path_field(tab, "Audio mẫu", types=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("Tất cả", "*.*")])
        self.clone_name, _, _ = self._field(tab, "Tên giọng riêng")
        self.clone_desc, _, _ = self._field(tab, "Mô tả")
        row = ttk.Frame(tab)
        row.pack(fill="x")
        self.record_button = ttk.Button(row, text="Ghi micro", command=lambda: self._guard(self.record_microphone))
        self.record_button.pack(side="left", padx=(0, 7), pady=6)
        self.action_buttons.append(self.record_button)
        self._button(row, "Clone và lưu giọng", self.add_voice, primary=True)
        self._button(row, "Khử nhiễu audio mẫu", self.denoise)
        self._button(row, "Xuất embedding / codes", self.export_reference)
        self.saved_voice = self._voice_field(tab, "Giọng muốn quản lý")
        row = ttk.Frame(tab)
        row.pack(fill="x")
        self._button(row, "Xóa giọng riêng", self.delete_voice)
        self._button(row, "Xuất thư viện giọng JSON", lambda: self._job(self.tool.export_voices, self._result))
        self._button(row, "Nhập giọng từ JSON", self.import_voices)
        ttk.Label(tab, text=f"Giọng riêng tự nạp lại khi tải model. File lưu: {self.tool.voices_path}",
                  wraplength=1000).pack(anchor="w", pady=12)
        self.voice_details = self._text(tab, height=8)
        self.voice_details.configure(state="disabled")

    def _batch_tab(self):
        tab = self._tab("Hàng loạt")
        self.batch_voice = self._voice_field(tab)
        ttk.Label(tab, text="Mỗi dòng là một mục, hoặc chọn nhiều file TXT/PDF. Kết quả ZIP chứa audio và manifest.json.").pack(anchor="w")
        self.batch_text = self._text(tab)
        self.batch_files = []
        self.batch_file_label = tk.StringVar(value="Chưa chọn file; sẽ dùng các dòng bên trên.")
        summary = ttk.Label(tab, textvariable=self.batch_file_label, style="Muted.TLabel")
        summary.pack(side="bottom", anchor="w", before=self.batch_text.master)
        row = ttk.Frame(tab)
        row.pack(side="bottom", fill="x", before=summary)
        ttk.Button(row, text="Chọn nhiều file", command=self.choose_batch).pack(side="left", padx=(0, 7))
        ttk.Button(row, text="Bỏ file đã chọn", command=self.clear_batch).pack(side="left", padx=(0, 7))
        self._button(row, "Tạo batch và ZIP", self.batch, primary=True)

    def _conversation_tab(self):
        tab = self._tab("Hội thoại")
        ttk.Label(tab, text="Mỗi dòng: Nhân vật: lời thoại. Thêm nhân vật và chọn giọng phía dưới.").pack(anchor="w")
        self.script = self._text(tab, height=9)
        self.script.insert("1.0", "An: Xin chào, hôm nay bạn thế nào?\nBình: Mình rất vui. [cười] Cảm ơn bạn!")
        editor_frame = self.script.master
        controls = ttk.Frame(tab)
        controls.pack(side="bottom", fill="x", before=editor_frame)
        row = ttk.Frame(controls)
        row.pack(fill="x")
        self.actor = tk.StringVar(value="An")
        ttk.Entry(row, textvariable=self.actor, width=20).pack(side="left", padx=5)
        self.actor_voice = tk.StringVar()
        box = ttk.Combobox(row, textvariable=self.actor_voice, state="readonly", width=28)
        box.pack(side="left", padx=5)
        self.voice_boxes.append((self.actor_voice, box))
        ttk.Button(row, text="Gán giọng", command=lambda: self._guard(self.assign_actor)).pack(side="left")
        ttk.Button(row, text="Xóa nhân vật", command=self.remove_actor).pack(side="left", padx=5)
        self.actors = ttk.Treeview(controls, columns=("actor", "voice"), show="headings", height=3)
        self.actors.heading("actor", text="Nhân vật")
        self.actors.heading("voice", text="Giọng")
        self.actors.pack(fill="x", pady=7)
        self.gap, _, _ = self._field(controls, "Khoảng nghỉ giữa lượt (giây)", 0.4)
        row = ttk.Frame(controls)
        row.pack(fill="x")
        self._button(row, "Tạo hội thoại", self.conversation, primary=True)

    def _srt_tab(self):
        tab = self._tab("SRT", scroll=True)
        self.srt_path = self._path_field(tab, "File phụ đề SRT", types=[("Phụ đề SRT", "*.srt")])
        self.srt_voice = self._voice_field(tab)
        self.srt_min_speed, _, _ = self._field(tab, "Tốc độ ban đầu / tối thiểu (x)", 1.0)
        ttk.Label(tab, text="Chế độ SRT: luôn giữ đúng mốc bắt đầu và mọi khoảng lặng.",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=12)
        ttk.Label(tab, text="Tốc độ SRT riêng: 0.8x chậm, 1.0x gốc, 1.2x nhanh; cho phép 0.5–2.0x.\n"
                  "Bỏ khoảng lặng model sinh đầu/cuối, đặt lời đọc đúng mốc, chỉ tăng tốc khi vượt khung.\n"
                  "Câu ngắn giữ tốc độ ban đầu; khoảng trống còn lại là im lặng, không lùi câu sau.\n"
                  "Giữ cả khoảng lặng trước câu đầu và khoảng nghỉ dài giữa các câu.\n"
                  "Rubber Band giữ cao độ; làm mềm 5 ms cuối câu để giảm tiếng tách.\n"
                  "Phụ đề chồng mốc: câu trước kết thúc trước khi câu sau bắt đầu. Tốc độ ép cao có thể kém tự nhiên.",
                  wraplength=1000).pack(anchor="w")
        row = ttk.Frame(tab)
        row.pack(fill="x")
        self._button(row, "Tạo audio từ SRT", self.srt, primary=True)

    def _video_tab(self):
        tab = self._tab("Video", scroll=True)
        self.video_path = self._path_field(tab, "Video đầu vào", types=[("Video", "*.mp4 *.mov *.mkv *.avi *.webm"), ("Tất cả", "*.*")])
        self.video_details = tk.StringVar(value="Chọn video, rồi đọc thông tin để xem thời lượng và FPS.")
        ttk.Label(tab, textvariable=self.video_details, style="Muted.TLabel", wraplength=700).pack(anchor="w", pady=(4, 6))
        row = ttk.Frame(tab)
        row.pack(fill="x")
        self._button(row, "Đọc thông tin video", self.inspect_video)
        settings = ttk.LabelFrame(tab, text="Đoạn cần thay đổi tốc độ", padding=14)
        settings.pack(fill="x", pady=(12, 10))
        self.video_start, _, _ = self._field(settings, "Bắt đầu trong video gốc (s)", 0)
        self.video_end, _, _ = self._field(settings, "Kết thúc trong video gốc (s)", 10)
        self.video_speed, _, _ = self._field(settings, "Tốc độ đoạn (0.25–4.0x)", 1.5)
        self.video_ramp, _, _ = self._field(settings, "Chuyển tốc độ mỗi đầu (s)", 0.5)
        self.video_curve = tk.Canvas(settings, height=95, background="#f8faff", highlightthickness=0)
        self.video_curve.pack(fill="x", pady=(10, 0))
        self.video_curve.bind("<Configure>", lambda event: self._draw_video_curve())
        for variable in (self.video_start, self.video_end, self.video_speed, self.video_ramp):
            variable.trace_add("write", lambda *_: self._draw_video_curve())
        self.video_keep_audio = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text="Giữ tiếng gốc và đổi tốc độ đồng bộ · Rubber Band giữ cao độ",
                        variable=self.video_keep_audio).pack(anchor="w", pady=6)
        self.video_quality, _, _ = self._field(tab, "Chất lượng MP4 · CRF", 20, ["18", "20", "23", "26", "28"])
        self.video_preset, _, _ = self._field(tab, "Tốc độ xuất video", "fast", ["faster", "fast", "medium"])
        ttk.Label(tab, text="Mốc và thời gian chuyển tính theo video gốc. Chuyển 0 giây để đổi ngay; "
                  "tối đa nửa độ dài đoạn. Các phần khác giữ tốc độ 1.0x.\n"
                  "Độ mượt khi làm chậm phụ thuộc FPS của nguồn.\n"
                  "Xuất MP4 mới trong nơi lưu đang chọn; video đầu vào được giữ lại. Không cần tải model TTS.",
                  wraplength=700, style="Muted.TLabel").pack(anchor="w", pady=(10, 4))
        page = self.pages["Video"]
        row = ttk.Frame(page, padding=(16, 4))
        row.pack(side="bottom", fill="x", before=page.winfo_children()[0])
        self.video_export_button = self._button(row, "Xuất video", self.export_video, primary=True)

    def _draw_video_curve(self):
        canvas = self.video_curve
        canvas.delete("all")
        width = max(240, canvas.winfo_width())
        try:
            start, end, speed, ramp = (float(variable.get()) for variable in
                                       (self.video_start, self.video_end, self.video_speed, self.video_ramp))
            if not all(math.isfinite(value) for value in (start, end, speed, ramp)) or end <= start or speed <= 0:
                return
        except ValueError:
            return
        fraction = min(0.5, max(0, ramp / (end - start)))
        low, high = min(1, speed), max(1, speed)
        span = max(0.5, high - low)
        def y(value):
            return 62 - (value - low) / span * 40
        coordinates = []
        for index in range(161):
            position = index / 160
            local = (position - 0.15) / 0.7
            if local < 0 or local > 1:
                value = 1
            elif fraction and (local < fraction or local > 1 - fraction):
                amount = min(local, 1 - local) / fraction
                value = 1 + (speed - 1) * amount * amount * (3 - 2 * amount)
            else:
                value = speed
            coordinates.extend((16 + position * (width - 32), y(value)))
        canvas.create_line(16, y(1), width - 16, y(1), fill="#d7dfea", dash=(3, 3))
        canvas.create_line(*coordinates, fill="#5b48ef", width=2, smooth=False)
        canvas.create_text(20, 83, text="1.0x", anchor="w", fill="#64718a", font=("Segoe UI", 9))
        canvas.create_text(width / 2, 83, text=f"Đoạn đã chọn · {speed:g}x", fill="#5b48ef", font=("Segoe UI", 9))
        canvas.create_text(width - 20, 83, text="1.0x", anchor="e", fill="#64718a", font=("Segoe UI", 9))

    def inspect_video(self):
        path = self.video_path.get().strip()
        if not path:
            raise ValueError("Hãy chọn video đầu vào.")
        def run():
            from apps.video_editor import probe_video
            with self.tool.operation(require_model=False):
                self.tool.check_stop()
                return probe_video(path)
        def done(info):
            self.video_details.set(f"{info['width']} × {info['height']} · {info['fps']:g} FPS · "
                                   f"{info['duration']:.3f} giây · {'có tiếng gốc' if info['has_audio'] else 'không có tiếng gốc'}")
            self.video_start.set("0")
            self.video_end.set(f"{info['duration']:.6f}")
            self.video_ramp.set(str(min(0.5, info["duration"] / 2)))
            self.status.set("Đã đọc thông tin video. Chọn mốc bắt đầu/kết thúc để đổi tốc độ một đoạn.")
        self._job(run, done)

    def export_video(self):
        path = self.video_path.get().strip()
        if not path:
            raise ValueError("Hãy chọn video đầu vào.")
        start, end, speed, ramp = (float(variable.get()) for variable in
                                  (self.video_start, self.video_end, self.video_speed, self.video_ramp))
        if not all(math.isfinite(value) for value in (start, end, speed, ramp)):
            raise ValueError("Mốc thời gian và tốc độ video phải là số hữu hạn.")
        if start < 0 or end <= start or not 0.25 <= speed <= 4 or not 0 <= ramp <= (end - start) / 2:
            raise ValueError("Kiểm tra mốc bắt đầu/kết thúc, tốc độ 0.25–4.0x và thời gian chuyển không vượt nửa đoạn.")
        quality, preset, keep_audio = int(self.video_quality.get()), self.video_preset.get(), self.video_keep_audio.get()
        self._job(lambda: self.tool.edit_video(path, start, end, speed=speed, ramp_seconds=ramp,
                  keep_audio=keep_audio, quality=quality, preset=preset,
                  progress=lambda fraction: self._post("status", f"Đang xử lý video · {fraction * 100:.0f}%")), self._result)

    def _tools_tab(self):
        tab = self._tab("Fine-tune / API", scroll=True)
        ttk.Label(tab, text="Fine-tune một giọng bằng LoRA", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.training = {}
        self.training["task"], _, _ = self._field(tab, "Bước thực hiện", "Chuẩn bị dữ liệu", ["Chuẩn bị dữ liệu", "Train LoRA", "Merge LoRA", "Đóng gói giọng"])
        for key, label, default, kind in [
            ("dataset", "Thư mục dataset", str(ROOT / "finetune" / "dataset"), "directory"),
            ("data", "Dữ liệu train parquet / jsonl", "", "file"),
            ("adapter", "Thư mục adapter", "", "directory"),
            ("out", "Thư mục xuất / model merge", str(ROOT / "finetune" / "output"), "directory"),
            ("audio", "Audio đóng gói giọng", "", "file"),
        ]:
            self.training[key] = self._path_field(tab, label, default, kind)
        for key, label, default in [("run", "Tên lượt train / tên giọng", "my_voice"),
                                    ("epochs", "Epoch", 3), ("rank", "LoRA rank", 16)]:
            self.training[key], _, _ = self._field(tab, label, default)
        row = ttk.Frame(tab)
        row.pack(fill="x")
        self._button(row, "Chạy bước đã chọn", self.training_job)
        ttk.Label(tab, text="Cài uv sync --extra finetune trước khi train. Dataset: metadata.csv và raw_audio/.\n"
                  "Tool giải phóng model đang dùng trước khi train/merge. Log xuất trong tab Kết quả.", wraplength=1050).pack(anchor="w", pady=5)
        ttk.Separator(tab).pack(fill="x", pady=8)
        self.api_backend, _, _ = self._field(tab, "API model gốc · thiết bị", "cpu", ["cpu", "cuda"])
        self.api_port, _, _ = self._field(tab, "Cổng API trên máy", 8000)
        self.api_key, key_widget, _ = self._field(tab, "API key (tùy chọn)")
        key_widget.configure(show="•")
        row = ttk.Frame(tab)
        row.pack(fill="x")
        self._button(row, "Khởi chạy API", self.start_api)
        ttk.Button(row, text="Dừng API", command=self.stop_api).pack(side="left")
        ttk.Label(tab, text="API dùng model gốc tại http://127.0.0.1:<cổng>/v1/audio/speech.\n"
                  "Tiến trình riêng, dùng thư viện apps.openai_speech; trạng thái tải/ready xem trong log.", wraplength=1050).pack(anchor="w")

    def _history_tab(self):
        tab = self._tab("Kết quả / Log")
        filters = ttk.Frame(tab)
        filters.pack(fill="x", pady=(0, 10))
        ttk.Label(filters, text="Tìm file", style="Muted.TLabel").pack(side="left", padx=(0, 8))
        self.history_query = tk.StringVar()
        ttk.Entry(filters, textvariable=self.history_query, width=16).pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.history_kind = tk.StringVar(value="Tất cả")
        ttk.Combobox(filters, textvariable=self.history_kind, width=20, state="readonly",
                     values=["Tất cả", *dict.fromkeys(OUTPUT_KINDS.values())]).pack(side="left", padx=(0, 10))
        ttk.Button(filters, text="Làm mới", command=lambda: self._guard(self.refresh_history)).pack(side="left")
        self.history_count = tk.StringVar(value="0 file")
        ttk.Label(filters, textvariable=self.history_count, style="Muted.TLabel", width=10).pack(side="left", padx=(8, 0))
        split = ttk.Notebook(tab)
        split.pack(fill="both", expand=True)
        results = ttk.Frame(split)
        split.add(results, text="File đã tạo")
        table = ttk.Frame(results)
        table.pack(fill="both", expand=True)
        self.history = ttk.Treeview(table, columns=("name", "kind", "modified", "size"), show="headings",
                                    height=7, selectmode="extended")
        for key, title, width in [("name", "Tên file", 290), ("kind", "Loại", 110),
                                  ("modified", "Thời gian", 150), ("size", "Dung lượng", 100)]:
            self.history.heading(key, text=title)
            self.history.column(key, width=width, minwidth=80, stretch=key == "name")
        scrollbar = ttk.Scrollbar(table, command=self.history.yview)
        self.history.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.history.pack(fill="both", expand=True)
        self.history.tag_configure("alternate", background="#f8faff")
        self.history_detail = tk.StringVar(value="Chọn file để xem đường dẫn. Ctrl / Shift để chọn nhiều file.")
        detail = ttk.Label(results, textvariable=self.history_detail, wraplength=700, style="Muted.TLabel")
        detail.pack(side="bottom", anchor="w", pady=8, before=table)
        actions = ttk.Frame(results)
        actions.pack(side="bottom", fill="x", before=detail)
        self._button(actions, "Mở / nghe", self.play_selected, primary=True)
        self._button(actions, "Lưu thành…", self.save_selected)
        self._button(actions, "Lưu cặp audio + SRT…", self.save_subtitle_pair)
        ttk.Button(actions, text="Mở thư mục file", command=lambda: self._guard(self.open_selected_folder)).pack(side="left", padx=5)
        management = ttk.Frame(results)
        management.pack(side="bottom", fill="x", pady=(0, 6), before=actions)
        ttk.Button(management, text="Chọn tất cả", command=lambda: self.history.selection_set(self.history.get_children())).pack(side="left", padx=(0, 10))
        self.delete_with_subtitles = tk.BooleanVar(value=True)
        ttk.Checkbutton(management, text="Xóa kèm SRT cùng tên", variable=self.delete_with_subtitles).pack(side="left")
        self.delete_files_button = self._button(management, "Xóa file đã chọn", self.delete_selected_files)
        self.delete_files_button.configure(style="Danger.TButton")
        logs = ttk.Frame(split)
        split.add(logs, text="Nhật ký hoạt động")
        ttk.Label(logs, text="Nhật ký hoạt động", style="Section.TLabel").pack(anchor="w", pady=(12, 0))
        self.log = self._text(logs, height=4)
        self.log.configure(state="disabled")
        self.history.bind("<<TreeviewSelect>>", lambda event: self._history_selection())
        self.history.bind("<Double-1>", lambda event: self._guard(self.open_selected_folder))
        self.history_query.trace_add("write", lambda *_: self._filter_history())
        self.history_kind.trace_add("write", lambda *_: self._filter_history())

    @staticmethod
    def _file_size(size):
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
            size /= 1024

    def refresh_history(self, selected=None):
        keep = [Path(selected)] if selected is not None else [self.audio_files[item] for item in self.history.selection() if item in self.audio_files]
        self.history_records = self.tool.list_outputs()
        self._filter_history(keep)

    def _filter_history(self, selected=None):
        if selected is None:
            selected = [self.audio_files[item] for item in self.history.selection() if item in self.audio_files]
        self.history.delete(*self.history.get_children())
        self.audio_files.clear()
        query, kind = self.history_query.get().strip().casefold(), self.history_kind.get()
        chosen = []
        for record in self.history_records:
            if query and query not in str(record.path).casefold():
                continue
            if kind != "Tất cả" and record.kind != kind:
                continue
            item = self.history.insert("", "end", values=(record.path.name, record.kind,
                                       datetime.fromtimestamp(record.modified).strftime("%d/%m/%Y %H:%M"),
                                       self._file_size(record.size)), tags=("alternate",) if len(self.audio_files) % 2 else ())
            self.audio_files[item] = record.path
            if record.path in selected:
                chosen.append(item)
        if chosen:
            self.history.selection_set(chosen)
            self.history.see(chosen[0])
        self.history_count.set(f"{len(self.audio_files)} / {len(self.history_records)} file")
        self._history_selection()

    def _history_selection(self):
        paths = [self.audio_files[item] for item in self.history.selection() if item in self.audio_files]
        self.history_detail.set(str(paths[0]) if len(paths) == 1 else f"Đã chọn {len(paths)} file." if paths else
                                "Chọn file để xem đường dẫn. Ctrl / Shift để chọn nhiều file.")

    def _ensure_storage_idle(self):
        if self.busy or self.recorder is not None:
            raise RuntimeError("Hãy dừng hoặc chờ tác vụ và bản ghi hiện tại trước khi đổi nơi lưu hay xóa file.")

    def choose_output_dir(self):
        self._ensure_storage_idle()
        folder = filedialog.askdirectory(parent=self.root, title="Chọn nơi lưu file được tạo", initialdir=self.tool.output_dir)
        if folder:
            self.output_dir_var.set(folder)
            self.apply_output_dir()

    def apply_output_dir(self):
        self._ensure_storage_idle()
        directory = self.tool.set_output_dir(self.output_dir_var.get())
        self.output_location.set(str(directory))
        self.output_dir_var.set(str(directory))
        self.refresh_history()
        self.status.set(f"Đã đổi nơi lưu: {directory}. File mới sẽ lưu tại đây.")
        self._log(f"Nơi lưu kết quả: {directory}")

    def delete_selected_files(self):
        self._ensure_storage_idle()
        paths = [self.audio_files[item] for item in self.history.selection()]
        if not paths:
            raise ValueError("Hãy chọn file muốn xóa trong lịch sử.")
        include_subtitles = self.delete_with_subtitles.get()
        paired = [path.with_suffix(".srt") for path in paths if include_subtitles and path.suffix.lower() in (".wav", ".flac", ".mp3")]
        known = {record.path for record in self.history_records}
        targets = list(dict.fromkeys(paths + [path for path in paired if path in known]))
        preview = "\n".join(path.name for path in targets[:6])
        if len(targets) > 6:
            preview += f"\n… và {len(targets) - 6} file khác"
        if not messagebox.askyesno("Xóa file đã tạo", f"Xóa vĩnh viễn {len(targets)} file?\n\n{preview}\n\nFile sẽ bị xóa khỏi máy.", parent=self.root):
            return
        try:
            removed = self.tool.delete_outputs(paths, include_subtitles=include_subtitles)
        finally:
            self.refresh_history()
        self.status.set(f"Đã xóa {len(removed)} file.")
        self._log(f"Đã xóa {len(removed)} file: " + ", ".join(path.name for path in removed))

    def _post(self, kind, value):
        self.events.put((kind, value))

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _drain(self):
        if self.closed:
            return
        for _ in range(100):
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status.set(value)
            elif kind == "log":
                self._log(value)
            elif kind == "done":
                self.busy = False
                self.progress.stop()
                for button in self.action_buttons:
                    button.configure(state="normal")
                result, callback, error = value
                self.refresh_voices()
                if error:
                    self.refresh_history()
                    self.status.set(str(error))
                    self._log(str(error))
                    if not isinstance(error, (Cancelled, PlaybackStopped)):
                        messagebox.showerror("VieNeu v3 Turbo", str(error), parent=self.root)
                else:
                    self._guard(lambda: callback(result))
        self._drain_handle = self.root.after(80, self._drain)

    def _job(self, function, callback=lambda result: None):
        if self.busy:
            raise RuntimeError("Đang có tác vụ chạy. Hãy chờ hoặc bấm Dừng.")
        if self.recorder is not None:
            raise RuntimeError("Hãy dừng và lưu bản ghi micro trước.")
        if self.api_process is not None and self.api_process.poll() is None:
            raise RuntimeError("Hãy dừng API trước khi chạy tác vụ desktop.")
        self.busy = True
        self.tool.stop_event.clear()
        self.progress.start(12)
        for button in self.action_buttons:
            button.configure(state="disabled")
        self.status.set("Đang xử lý…")
        def run():
            try:
                result = function()
                self._post("done", (result, callback, None))
            except Exception as exc:
                self._post("done", (None, callback, exc))
        threading.Thread(target=run, daemon=True).start()

    def _sampling(self):
        values = {}
        defaults = Sampling()
        for name in Sampling.__dataclass_fields__:
            var = self.vars[name]
            default = getattr(defaults, name)
            values[name] = bool(var.get()) if isinstance(default, bool) else type(default)(var.get())
        sampling = Sampling(**values)
        sampling.kwargs()
        return sampling

    def load_model(self):
        kw = {key: self.vars[key].get().strip() for key in ["backbone_repo", "backend", "device", "dtype", "precision", "model_subfolder", "moss_tokenizer"]}
        for key in ["threads", "max_streams", "babble_retries"]:
            kw[key] = int(self.vars[key].get())
        if kw["threads"] < 0 or not 1 <= kw["max_streams"] <= 32 or not 0 <= kw["babble_retries"] <= 5:
            raise ValueError("Luồng CPU >= 0, streaming 1–32, lượt thử lại 0–5.")
        for key in ["onnx_dir", "onnx_repo"]:
            kw[key] = self.vars[key].get().strip() or None
        kw["max_batch_size"] = self._sampling().batch_size
        if not kw["backbone_repo"]:
            raise ValueError("Hãy nhập model hoặc thư mục model.")
        self._job(lambda: self.tool.load(**kw), self._loaded)


    def _loaded(self, message):
        self.status.set(message)
        self._log(message)
        self.refresh_voices()

    def _unloaded(self):
        self.status.set("Đã giải phóng model.")
        self.refresh_voices()

    def refresh_voices(self):
        names = self.tool.voice_names()
        default = getattr(self.tool.tts, "_default_voice", "") if self.tool.tts else ""
        for variable, box in self.voice_boxes:
            box.configure(values=names)
            if variable.get() not in names:
                variable.set(default if default in names else (names[0] if names else ""))
        details = []
        if self.tool.tts:
            for name in names:
                voice = self.tool.tts._preset_voices[name]
                details.append(f"{name} — {voice.get('description', '')}")
        self.voice_details.configure(state="normal")
        self.voice_details.delete("1.0", "end")
        self.voice_details.insert("1.0", "\n".join(details))
        self.voice_details.configure(state="disabled")
        self.model_state.set(f"{self.tool.tts.backend.upper()} · {len(names)} giọng" if self.tool.tts else "Model chưa tải")

    def import_document(self):
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("Văn bản", "*.txt *.md *.pdf")])
        if path:
            def done(text):
                self.speech.delete("1.0", "end")
                self.speech.insert("1.0", text)
                self.status.set(f"Đã nhập {len(text):,} ký tự.")
            self._job(lambda: read_document(path), done)

    def _result(self, result):
        path, note = result if isinstance(result, tuple) else (result, "")
        path = Path(path)
        self.history_query.set("")
        self.history_kind.set("Tất cả")
        self.refresh_history(selected=path)
        item = next((item for item, source in self.audio_files.items() if source == path), None)
        self.status.set(f"Đã lưu: {path.name}. {note}")
        self._log(f"{path}\n{note}")
        self.book.select(self.pages["Kết quả / Log"])
        return item

    def synthesize(self):
        text = self.speech.get("1.0", "end").strip()
        voice, ref, sampling, fmt = self.voice.get(), self.reference.get().strip() or None, self._sampling(), self.format.get()
        self._job(lambda: self.tool.synthesize(text, voice, ref, sampling, fmt), self._result)

    def synthesize_with_subtitles(self):
        text = self.speech.get("1.0", "end").strip()
        voice, ref, sampling, fmt = self.voice.get(), self.reference.get().strip() or None, self._sampling(), self.format.get()
        def done(result):
            self._result(result.audio)
            self._log(str(result.subtitles))
            self.status.set(f"Đã lưu audio + SRT · {result.sentences} câu · {result.duration:.1f} giây.")
        self._job(lambda: self.tool.synthesize_with_subtitles(
            text, voice, ref, sampling, fmt, lambda msg: self._post("status", msg)), done)

    def stream(self):
        text = self.speech.get("1.0", "end").strip()
        voice, ref, sampling = self.voice.get(), self.reference.get().strip() or None, self._sampling()
        def run():
            player = WavePlayer()
            self.player = player
            result = None
            try:
                generator = self.tool.stream(text, voice, ref, sampling, player.write)
                try:
                    for path, status in generator:
                        self._post("status", status)
                        if path:
                            result = path
                finally:
                    generator.close()
                player.finish()
                return result
            finally:
                player.close()
                self.player = None
        self._job(run, self._result)

    def add_voice(self):
        args = (self.clone_name.get(), self.clone_path.get(), self.clone_desc.get(),
                bool(self.vars["denoise"].get()), bool(self.vars["use_ref_codes"].get()))
        self._job(lambda: self.tool.add_voice(*args), lambda _: self._loaded("Đã clone và lưu giọng riêng."))

    def delete_voice(self):
        name = self.saved_voice.get()
        if messagebox.askyesno("Xóa giọng riêng", f"Xóa giọng '{name}' khỏi thư viện riêng?", parent=self.root):
            self._job(lambda: self.tool.delete_voice(name), lambda _: self._loaded("Đã xóa giọng riêng."))

    def denoise(self):
        path = self.clone_path.get()
        self._job(lambda: self.tool.denoise(path), self._result)

    def export_reference(self):
        path, denoise = self.clone_path.get(), bool(self.vars["denoise"].get())
        self._job(lambda: self.tool.export_reference(path, denoise), self._result)

    def import_voices(self):
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("Thư viện giọng JSON", "*.json")])
        if path:
            self._job(lambda: self.tool.import_voices(path), self._loaded)

    def record_microphone(self):
        if self.recorder is None:
            recorder = MicrophoneRecorder()
            recorder.start()
            self.recorder = recorder
            self.record_button.configure(text="Dừng và lưu bản ghi")
            self.status.set("Đang ghi micro. Đọc khoảng 3–8 giây rồi bấm Dừng và lưu.")
        else:
            recorder, self.recorder = self.recorder, None
            self.record_button.configure(text="Ghi micro")
            path = recorder.save(self.tool._path("microphone", ".wav"))
            self.clone_path.set(str(path))
            self.reference.set(str(path))
            self.refresh_history(selected=path)
            self.status.set(f"Đã lưu bản ghi: {path.name}")

    def choose_batch(self):
        paths = filedialog.askopenfilenames(parent=self.root, filetypes=[("Văn bản", "*.txt *.md *.pdf")])
        if paths:
            self.batch_files = list(paths)
            self.batch_file_label.set(f"Đã chọn {len(paths)} file; sẽ dùng các file này.")

    def clear_batch(self):
        self.batch_files = []
        self.batch_file_label.set("Chưa chọn file; sẽ dùng các dòng bên trên.")

    def batch(self):
        paths = list(self.batch_files)
        lines = [line.strip() for line in self.batch_text.get("1.0", "end").splitlines() if line.strip()]
        voice, sampling, fmt = self.batch_voice.get(), self._sampling(), self.format.get()
        self._job(lambda: self.tool.batch([read_document(path) for path in paths] if paths else lines,
                                         voice=voice, sampling=sampling, fmt=fmt,
                                         progress=lambda msg: self._post("status", msg)), self._result)

    def assign_actor(self):
        name, voice = self.actor.get().strip(), self.actor_voice.get()
        if not name or ":" in name or not voice:
            raise ValueError("Nhập tên nhân vật không chứa ':' và chọn giọng.")
        for item in self.actors.get_children():
            if self.actors.item(item, "values")[0] == name:
                self.actors.delete(item)
        self.actors.insert("", "end", values=(name, voice))

    def remove_actor(self):
        for item in self.actors.selection():
            self.actors.delete(item)

    def conversation(self):
        script = self.script.get("1.0", "end")
        mapping = dict(self.actors.item(item, "values") for item in self.actors.get_children())
        sampling, gap, fmt = self._sampling(), float(self.gap.get()), self.format.get()
        self._job(lambda: self.tool.conversation(script, mapping, sampling, gap, fmt,
                                                lambda msg: self._post("status", msg)), self._result)

    def srt(self):
        path, voice, sampling, fmt = self.srt_path.get(), self.srt_voice.get(), self._sampling(), self.format.get()
        if not path:
            raise ValueError("Hãy chọn file SRT.")
        min_speed = float(self.srt_min_speed.get())
        if not 0.5 <= min_speed <= 2:
            raise ValueError("Tốc độ ban đầu / tối thiểu SRT phải nằm trong 0.5–2.0x.")
        def progress(message):
            self._post("status", message)
            self._post("log", message)
        self._job(lambda: self.tool.srt(path, voice, sampling, True, fmt,
                                      progress, fit_to_timing=True, min_speed=min_speed), self._result)

    def training_job(self):
        import importlib.util
        values = {key: var.get().strip() for key, var in self.training.items()}
        base = self.vars["backbone_repo"].get().strip() or DEFAULT_MODEL
        subfolder = self.vars["model_subfolder"].get().strip()
        task = values["task"]
        modules = {
            "Chuẩn bị dữ liệu": ["pyarrow"],
            "Train LoRA": ["torch", "transformers", "peft", "accelerate", "pyarrow"],
            "Merge LoRA": ["torch", "transformers", "peft"],
            "Đóng gói giọng": [],
        }
        missing = [name for name in modules[task] if importlib.util.find_spec(name) is None]
        if missing:
            raise ValueError(f"Thiếu thư viện: {', '.join(missing)}. Cài uv sync --extra finetune rồi mở lại tool.")
        if task == "Chuẩn bị dữ liệu":
            script = "prepare_dataset.py"
            args = ["--dataset-dir", values["dataset"], "--base", base]
        elif task == "Train LoRA":
            if not Path(values["data"]).is_file():
                raise ValueError("Chọn file train.parquet hoặc JSONL trước.")
            if not values["run"] or Path(values["run"]).name != values["run"]:
                raise ValueError("Tên lượt train phải là một tên thư mục.")
            epochs, rank = float(values["epochs"]), int(values["rank"])
            if not 0 < epochs <= 100 or not 1 <= rank <= 256:
                raise ValueError("Epoch phải > 0 và <= 100; rank 1–256.")
            script = "train_lora.py"
            args = ["--data", values["data"], "--run", values["run"], "--output-dir", values["out"],
                    "--base", base, "--subfolder", subfolder, "--epochs", str(epochs), "--r", str(rank),
                    "--merge", "--num-workers", "0"]
        elif task == "Merge LoRA":
            script = "merge_lora.py"
            args = ["--adapter", values["adapter"], "--out", values["out"], "--base", base, "--subfolder", subfolder]
        else:
            script = "make_voice.py"
            args = ["--audio", values["audio"], "--name", values["run"], "--out", values["out"], "--base", base, "--default"]
        command = [sys.executable, "-u", str(ROOT / "finetune" / script), *args]
        def run():
            self.tool.unload()
            self._post("log", f"Đang chạy {task}. Model desktop đã được giải phóng.")
            self._run_process(command)
            return f"Hoàn tất: {task}. Tải model lại để tạo audio."
        self._job(run, lambda msg: (self._loaded(msg), self.book.select(self.pages["Kết quả / Log"])))

    def _run_process(self, command):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        environment = dict(os.environ, PYTHONIOENCODING="utf-8")
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                   creationflags=flags)
        self.job_process = process
        try:
            if self.tool.stop_event.is_set():
                process.terminate()
            for line in process.stdout:
                self._post("log", line)
            code = process.wait()
            if self.tool.stop_event.is_set():
                raise Cancelled("Đã dừng tiến trình.")
            if code:
                raise RuntimeError(f"Tiến trình kết thúc với mã {code}. Xem log để biết nguyên nhân.")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            process.stdout.close()
            self.job_process = None

    def start_api(self):
        if self.busy or self.recorder is not None:
            raise ValueError("Hãy hoàn tất tác vụ hoặc bản ghi micro trước khi chạy API.")
        if self.api_process is not None and self.api_process.poll() is None:
            raise ValueError("API đang chạy.")
        port, backend, key = int(self.api_port.get()), self.api_backend.get(), self.api_key.get()
        if not 1024 <= port <= 65535:
            raise ValueError("Cổng API phải nằm trong 1024–65535.")
        self.tool.unload()
        self.refresh_voices()
        env = dict(os.environ, HOST="127.0.0.1", PORT=str(port), VIENEU_DEVICE=backend,
                   VIENEU_BACKEND="pytorch" if backend == "cuda" else "onnx",
                   VIENEU_API_KEY=key, PYTHONIOENCODING="utf-8")
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")])
        process = subprocess.Popen([sys.executable, "-u", "-m", "apps.openai_speech"], cwd=ROOT,
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace",
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.api_process = process
        def logs():
            try:
                for line in process.stdout:
                    self._post("log", "API · " + line)
                self._post("log", f"API đã thoát, mã {process.wait()}.")
            finally:
                process.stdout.close()
        threading.Thread(target=logs, daemon=True).start()
        self.status.set(f"API đang khởi động tại 127.0.0.1:{port}; xem log để biết khi sẵn sàng.")

    def stop_api(self):
        if self.api_process is not None and self.api_process.poll() is None:
            self.api_process.terminate()
        self.status.set("Đã yêu cầu dừng API.")

    def stop(self):
        self.tool.stop()
        if self.player is not None:
            self.player.stop()
        if self.job_process is not None and self.job_process.poll() is None:
            self.job_process.terminate()
        self.status.set("Đã yêu cầu dừng. Tác vụ sinh thường dừng sau lượt suy luận hiện tại.")

    def _selected(self):
        selection = self.history.selection()
        if not selection:
            raise ValueError("Hãy chọn một kết quả.")
        if len(selection) != 1:
            raise ValueError("Hãy chọn một file cho thao tác này.")
        path = self.audio_files[selection[0]]
        if not path.is_file():
            raise ValueError("File không còn tồn tại. Hãy làm mới lịch sử.")
        return path

    def play_selected(self):
        path = self._selected()
        if path.suffix.lower() == ".mp4":
            self._open_file(path)
            self.status.set(f"Đã mở video: {path.name}")
            return
        if path.suffix not in (".wav", ".flac", ".mp3"):
            raise ValueError("Chọn file audio để nghe.")
        def run():
            player = WavePlayer()
            self.player = player
            try:
                player.play_file(path)
            finally:
                self.player = None
        self._job(run, lambda _: self.status.set("Đã phát xong."))

    def save_selected(self):
        import shutil
        path = self._selected()
        target = filedialog.asksaveasfilename(parent=self.root, initialfile=path.name, defaultextension=path.suffix,
                                            filetypes=[(path.suffix, "*" + path.suffix)])
        if target and Path(target).resolve() != path.resolve():
            shutil.copy2(path, target)
            self.status.set(f"Đã lưu: {target}")

    def save_subtitle_pair(self):
        import shutil
        selected = self._selected()
        subtitle = selected.with_suffix(".srt")
        candidates = [path for path in self.audio_files.values()
                      if path.stem == selected.stem and path.parent == selected.parent
                      and path.suffix in (".wav", ".flac", ".mp3")]
        if not subtitle.is_file() or not candidates:
            raise ValueError("Chọn kết quả được tạo bằng nút Tạo audio + SRT từng câu.")
        folder = filedialog.askdirectory(parent=self.root, title="Chọn thư mục lưu audio và SRT")
        if folder:
            for source in (candidates[0], subtitle):
                target = Path(folder) / source.name
                if target.resolve() != source.resolve():
                    shutil.copy2(source, target)
            self.status.set(f"Đã lưu cặp audio + SRT vào: {folder}")

    def open_output(self):
        self.tool.output_dir.mkdir(parents=True, exist_ok=True)
        self._open_folder(self.tool.output_dir)

    def open_selected_folder(self):
        self._open_folder(self._selected().parent)

    @staticmethod
    def _open_folder(path):
        DesktopApp._open_file(path)

    @staticmethod
    def _open_file(path):
        if os.name == "nt":
            os.startfile(path)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])

    def _terminate_children(self):
        for process in (self.job_process, self.api_process):
            if process is not None and process.poll() is None:
                process.terminate()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.root.after_cancel(self._drain_handle)
        self.stop()
        self._terminate_children()
        if self.recorder is not None:
            self.recorder.close()
        if not self.busy and self.tool.tts is not None:
            self.tool.unload()
        self.root.destroy()
        atexit.unregister(self._terminate_children)


def main():
    parser = argparse.ArgumentParser(description="VieNeu v3 Turbo — ứng dụng desktop")
    parser.add_argument("--output-dir", type=Path, default=None, help="Ưu tiên thư mục này thay cho nơi lưu đã nhớ")
    parser.add_argument("--check", action="store_true", help="Kiểm tra cấu trúc giao diện, không tải model")
    args = parser.parse_args()
    root = tk.Tk()
    if args.check:
        root.withdraw()
    tool = TurboTool(args.output_dir, settings_path=ROOT / "outputs" / "v3turbo_desktop_settings.json")
    if args.output_dir is not None and not args.check:
        tool.set_output_dir(args.output_dir)
    app = DesktopApp(root, tool)
    if args.check:
        root.update_idletasks()
        print(json.dumps({"tabs": len(app.book.tabs()), "sampling_fields": len(Sampling.__dataclass_fields__),
                          "voices": len(app.tool.voice_names())}))
        app.close()
    else:
        root.mainloop()


if __name__ == "__main__":
    main()
