"""Tkinter desktop UI for the v3 Turbo SDK."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
from dataclasses import replace
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from apps.desktop_queue import QueuePanel, QueueTask, run_queue
from apps.desktop_audio import PlaybackStopped, WavePlayer
from apps.desktop_theme import COLORS, apply_theme
from apps.desktop_events import DesktopEvents
from apps.v3turbo_tool import Cancelled, DEFAULT_MODEL, OUTPUT_KINDS, ROOT, Sampling, TurboTool, read_document


class DesktopApp:
    def __init__(self, root: tk.Tk, tool: TurboTool | None = None, *, autoload: bool = True):
        self.root = root
        self.tool = tool or TurboTool(settings_path=ROOT / "outputs" / "v3turbo_desktop_settings.json")
        self.events = DesktopEvents()
        self.busy = False
        self.closed = False
        self._autoload_handle = None
        self.player = None
        self.audio_files = {}
        self.action_buttons = []
        self.voice_boxes = []
        self._voice_model = object()
        self.vars = {}
        self.pages = {}
        self.nav_buttons = {}
        self.history_records = []
        self.queue_panels = {}
        self.inline_histories = {}
        self.queue_cancel = threading.Event()
        self._worker = None
        self._history_signature = None
        self._history_refresh_version = 0
        self._inline_signatures = {}
        root.title("VieNeu v3 Turbo — Studio giọng nói")
        root.geometry(f"{min(1280, root.winfo_screenwidth() - 60)}x{min(850, root.winfo_screenheight() - 100)}")
        root.minsize(980, 620)
        self.style = apply_theme(root)
        self.status = tk.StringVar(value="Đang chuẩn bị tải model…" if autoload else "Sẵn sàng.")
        shell = ttk.Frame(root, style="Shell.TFrame")
        shell.pack(fill="both", expand=True)
        self.sidebar = ttk.Frame(shell, width=196, style="Sidebar.TFrame", padding=(10, 22))
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        ttk.Label(self.sidebar, text="VieNeu", style="Sidebar.TLabel").pack(anchor="w", padx=12)
        ttk.Label(self.sidebar, text="VOICE & VIDEO STUDIO", style="SidebarMuted.TLabel").pack(anchor="w", padx=12, pady=(4, 12))
        wave = tk.Canvas(self.sidebar, height=30, background=COLORS["navy"], highlightthickness=0)
        wave.pack(fill="x", padx=12, pady=(0, 18))
        for index, height in enumerate((6, 12, 22, 14, 28, 18, 10, 24, 16, 8, 20, 12)):
            x = 6 + index * 12
            wave.create_line(x, 15 - height / 2, x, 15 + height / 2, width=5,
                             fill=(COLORS["purple"], COLORS["blue"], COLORS["teal"])[index % 3], capstyle="round")
        self.navigation = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        self.navigation.pack(fill="x")
        ttk.Label(self.sidebar, text="Ctrl + 1…5   Chuyển trang\nF5   Làm mới kết quả\nEsc   Dừng tác vụ", style="SidebarMuted.TLabel").pack(side="bottom", anchor="w", padx=12, pady=8)
        body = ttk.Frame(shell, style="Shell.TFrame", padding=(16, 12))
        body.pack(side="left", fill="both", expand=True)
        header = ttk.Frame(body, style="Hero.TFrame", padding=(18, 12))
        header.pack(fill="x")
        headings = ttk.Frame(header, style="Hero.TFrame")
        headings.pack(side="left", fill="x", expand=True)
        self.page_title = tk.StringVar(value="Văn bản")
        self.page_caption = tk.StringVar()
        ttk.Label(headings, textvariable=self.page_title, style="Title.TLabel").pack(anchor="w")
        caption = ttk.Label(headings, textvariable=self.page_caption, style="HeroMuted.TLabel", wraplength=450)
        caption.pack(anchor="w", pady=(4, 0))
        headings.bind("<Configure>", lambda event: caption.configure(wraplength=max(200, event.width)))
        controls = ttk.Frame(header, style="Hero.TFrame")
        controls.pack(side="right")
        self.model_state = tk.StringVar(value="Model chưa tải")
        self.model_badge = ttk.Label(controls, textvariable=self.model_state, style="Idle.Badge.TLabel")
        self.model_badge.pack(anchor="e", pady=(0, 7))
        self.model_state.trace_add("write", self._update_model_badge)
        self.header_stop = ttk.Button(controls, text="Dừng tác vụ / phát", style="Danger.TButton", command=self.stop)
        self.header_stop.pack(anchor="e")
        storage = self.storage_bar = ttk.Frame(body, padding=(16, 8))
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
        self._srt_tab()
        self._vietsub_tab()
        self._history_tab()
        footer = self.footer_bar = ttk.Frame(body, padding=(15, 10))
        footer.pack(side="bottom", fill="x", pady=(12, 0), before=self.book)
        status_label = ttk.Label(footer, textvariable=self.status, wraplength=900, style="Muted.TLabel")
        status_label.pack(anchor="w")
        footer.bind("<Configure>", lambda event: status_label.configure(wraplength=max(200, event.width - 30)))
        self.progress = ttk.Progressbar(footer, mode="indeterminate")
        self.progress.pack(fill="x", pady=(6, 0))
        self.book.select(self.pages["Văn bản"])
        self.refresh_history()
        if getattr(self.tool, "settings_error", None):
            self._log(self.tool.settings_error)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._drain_handle = root.after(80, self._drain)
        for index, title in enumerate(("Văn bản", "SRT", "Tự động vietsub video", "Kết quả / Log", "Cấu hình"), 1):
            root.bind(f"<Control-Key-{index}>", lambda event, page=title: self.book.select(self.pages[page]))
        root.bind("<F5>", lambda event: self._guard(self.refresh_history))
        root.bind("<Escape>", lambda event: self.stop() if self.busy else None)
        if autoload:
            self._autoload_handle = root.after_idle(self._autoload_model)

    def _tab(self, title, scroll=False):
        frame = ttk.Frame(self.book, style="Shell.TFrame")
        self.book.add(frame, text=title)
        self.pages[title] = frame
        labels = {"Văn bản": "01   Đọc văn bản", "SRT": "02   Đọc phụ đề SRT",
                  "Tự động vietsub video": "03   Tự động vietsub",
                  "Kết quả / Log": "04   Lịch sử & file", "Cấu hình": "05   Cấu hình"}
        nav_styles = {"SRT": "Blue.Nav.TButton", "Tự động vietsub video": "Teal.Nav.TButton",
                      "Kết quả / Log": "Orange.Nav.TButton", "Cấu hình": "Slate.Nav.TButton"}
        button = ttk.Button(self.navigation, text=labels[title], style=nav_styles.get(title, "Nav.TButton"), command=lambda: self.book.select(frame))
        self.nav_buttons[title] = button
        # The visual order follows the workflow, independent of widget creation.
        order = list(labels)
        for key in order:
            if key in self.nav_buttons:
                self.nav_buttons[key].pack_forget()
                self.nav_buttons[key].pack(fill="x", pady=2)
        if not scroll:
            content = ttk.Frame(frame, style="Shell.TFrame", padding=(0, 12, 0, 0))
            content.pack(fill="both", expand=True)
            return content
        canvas = tk.Canvas(frame, highlightthickness=0, background=COLORS["background"])
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        content = ttk.Frame(canvas, style="Shell.TFrame", padding=(0, 12, 12, 0))
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
        vietsub_selected = current == str(self.pages.get("Tự động vietsub video"))
        if vietsub_selected:
            self.header_stop.pack_forget()
        elif not self.header_stop.winfo_manager():
            self.header_stop.pack(anchor='e')
        if current in (str(self.pages.get("Văn bản")), str(self.pages.get("SRT"))) or vietsub_selected:
            self.storage_bar.pack_forget()
        elif not self.storage_bar.winfo_manager():
            self.storage_bar.pack(fill="x", pady=(10, 12), before=self.book)
        if hasattr(self, "footer_bar"):
            if vietsub_selected:
                self.footer_bar.pack_forget()
            elif not self.footer_bar.winfo_manager():
                self.footer_bar.pack(side="bottom", fill="x", pady=(12, 0), before=self.book)
        captions = {"Văn bản": "Đọc văn bản và chạy hàng đợi TXT.", "SRT": "Đọc hàng đợi SRT, giữ mốc và khoảng lặng.",
                    "Kết quả / Log": "Lịch sử, phát và quản lý file.",
                    "Cấu hình": "Model, thiết bị, giọng đọc và nơi lưu.",
                    "Tự động vietsub video": "Thêm video + SRT → chạy hàng đợi → mở MP4, SRT và WAV."}
        for title, frame in self.pages.items():
            selected = str(frame) == current
            self.nav_buttons[title].state(["selected" if selected else "!selected"])
            if selected:
                self.page_title.set("Lịch sử & file" if title == "Kết quả / Log" else title)
                self.page_caption.set(captions[title])

    def _update_model_badge(self, *_):
        value = self.model_state.get()
        name = "Loading" if value.startswith("Đang") else "Idle" if value == "Model chưa tải" else "Ready"
        self.model_badge.configure(style=f"{name}.Badge.TLabel")

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
        destination = ttk.LabelFrame(tab, text="01 · Nơi lưu kết quả", style="Teal.TLabelframe", padding=12)
        destination.pack(fill="x", pady=(0, 18))
        self.output_dir_var = self._path_field(destination, "Thư mục lưu", str(self.tool.output_dir), kind="directory")
        actions = ttk.Frame(destination)
        actions.pack(fill="x")
        self._button(actions, "Áp dụng nơi lưu", self.apply_output_dir, primary=True)
        ttk.Label(destination, text="Áp dụng cho file tạo tiếp theo. Lịch sử giữ cả các thư mục đã dùng.",
                  style="Muted.TLabel").pack(anchor="w", pady=(5, 0))
        columns = ttk.Frame(tab)
        columns.pack(fill="x")
        left = ttk.LabelFrame(columns, text="02 · Model và thiết bị", style="Purple.TLabelframe", padding=12)
        right = ttk.LabelFrame(columns, text="03 · Giọng đọc và đầu ra", style="Blue.TLabelframe", padding=12)
        left.pack(side="left", fill="both", expand=True, padx=(0, 22))
        right.pack(side="left", fill="both", expand=True)
        orientation = [None]
        def arrange(event):
            horizontal = event.width >= 850
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
            ("dtype", "Kiểu số GPU", "float32", ["auto", "float32", "bfloat16"]),
            ("precision", "Độ chính xác ONNX", "fp32", ["fp32", "int8"]),
            ("model_subfolder", "Thư mục trọng số", "update", None),
            ("threads", "Số luồng CPU (0 = tự động)", 0, None),
            ("max_streams", "Số luồng streaming CUDA", 16, None),
            ("babble_retries", "Lượt thử lại câu ngắn", 2, None),
        ]
        for key, label, default, choices in fields[:5]:
            self.vars[key], _, _ = self._field(left, label, default, choices)
        row = ttk.Frame(left)
        row.pack(fill="x")
        self._button(row, "Tải model", self.load_model, primary=True)
        self._button(row, "Giải phóng model", lambda: self._job(self.tool.unload, lambda _: self._unloaded()))
        ttk.Label(left, text="Tự tải khi mở tool. Auto chọn GPU khi có CUDA.\n"
                  "FP32 giữ độ chính xác hiện tại; tải lại khi đổi model.",
                  style="Muted.TLabel", wraplength=400).pack(anchor="w", pady=8)
        advanced_model = ttk.Frame(left)
        self.show_model_options = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Thiết lập model nâng cao", variable=self.show_model_options,
                        command=lambda: self._toggle_options(advanced_model, self.show_model_options)).pack(anchor="w", pady=4)
        for key, label, default, choices in fields[5:]:
            self.vars[key], _, _ = self._field(advanced_model, label, default, choices)
        self.vars["onnx_dir"] = self._path_field(advanced_model, "Thư mục ONNX (tùy chọn)", kind="directory")
        self.vars["onnx_repo"], _, _ = self._field(advanced_model, "Repo ONNX (tùy chọn)")
        self.vars["moss_tokenizer"], _, _ = self._field(advanced_model, "Codec MOSS", "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano")
        labels = {"speed": "Tốc độ đọc (0.5–2.0x)", "temperature": "Temperature", "top_k": "Top K", "top_p": "Top P",
                  "repetition_penalty": "Phạt lặp", "repetition_window": "Cửa sổ phạt lặp",
                  "max_new_frames": "Giới hạn frame / đoạn", "max_chars": "Ký tự / đoạn",
                  "batch_size": "Số mục / batch"}
        defaults = Sampling()
        self.format, _, _ = self._field(right, "Định dạng xuất", "wav", ["wav", "flac", "mp3"])
        self.vars["speed"], _, _ = self._field(right, labels["speed"], defaults.speed)
        ttk.Label(right, text="0.8× chậm · 1.0× gốc · 1.2× nhanh\n"
                  "Đổi tốc độ bằng Rubber Band, giữ cao độ.",
                  style="Muted.TLabel", wraplength=400).pack(anchor="w", pady=8)
        advanced_sampling = ttk.Frame(right)
        self.show_sampling_options = tk.BooleanVar(value=False)
        ttk.Checkbutton(right, text="Tham số tạo giọng nâng cao", variable=self.show_sampling_options,
                        command=lambda: self._toggle_options(advanced_sampling, self.show_sampling_options)).pack(anchor="w", pady=4)
        self.model_options_panel = advanced_model
        self.sampling_options_panel = advanced_sampling
        for key, label in labels.items():
            if key != "speed":
                self.vars[key], _, _ = self._field(advanced_sampling, label, getattr(defaults, key))
        ttk.Label(right, text="Văn bản dùng tốc độ chung; SRT có tốc độ ban đầu riêng.\n"
                  "Vietsub luôn xuất giọng WAV riêng. MP3 cần bộ mã hóa trên máy.",
                  style="Muted.TLabel", wraplength=400).pack(anchor="w", pady=10)

    @staticmethod
    def _toggle_options(panel, variable):
        if variable.get():
            panel.pack(fill="x", pady=(4, 0))
        else:
            panel.pack_forget()

    def _speech_tab(self):
        tab = self._tab("Văn bản")
        panes = ttk.Panedwindow(tab, orient="horizontal")
        panes.pack(fill="both", expand=True)
        editor = ttk.LabelFrame(panes, text="01 · Nội dung & giọng đọc", style="Purple.TLabelframe", padding=12)
        results = ttk.Frame(panes, style="Shell.TFrame", padding=(12, 0, 0, 0))
        panes.add(editor, weight=2)
        panes.add(results, weight=3)
        row = ttk.Frame(editor)
        row.pack(fill="x", pady=(0, 8))
        ttk.Label(row, text="Giọng", width=7).pack(side="left")
        self.voice = tk.StringVar()
        box = ttk.Combobox(row, textvariable=self.voice, state="readonly", width=16)
        box.pack(side="left", fill="x", expand=True)
        self.voice_boxes.append((self.voice, box))
        row = ttk.Frame(editor)
        row.pack(fill="x")
        ttk.Label(row, text="Tốc độ", width=7).pack(side="left")
        self.speech_speed = self.vars["speed"]
        ttk.Spinbox(row, textvariable=self.speech_speed, from_=0.5, to=2.0,
                    increment=0.05, width=8).pack(side="left")
        ttk.Label(row, text="x", style="Muted.TLabel").pack(side="left", padx=5)
        actions = ttk.Frame(editor)
        actions.pack(side="bottom", fill="x", pady=(8, 0))
        self._button(actions, "Thêm văn bản", self.add_text_queue, primary=True).configure(style="CompactPrimary.TButton")
        self._button(actions, "Thêm TXT", lambda: self.choose_queue_files("Văn bản")).configure(style="Compact.TButton")
        self.text_count = tk.StringVar(value="0 ký tự · Ctrl + Enter để thêm")
        ttk.Label(editor, textvariable=self.text_count, style="Muted.TLabel", wraplength=260).pack(side="bottom", anchor="w", pady=6)
        self.speech = self._text(editor, height=7)
        self.speech.configure(width=24)
        self.speech.bind("<<Modified>>", self._text_changed)
        self.speech.bind("<Control-Return>", self._add_text_shortcut)
        self.speech.edit_modified(False)
        self._queue_section(results, "Văn bản")
        last_width = [0]
        def arrange(event):
            if event.width == last_width[0] or event.width < 600:
                return
            last_width[0] = event.width
            panes.sashpos(0, max(260, min(int(event.width * .4), event.width - 380)))
        panes.bind("<Configure>", arrange)

    def _text_changed(self, event=None):
        if self.speech.edit_modified():
            count = self.speech.count("1.0", "end-1c", "chars")
            self.text_count.set(f"{count[0] if count else 0:,} ký tự · Ctrl + Enter để thêm")
            self.speech.edit_modified(False)

    def _add_text_shortcut(self, event=None):
        self._guard(self.add_text_queue)
        return "break"

    def _srt_tab(self):
        tab = self._tab("SRT")
        settings = ttk.LabelFrame(tab, text="01 · Phụ đề & giọng đọc", style="Blue.TLabelframe", padding=(12, 8))
        settings.pack(fill="x", pady=(0, 12))
        row = ttk.Frame(settings)
        row.pack(fill="x")
        ttk.Label(row, text="Giọng đọc").pack(side="left", padx=(0, 8))
        self.srt_voice = tk.StringVar()
        voice_box = ttk.Combobox(row, textvariable=self.srt_voice, state="readonly", width=18)
        voice_box.pack(side="left", fill="x", expand=True, padx=(0, 16))
        self.voice_boxes.append((self.srt_voice, voice_box))
        ttk.Label(row, text="Tốc độ").pack(side="left", padx=(0, 8))
        self.srt_min_speed = tk.StringVar(value="1.0")
        ttk.Spinbox(row, textvariable=self.srt_min_speed, from_=0.5, to=2.0, increment=.05, width=5).pack(side="left")
        ttk.Label(row, text="×", style="Muted.TLabel").pack(side="left", padx=(5, 16))
        self._button(row, "Thêm nhiều file SRT", lambda: self.choose_queue_files("SRT"), primary=True).configure(style="CompactPrimary.TButton")
        ttk.Label(settings, text="Giữ mốc và khoảng lặng · Tăng tốc khi câu vượt khung · Xuất audio + SRT xuống dòng 70 ký tự",
                  style="Muted.TLabel", wraplength=850).pack(anchor="w", pady=(7, 2))
        self._queue_section(tab, "SRT")

    def _vietsub_tab(self):
        from apps.vietsub_panel import VietsubPanel
        tab = self._tab("Tự động vietsub video")
        tab.configure(padding=(12, 6))
        self.vietsub_running = False
        self.vietsub_panel = VietsubPanel(tab, start=self.start_vietsub, stop=self.stop,
                                          guard=self._guard, open_file=self._open_file,
                                          make_task=self._vietsub_task)
        self.vietsub_panel.pack(fill="both", expand=True)

    def _update_vietsub_controls(self):
        self.vietsub_panel.set_busy(self.busy, self.vietsub_running)

    def _vietsub_task(self, video, subtitles):
        from apps.vietsub_queue import VietsubTask
        if not str(video).strip() or not str(subtitles).strip():
            raise ValueError("Hãy chọn video và SRT gốc.")
        video, subtitles = (Path(path).expanduser().resolve() for path in (video, subtitles))
        if not video.is_file() or not subtitles.is_file() or subtitles.suffix.lower() != '.srt':
            raise ValueError("Hãy chọn video và file SRT còn tồn tại trên máy.")
        panel = self.vietsub_panel
        sampling = replace(self._sampling(), speed=float(panel.speed.get()))
        sampling.kwargs()
        return VietsubTask(video, subtitles, panel.voice.get() or None, sampling)

    def start_vietsub(self):
        if self.busy:
            raise RuntimeError("Hãy chờ hoặc dừng tác vụ hiện tại.")
        panel = self.vietsub_panel
        tasks = panel.queue.pending()
        if not tasks:
            if panel.queue.tasks:
                raise ValueError("Hàng đợi đã xong. Bấm Thêm vào hàng đợi để thêm video/SRT mới.")
            panel.queue.add(self._vietsub_task(panel.source_video.get(), panel.source_srt.get()))
            tasks = panel.queue.pending()
        settings = self._model_kwargs() if self.tool.tts is None else None
        self.queue_cancel.clear()
        indexed = {task.identifier: index for index, task in enumerate(tasks, 1)}
        by_id = {task.identifier: task for task in tasks}
        self.vietsub_running = True
        def notify(identifier, state, value):
            self._post('vietsub_queue', (indexed[identifier], len(tasks), by_id[identifier], state, value))
        def done(counts):
            self.vietsub_running = False
            panel.finish_queue(*counts, self.queue_cancel.is_set())
            self.status.set(panel.stage.get())
            self.refresh_history()
            self._update_vietsub_controls()
        self._job(lambda: self.tool.create_vietsub_queue(tasks, model_settings=settings,
                        cancelled=self.queue_cancel.is_set, notify=notify), done)

    def _history_tab(self):
        tab = self._tab("Kết quả / Log")
        filters = ttk.Frame(tab)
        filters.pack(fill="x", pady=(0, 10))
        ttk.Label(filters, text="Tìm file", style="Muted.TLabel").pack(side="left", padx=(0, 8))
        self.history_query = tk.StringVar()
        ttk.Entry(filters, textvariable=self.history_query, width=16).pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.history_kind = tk.StringVar(value="Tất cả")
        ttk.Combobox(filters, textvariable=self.history_kind, width=20, state="readonly",
                     values=["Tất cả", *dict.fromkeys([*OUTPUT_KINDS.values(), "Lời thoại"])]).pack(side="left", padx=(0, 10))
        ttk.Button(filters, text="Làm mới", command=lambda: self._guard(self.refresh_history)).pack(side="left")
        self.history_count = tk.StringVar(value="0 file")
        ttk.Label(filters, textvariable=self.history_count, style="Accent.TLabel", width=14).pack(side="left", padx=(8, 0))
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
        self.history.tag_configure("alternate", background="#f5f7ff")
        self.history_detail = tk.StringVar(value="Chọn file để xem đường dẫn. Ctrl / Shift để chọn nhiều file.")
        detail = ttk.Label(results, textvariable=self.history_detail, wraplength=700, style="Muted.TLabel")
        detail.pack(side="bottom", anchor="w", pady=8, before=table)
        results.bind("<Configure>", lambda event: detail.configure(wraplength=max(200, event.width - 24)))
        actions = ttk.Frame(results)
        actions.pack(side="bottom", fill="x", before=detail)
        self._button(actions, "Mở / nghe", self.play_selected, primary=True).configure(style="CompactPrimary.TButton")
        self._button(actions, "Lưu thành…", self.save_selected).configure(style="Compact.TButton")
        self._button(actions, "Lưu cặp audio + SRT…", self.save_subtitle_pair).configure(style="Compact.TButton")
        ttk.Button(actions, text="Mở thư mục file", style="Compact.TButton", command=lambda: self._guard(self.open_selected_folder)).pack(side="left", padx=5)
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
        self.log.configure(state="disabled", background="#17233f", foreground="#dce5ff",
                           selectbackground="#475b85", font=("Consolas", 10))
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
        self._history_refresh_version += 1
        signature = tuple((record.path, record.modified, record.size, record.kind) for record in self.history_records)
        if signature != self._history_signature or selected is not None:
            self._filter_history(keep)
            self._history_signature = signature
        self._refresh_inline_history()

    def _filter_history(self, selected=None):
        if selected is None:
            selected = [self.audio_files[item] for item in self.history.selection() if item in self.audio_files]
        self.history.delete(*self.history.get_children())
        self.audio_files.clear()
        query, kind = self.history_query.get().strip().casefold(), self.history_kind.get()
        chosen = []
        selected = set(selected)
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
        if self.busy:
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
        if not self.closed:
            self.events.put((kind, value))

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > 5000:
            self.log.delete("1.0", f"{lines - 5000 + 1}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _drain(self):
        if self.closed:
            return
        queue_history_dirty = False
        history_version = self._history_refresh_version
        deadline = time.perf_counter() + .012
        for _ in range(100):
            if time.perf_counter() >= deadline:
                break
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status.set(value)
            elif kind == "vietsub_progress":
                self.vietsub_panel.set_stage(value)
                self.status.set(self.vietsub_panel.stage.get())
            elif kind == 'vietsub_queue':
                index, total, task, state, payload = value
                panel = self.vietsub_panel
                panel.queue.update_task(task.identifier, state, payload)
                if state == 'running':
                    panel.show_task(task)
                elif state == 'progress':
                    panel.set_stage(payload)
                    panel.detail.set(f'Video {index}/{total} · {task.video.name}'
                                     + (f' · {payload.detail}' if payload.detail else ''))
                    self.status.set(panel.stage.get())
                elif state == 'done':
                    panel.finish(payload)
                    self._log(f'{payload.video}\n{payload.subtitles}\n{payload.audio}\n{payload.elapsed:.2f}s')
                    queue_history_dirty = True
                else:
                    panel.fail(str(payload), state == 'cancelled')
                    self._log(f'{task.video.name}: {payload}')
            elif kind == "log":
                self._log(value)
            elif kind == "queue":
                page, identifier, state, payload = value
                self.queue_panels[page].update_task(identifier, state, payload)
                if state == "running":
                    self.status.set(f"{page} · Đang tạo giọng…")
                elif state == "progress":
                    self.status.set(f"{page} · {payload}")
                if state in ("done", "error", "cancelled"):
                    queue_history_dirty = True
                if state in ("error", "cancelled"):
                    self._log(str(payload))
                elif state == "done" and isinstance(payload, tuple) and len(payload) > 1:
                    self._log(f"{payload[0]}: {payload[1]}")
                elif state == "done" and getattr(payload, "timings", None):
                    self._log_speech_timings(payload)
            elif kind == "done":
                self.busy = False
                self.progress.stop()
                for button in self.action_buttons:
                    button.configure(state="normal")
                self._update_vietsub_controls()
                result, callback, error = value
                self.refresh_voices()
                if error:
                    if self.vietsub_running:
                        self.vietsub_running = False
                        self.vietsub_panel.fail(error, isinstance(error, Cancelled))
                        self._update_vietsub_controls()
                    self.refresh_history()
                    self.status.set(str(error))
                    self._log(str(error))
                    if not isinstance(error, (Cancelled, PlaybackStopped)):
                        messagebox.showerror("VieNeu v3 Turbo", str(error), parent=self.root)
                else:
                    self._guard(lambda: callback(result))
        if queue_history_dirty and self._history_refresh_version == history_version:
            self.refresh_history()
        self._drain_handle = self.root.after(10 if not self.events.empty() else 80, self._drain)

    def _log_speech_timings(self, result):
        timing = result.timings
        if not timing:
            return
        backend = getattr(self.tool.tts, "backend", "unknown")
        note = (f"{result.audio.name}: backend={backend}; total={timing['total']:.2f}s; "
                f"model={timing['model']:.2f}s; tempo={timing.get('tempo', 0):.2f}s; "
                f"export={timing.get('export', 0):.2f}s; "
                f"other={timing['other']:.2f}s")
        if result.duration > 0:
            note += f"; RTF={timing['total'] / result.duration:.3f}"
        self._log(note)

    def _job(self, function, callback=lambda result: None):
        if self.busy:
            raise RuntimeError("Đang có tác vụ chạy. Hãy chờ hoặc bấm Dừng.")
        self.busy = True
        self._update_vietsub_controls()
        self.tool.stop_event.clear()
        if not self.vietsub_running:
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
            finally:
                if self.closed and self.tool.tts is not None:
                    self.tool.unload()
        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()

    def _sampling(self):
        values = {}
        defaults = Sampling()
        for name in Sampling.__dataclass_fields__:
            var = self.vars.get(name)
            default = getattr(defaults, name)
            values[name] = default if var is None else bool(var.get()) if isinstance(default, bool) else type(default)(var.get())
        sampling = Sampling(**values)
        sampling.kwargs()
        return sampling

    def _model_kwargs(self):
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
        return kw

    def _autoload_model(self):
        self._autoload_handle = None
        if self.closed:
            return
        if self.busy:
            self._autoload_handle = self.root.after(80, self._autoload_model)
            return
        if self.tool.tts is not None:
            self.refresh_voices()
            self.status.set("Model đã sẵn sàng.")
            return
        self._guard(self.load_model)

    def load_model(self):
        kw = self._model_kwargs()
        self._job(lambda: self.tool.load(**kw), self._loaded)
        self._voice_model = object()
        self.model_state.set("Đang tải model…")
        self.status.set("Đang tải model đọc giọng nói… Lần đầu có thể cần tải dữ liệu về máy.")


    def _loaded(self, message):
        self.status.set(message)
        self._log(message)
        self.refresh_voices()

    def _unloaded(self):
        self.status.set("Đã giải phóng model.")
        self.refresh_voices()

    def refresh_voices(self):
        if self._voice_model is self.tool.tts:
            return
        names = self.tool.voice_names()
        default = getattr(self.tool.tts, "_default_voice", "") if self.tool.tts else ""
        if self.tool.tts:
            self.vietsub_panel.set_voices(names, default)
        for variable, box in self.voice_boxes:
            box.configure(values=names)
            if variable.get() not in names:
                variable.set(default if default in names else (names[0] if names else ""))
        self._voice_model = self.tool.tts
        if self.tool.tts:
            engine = getattr(self.tool.tts, "engine", None)
            device = getattr(getattr(engine, "device", None), "type", "")
            dtype = str(getattr(engine, "dtype", "")).removeprefix("torch.")
            details = " / " + device.upper() if device else ""
            details += " · " + dtype if device == "cuda" and dtype else ""
            self.model_state.set(f"{self.tool.tts.backend.upper()}{details} · {len(names)} giọng")
        else:
            self.model_state.set("Model chưa tải")

    def _queue_section(self, parent, page):
        tabs = ttk.Notebook(parent)
        tabs.pack(fill="both", expand=True)
        queue_page = ttk.Frame(tabs, padding=10)
        tabs.add(queue_page, text="Hàng đợi")
        actions = ttk.Frame(queue_page)
        actions.pack(fill="x", pady=(0, 7))
        self._button(actions, "Chạy hàng đợi", lambda: self.start_queue(page), primary=True).configure(style="CompactPrimary.TButton")
        self._button(actions, "Bỏ mục chọn", lambda: self.remove_queue_items(page)).configure(style="Compact.TButton")
        ttk.Button(actions, text="Dừng", style="Danger.TButton", command=self.stop).pack(side="left", padx=3)
        ttk.Label(queue_page, text="Audio + SRT từng câu" if page == "Văn bản" else "Audio khớp mốc + SRT mới, mỗi dòng tối đa 70 ký tự",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 8))
        panel = QueuePanel(queue_page)
        panel.pack(fill="both", expand=True)
        self.queue_panels[page] = panel
        history_page = ttk.Frame(tabs, padding=10)
        tabs.add(history_page, text="Lịch sử tạo")
        row = ttk.Frame(history_page)
        row.pack(fill="x", pady=(0, 6))
        self._button(row, "Phát", lambda: self.play_inline_history(page)).configure(style="CompactPrimary.TButton")
        ttk.Button(row, text="Dừng", style="Compact.TButton", command=self.stop).pack(side="left", padx=3)
        if page == 'SRT':
            self._button(row, 'Mở SRT', lambda: self.open_inline_subtitle(page)).configure(style='Compact.TButton')
            self._button(row, 'Lưu audio + SRT…', lambda: self._save_subtitle_pair(self._inline_selected(page))).configure(style='Compact.TButton')
        self._button(row, "Xóa", lambda: self.delete_inline_history(page)).configure(style="Compact.TButton")
        secondary = ttk.Frame(history_page)
        secondary.pack(fill="x", pady=(0, 6))
        self._button(secondary, "Làm mới", self.refresh_history).configure(style="Compact.TButton")
        self._button(secondary, "Thư mục", lambda: self.open_inline_history(page, folder=True)).configure(style="Compact.TButton")
        table = ttk.Treeview(history_page, columns=("name", "time"), show="headings", height=7, selectmode="browse")
        table.heading("name", text="Audio đã tạo")
        table.heading("time", text="Thời gian")
        table.column("name", width=220, minwidth=100)
        table.column("time", width=125, minwidth=110, stretch=False)
        scroll = ttk.Scrollbar(history_page, command=table.yview)
        table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        table.pack(fill="both", expand=True)
        table.bind("<Double-1>", lambda event: self._guard(lambda: self.play_inline_history(page)))
        self.inline_histories[page] = (table, {})

    def _queue_idle(self):
        if self.busy:
            raise RuntimeError("Hãy dừng hoặc chờ tác vụ xong trước khi sửa hàng đợi.")

    def choose_queue_files(self, page):
        self._queue_idle()
        extension = "*.srt" if page == "SRT" else "*.txt"
        paths = filedialog.askopenfilenames(parent=self.root, filetypes=[("File đầu vào", extension)])
        for path in paths:
            self.queue_panels[page].add(QueueTask(Path(path).name, source=Path(path)))
        if paths:
            self.status.set(f"Đã thêm {len(paths)} file vào hàng đợi {page}.")

    def add_text_queue(self):
        self._queue_idle()
        text = self.speech.get("1.0", "end").strip()
        if not text:
            raise ValueError("Hãy nhập văn bản.")
        texts = [text]
        for part in texts:
            if part:
                self.queue_panels["Văn bản"].add(QueueTask(part[:70].replace("\n", " "), text=part))
        self.status.set(f"Hàng đợi văn bản: {len(self.queue_panels['Văn bản'].tasks)} mục.")

    def remove_queue_items(self, page):
        self._queue_idle()
        self.queue_panels[page].remove_selected()

    def start_queue(self, page):
        self._queue_idle()
        if self.tool.tts is None:
            raise ValueError("Hãy tải model ở tab Cấu hình trước.")
        tasks = self.queue_panels[page].pending()
        if not tasks:
            raise ValueError("Không còn mục chờ. Thêm văn bản hoặc file vào hàng đợi.")
        sampling, fmt = self._sampling(), self.format.get()
        voice = self.srt_voice.get() if page == "SRT" else self.voice.get()
        min_speed = float(self.srt_min_speed.get()) if page == "SRT" else None
        if min_speed is not None and (not math.isfinite(min_speed) or not 0.5 <= min_speed <= 2):
            raise ValueError("Tốc độ SRT phải từ 0.5 đến 2.0x.")
        self.queue_cancel.clear()
        def process(task, progress):
            if page == "SRT":
                return self.tool.srt(task.source, voice, sampling, True, fmt,
                                     progress, fit_to_timing=True, min_speed=min_speed)
            text = read_document(task.source) if task.source else task.text
            return self.tool.synthesize_with_subtitles(text, voice, None, sampling, fmt, progress)
        def notify(identifier, state, value):
            self._post("queue", (page, identifier, state, value))
        def done(counts):
            self.refresh_history()
            completed, failed = counts
            self.status.set(f"Hàng đợi {page}: {completed} xong, {failed} lỗi."
                            + (" Đã dừng; các mục chưa chạy vẫn chờ." if self.queue_cancel.is_set() else ""))
        def run():
            return run_queue(tasks, process, self.queue_cancel.is_set, notify)
        self._job(run, done)

    def _refresh_inline_history(self):
        for page, (table, paths) in self.inline_histories.items():
            records = [record for record in self.history_records
                       if record.path.suffix.lower() in (".wav", ".flac", ".mp3")
                       and record.path.name.startswith("srt_" if page == "SRT" else ("speech_", "stream_"))]
            signature = tuple((record.path, record.modified) for record in records)
            if self._inline_signatures.get(page) == signature:
                continue
            self._inline_signatures[page] = signature
            selected = {paths[item] for item in table.selection() if item in paths}
            position = table.yview()
            table.delete(*table.get_children())
            paths.clear()
            chosen = []
            for record in records:
                path = record.path
                identifier = table.insert("", "end", values=(path.name,
                        datetime.fromtimestamp(record.modified).strftime("%d/%m/%Y %H:%M")))
                paths[identifier] = path
                if path in selected:
                    chosen.append(identifier)
            if chosen:
                table.selection_set(chosen)
            if position:
                table.yview_moveto(position[0])

    def _inline_selected(self, page):
        table, paths = self.inline_histories[page]
        selected = table.selection()
        if len(selected) != 1:
            raise ValueError("Chọn một file trong lịch sử.")
        path = paths[selected[0]]
        if not path.is_file():
            raise ValueError("File không còn tồn tại. Hãy làm mới lịch sử.")
        return path

    def open_inline_history(self, page, folder=False):
        path = self._inline_selected(page)
        self._open_file(path.parent if folder else path)

    def open_inline_subtitle(self, page):
        subtitle = self._inline_selected(page).with_suffix('.srt')
        if not subtitle.is_file():
            raise ValueError('Kết quả này chưa có SRT đi kèm. Hãy tạo lại audio từ SRT.')
        self._open_file(subtitle)

    def play_inline_history(self, page):
        self._play_audio_path(self._inline_selected(page))

    def delete_inline_history(self, page):
        self._ensure_storage_idle()
        path = self._inline_selected(page)
        pair = path.with_suffix(".srt")
        description = f"Xóa audio {path.name}?"
        if pair.is_file():
            description += "\nFile SRT đi kèm cũng sẽ bị xóa."
        if messagebox.askyesno("Xóa kết quả", description, parent=self.root):
            self.tool.delete_outputs([path], include_subtitles=True)
            self.refresh_history()
            self.status.set(f"Đã xóa: {path.name}")

    def _play_audio_path(self, path):
        def run():
            player = WavePlayer()
            self.player = player
            try:
                if Path(path).name.startswith("srt_"):
                    def started(seconds):
                        minutes, secs = divmod(seconds, 60)
                        self._post("status", f"Phát SRT từ {int(minutes):02}:{secs:04.1f}; giữ nguyên timeline file")
                    player.play_file(path, skip_initial_silence=True, on_start=started)
                else:
                    player.play_file(path)
            finally:
                self.player = None
        self._job(run, lambda _: self.status.set(f"Đã phát xong: {path.name}"))



    def _result(self, result, navigate=True):
        path, note = result if isinstance(result, tuple) else (result, "")
        path = Path(path)
        self.history_query.set("")
        self.history_kind.set("Tất cả")
        self.refresh_history(selected=path)
        item = next((item for item, source in self.audio_files.items() if source == path), None)
        self.status.set(f"Đã lưu: {path.name}. {note}")
        self._log(f"{path}\n{note}")
        if navigate:
            self.book.select(self.pages["Kết quả / Log"])
        return item





    def srt(self):
        self.start_queue("SRT")


    def stop(self):
        self.queue_cancel.set()
        self.tool.stop()
        if self.player is not None:
            self.player.stop()
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
        self._play_audio_path(path)

    def save_selected(self):
        import shutil
        path = self._selected()
        target = filedialog.asksaveasfilename(parent=self.root, initialfile=path.name, defaultextension=path.suffix,
                                            filetypes=[(path.suffix, "*" + path.suffix)])
        if target and Path(target).resolve() != path.resolve():
            shutil.copy2(path, target)
            self.status.set(f"Đã lưu: {target}")

    def save_subtitle_pair(self):
        self._save_subtitle_pair(self._selected())

    def _save_subtitle_pair(self, selected):
        import shutil
        subtitle = selected.with_suffix(".srt")
        candidates = [record.path for record in self.history_records
                      for path in (record.path,)
                      if path.stem == selected.stem and path.parent == selected.parent
                      and path.suffix in (".wav", ".flac", ".mp3")]
        if not subtitle.is_file() or not candidates:
            raise ValueError("Chọn một kết quả audio có SRT đi kèm.")
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


    def close(self):
        if self.closed:
            return
        self.closed = True
        for panel in self.queue_panels.values():
            panel.stop_bars()
        if self._autoload_handle is not None:
            self.root.after_cancel(self._autoload_handle)
            self._autoload_handle = None
        self.vietsub_panel.close()
        self.root.after_cancel(self._drain_handle)
        self.stop()
        worker_finished = self._worker is None or not self._worker.is_alive()
        if self.tool.tts is not None and (not self.busy or worker_finished):
            self.tool.unload()
        self.root.destroy()


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
    app = DesktopApp(root, tool, autoload=not args.check)
    if args.check:
        root.update_idletasks()
        print(json.dumps({"tabs": len(app.book.tabs()), "sampling_fields": len(Sampling.__dataclass_fields__),
                          "voices": len(app.tool.voice_names())}))
        app.close()
    else:
        root.mainloop()


if __name__ == "__main__":
    main()
