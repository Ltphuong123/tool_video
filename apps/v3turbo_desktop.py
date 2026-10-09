"""Tkinter desktop UI for the v3 Turbo SDK."""
from __future__ import annotations

import argparse
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

from apps.desktop_queue import QueuePanel, QueueTask, run_queue
from apps.desktop_audio import PlaybackStopped, WavePlayer
from apps.video_editor import SpeedSegment, build_multi_speed_time_map
from apps.video_markers import build_marker_time_map, read_marker_file
from apps.video_preview import VideoPreview
from apps.video_timeline import VideoTimeline, time_label
from apps.v3turbo_tool import Cancelled, DEFAULT_MODEL, OUTPUT_KINDS, ROOT, Sampling, TurboTool, read_document


class DesktopApp:
    def __init__(self, root: tk.Tk, tool: TurboTool | None = None):
        self.root = root
        self.tool = tool or TurboTool(settings_path=ROOT / "outputs" / "v3turbo_desktop_settings.json")
        self.events = queue.Queue()
        self.busy = False
        self.closed = False
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
        style.configure("Compact.TButton", font=("Segoe UI", 9), padding=(7, 6))
        style.configure("CompactPrimary.TButton", font=("Segoe UI", 9, "bold"), padding=(9, 6),
                        background="#5b48ef", foreground="#ffffff")
        style.map("CompactPrimary.TButton", background=[("disabled", "#d8d2ff"), ("active", "#4836d2")])
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
        self._video_tab()
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

    def _tab(self, title, scroll=False):
        frame = ttk.Frame(self.book)
        self.book.add(frame, text=title)
        self.pages[title] = frame
        labels = {"Văn bản": "01   Đọc văn bản", "SRT": "02   Đọc phụ đề SRT", "Video": "03   Chỉnh tốc độ video",
                  "Kết quả / Log": "04   Lịch sử & file", "Cấu hình": "05   Cấu hình"}
        button = ttk.Button(self.navigation, text=labels[title], style="Nav.TButton", command=lambda: self.book.select(frame))
        self.nav_buttons[title] = button
        # The visual order follows the workflow, independent of widget creation.
        order = list(labels)
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
        if current in (str(self.pages.get("Văn bản")), str(self.pages.get("SRT"))):
            self.storage_bar.pack_forget()
        elif not self.storage_bar.winfo_manager():
            self.storage_bar.pack(fill="x", pady=(10, 12), before=self.book)
        if hasattr(self, "video_preview") and current != str(self.pages["Video"]):
            self.video_preview.pause()
        captions = {"Văn bản": "Đọc văn bản và chạy hàng đợi TXT.", "SRT": "Đọc hàng đợi SRT, giữ mốc và khoảng lặng.",
                    "Video": "Xem và chỉnh tốc độ các đoạn video.", "Kết quả / Log": "Lịch sử, phát và quản lý file.",
                    "Cấu hình": "Model, thiết bị, giọng đọc và nơi lưu."}
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
        for key, label, value in [("bright", "EQ: giọng sáng nhẹ", True),
                                  ("compress", "Compressor: âm lượng đều hơn", True),
                                  ("peak_guard", "Giảm đỉnh vượt -1 dBFS", True)]:
            self.vars[key] = tk.BooleanVar(value=value)
            ttk.Checkbutton(right, text=label, variable=self.vars[key]).pack(anchor="w", pady=6)
        self.format, _, _ = self._field(right, "Định dạng xuất", "wav", ["wav", "flac", "mp3"])
        ttk.Label(right, text="Cấu hình áp dụng cho văn bản, hàng đợi, audio + SRT và streaming.\n"
                  "Tab SRT có tốc độ ban đầu riêng.\n"
                  "Streaming luôn lưu WAV. MP3 phụ thuộc libsndfile của máy.\n"
                  "Tốc độ: 0.8x chậm, 1.0x gốc, 1.2x nhanh; giữ cao độ.\n"
                  "Ưu tiên 0.85–1.15x để hạn chế biến đổi chất giọng.\n"
                  "Streaming khác 1.0x chờ tạo xong rồi chỉnh tốc độ và phát.\n"
                  "Phong cách đọc theo giọng tham chiếu; style/instructions chưa được điều khiển.",
                  wraplength=460).pack(anchor="w", pady=12)

    def _speech_tab(self):
        tab = self._tab("Văn bản")
        panes = ttk.Panedwindow(tab, orient="horizontal")
        panes.pack(fill="both", expand=True)
        editor = ttk.LabelFrame(panes, text="1. Nhập nội dung", padding=12)
        results = ttk.Frame(panes, padding=(10, 0, 0, 0))
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
        ttk.Label(editor, text="Mỗi lần thêm là một mục. Hàng đợi luôn xuất audio + SRT.",
                  style="Muted.TLabel", wraplength=270).pack(side="bottom", anchor="w", pady=6)
        self.speech = self._text(editor, height=7)
        self._queue_section(results, "Văn bản")

    def _srt_tab(self):
        tab = self._tab("SRT", scroll=True)
        self.srt_voice = self._voice_field(tab)
        self.srt_min_speed, _, _ = self._field(tab, "Tốc độ ban đầu / tối thiểu (x)", 1.0)
        ttk.Label(tab, text="Giữ mốc bắt đầu và khoảng lặng; chỉ tăng tốc câu vượt khung.\n"
                  "Tốc độ ban đầu 0.5–2.0x; Rubber Band giữ cao độ.", wraplength=850).pack(anchor="w", pady=10)
        row = ttk.Frame(tab)
        row.pack(fill="x")
        self._button(row, "Thêm nhiều file SRT", lambda: self.choose_queue_files("SRT"), primary=True)
        self._queue_section(tab, "SRT")

    def _video_tab(self):
        tab = self._tab("Video")
        tab.configure(padding=(12, 4))
        self.video_segments = []
        self.video_selected = None
        self.video_duration = 0.0
        self.video_old_markers = ()
        self.video_new_markers = ()
        self.video_marker_mapping = None
        self._video_manual_segments = None
        self._video_open_path = None
        self._video_sync = False
        row = ttk.Frame(tab)
        row.pack(fill="x")
        ttk.Label(row, text="Video", style="Muted.TLabel").pack(side="left", padx=(0, 8))
        self.video_path = tk.StringVar()
        ttk.Entry(row, textvariable=self.video_path).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._button(row, "Chọn…", self.choose_video).pack_configure(pady=0)
        self._button(row, "Mở video", self.inspect_video).pack_configure(pady=0)
        self.video_details = tk.StringVar(value="Chọn video để xem và kéo chọn các đoạn tốc độ.")
        workspace = ttk.Panedwindow(tab, orient="horizontal")
        workspace.pack(fill="both", expand=True, pady=(6, 0))
        viewer = ttk.Frame(workspace)
        sidebar = ttk.Frame(workspace, width=245)
        workspace.add(viewer, weight=3)
        workspace.add(sidebar, weight=1)
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(0, weight=1)
        self.video_preview = VideoPreview(viewer, on_position=self._video_position, height=150)
        self.video_preview.grid(row=0, column=0, sticky="nsew")
        playback = ttk.Frame(viewer)
        playback.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        self.video_play_button = ttk.Button(playback, text="▶ Phát", command=lambda: self._guard(self.toggle_video_play), state="disabled")
        self.video_play_button.pack(side="left")
        self.video_time = tk.StringVar(value="00:00.00 / 00:00.00")
        ttk.Label(playback, textvariable=self.video_time, style="Muted.TLabel").pack(side="left", padx=8)
        self.video_timeline = VideoTimeline(viewer, on_select=self._video_select,
                                           on_change=self._video_segments_changed, on_seek=self._video_seek)
        self.video_timeline.grid(row=2, column=0, sticky="ew")
        self.video_preview.set_audio_enabled(False)
        canvas = tk.Canvas(sidebar, width=245, highlightthickness=0, background="#ffffff")
        scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        settings = ttk.Frame(canvas, padding=(12, 0))
        ttk.Label(settings, textvariable=self.video_details, style="Muted.TLabel",
                  wraplength=225, font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 7))
        window = canvas.create_window((0, 0), window=settings, anchor="nw")
        settings.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        def wheel(event):
            canvas.yview_scroll(-int(event.delta / 120), "units")
            return "break"
        canvas.bind("<MouseWheel>", wheel)
        settings.bind("<MouseWheel>", wheel)
        markers = ttk.LabelFrame(settings, text="Tự căn video theo mốc", padding=8)
        markers.pack(fill="x", pady=(0, 12))
        self.video_old_marks_path = tk.StringVar()
        self.video_new_marks_path = tk.StringVar()
        actions = ttk.Frame(markers)
        actions.pack(fill="x")
        self._button(actions, "Mốc cũ…", lambda: self.import_video_markers("old"))
        self._button(actions, "Mốc mới…", lambda: self.import_video_markers("new"))
        self.video_marker_status = tk.StringVar(value="Nhập hai file số thứ tự + HH:MM:SS,mmm")
        ttk.Label(markers, textvariable=self.video_marker_status, style="Muted.TLabel",
                  wraplength=215, font=("Segoe UI", 8)).pack(anchor="w", pady=4)
        self.video_marker_table = ttk.Treeview(markers, columns=("index", "old", "new"),
                                              show="headings", height=4, selectmode="browse")
        for key, title, width in (("index", "#", 30), ("old", "Gốc (s)", 85), ("new", "Mới (s)", 85)):
            self.video_marker_table.heading(key, text=title)
            self.video_marker_table.column(key, width=width, minwidth=25, stretch=key != "index")
        self.video_marker_table.pack(fill="x", pady=4)
        self.video_marker_table.bind("<<TreeviewSelect>>", self._video_marker_selected)
        actions = ttk.Frame(markers)
        actions.pack(fill="x")
        self._button(actions, "Căn theo mốc", self.align_video_markers, primary=True)
        self._button(actions, "Bỏ căn", self.clear_video_marker_alignment)
        ttk.Label(markers, text="Ghép theo số thứ tự; phần đuôi giữ 1x.", style="Muted.TLabel",
                  wraplength=215, font=("Segoe UI", 8)).pack(anchor="w")
        self.video_selection_label = tk.StringVar(value="Chọn đoạn trên timeline")
        ttk.Label(settings, textvariable=self.video_selection_label, style="Section.TLabel").pack(anchor="w", pady=(0, 7))
        created_fields = []
        def field(label, default):
            line = ttk.Frame(settings)
            line.pack(fill="x", pady=3)
            ttk.Label(line, text=label, width=13).pack(side="left")
            variable = tk.StringVar(value=str(default))
            entry = ttk.Entry(line, textvariable=variable, width=10)
            entry.pack(side="left", fill="x", expand=True)
            created_fields.append(entry)
            return variable
        self.video_start = field("Bắt đầu (s)", 0)
        self.video_end = field("Kết thúc (s)", 10)
        self.video_speed = field("Tốc độ (x)", 1.5)
        self.video_speed_slider = tk.DoubleVar(value=1.5)
        self.video_speed_control = ttk.Scale(settings, from_=0.25, to=4, variable=self.video_speed_slider,
                                             command=self._video_speed_drag)
        self.video_speed_control.pack(fill="x", pady=5)
        self.video_ramp = field("Chuyển mỗi đầu (s)", 0.5)
        self.video_manual_fields = list(created_fields)
        actions = ttk.Frame(settings)
        actions.pack(fill="x", pady=4)
        self.video_manual_buttons = [self._button(actions, "Thêm đoạn", self.add_video_segment, primary=True),
                                     self._button(actions, "Áp dụng", self.apply_video_segment)]
        self.video_segment_list = ttk.Treeview(settings, columns=("range", "speed"), show="headings", height=4,
                                               selectmode="browse")
        self.video_segment_list.heading("range", text="Đoạn (giây)")
        self.video_segment_list.heading("speed", text="Tốc độ")
        self.video_segment_list.column("range", width=135, minwidth=85)
        self.video_segment_list.column("speed", width=65, minwidth=50)
        self.video_segment_list.pack(fill="x", pady=5)
        self.video_segment_list.bind("<<TreeviewSelect>>", self._video_list_select)
        actions = ttk.Frame(settings)
        actions.pack(fill="x")
        self.video_manual_buttons.extend([self._button(actions, "Xóa đoạn", self.delete_video_segment),
                                          self._button(actions, "Xóa tất cả", self.clear_video_segments)])
        self.video_output_duration = tk.StringVar(value="Các phần chưa chọn giữ tốc độ 1.0x")
        ttk.Label(settings, textvariable=self.video_output_duration, style="Muted.TLabel", wraplength=240).pack(anchor="w", pady=6)
        self.video_quality = field("Chất lượng (18–28)", 20)
        self.video_preset = tk.StringVar(value="fast")
        ttk.Label(settings, text="Preset xuất (nhanh → chậm)").pack(anchor="w")
        ttk.Combobox(settings, textvariable=self.video_preset,
                     values=["ultrafast", "veryfast", "faster", "fast", "medium"],
                     state="readonly").pack(fill="x", pady=4)
        ttk.Label(settings, text="Tự dùng GPU nếu hỗ trợ. veryfast ưu tiên tốc độ xuất.",
                  style="Muted.TLabel", wraplength=240).pack(anchor="w", pady=4)
        def bind_wheel(widget):
            widget.bind("<MouseWheel>", wheel, add="+")
            for child in widget.winfo_children():
                bind_wheel(child)
        bind_wheel(settings)
        page = self.pages["Video"]
        row = ttk.Frame(page, padding=(16, 4))
        row.pack(side="bottom", fill="x", before=page.winfo_children()[0])
        self.video_export_button = self._button(row, "Xuất video", self.export_video, primary=True)
        self.video_path.trace_add("write", self._video_path_changed)

    def _video_path_changed(self, *_):
        self._video_open_path = None
        self.video_preview.unload()
        self.video_duration = 0.0
        self._video_position(0, False)
        self.video_segments = []
        self.video_selected = None
        self.video_timeline.set_video(0)
        self.video_segment_list.delete(*self.video_segment_list.get_children())
        self.video_play_button.configure(state="disabled")
        self.video_selection_label.set("Chọn đoạn trên timeline")
        self.video_output_duration.set("Mở video để chỉnh các đoạn tốc độ")
        self.video_details.set("Bấm Mở video để xem và chỉnh file đã chọn.")
        self._reset_video_marker_state()

    def choose_video(self):
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("Video", "*.mp4 *.mov *.mkv *.avi *.webm"), ("Tất cả", "*.*")])
        if path:
            self.video_path.set(path)
            self.inspect_video()

    def inspect_video(self):
        if self.busy:
            raise RuntimeError("Hãy chờ tác vụ hiện tại hoàn tất trước khi mở video.")
        source = self.video_path.get().strip()
        if not source:
            raise ValueError("Hãy chọn video đầu vào.")
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise ValueError("Không tìm thấy video đầu vào.")
        self._video_open_path = path
        self.video_details.set("Đang mở video…")
        self.video_preview.open(path, on_loaded=self._video_loaded, on_error=self._video_error)

    def _video_loaded(self, info):
        self._reset_video_marker_state()
        self.video_duration = float(info["duration"])
        self.video_segments = []
        self.video_selected = None
        self.video_timeline.set_video(self.video_duration)
        self.video_start.set("0")
        self.video_end.set(f"{min(10, self.video_duration):.6f}")
        self.video_ramp.set(f"{min(0.5, self.video_duration / 2):g}")
        self.video_play_button.configure(state="normal")
        self.video_details.set(f"{info['width']} × {info['height']} · {info['fps']:g} FPS · "
                               f"{self.video_duration:.2f}s · {'có âm thanh' if info['has_audio'] else 'không có âm thanh'}")
        self._refresh_video_segments()
        self.status.set("Video đã mở. Kéo chọn vùng trên timeline rồi bấm Thêm đoạn.")

    def _video_error(self, message):
        self.video_play_button.configure(state="normal" if self.video_preview.ready else "disabled", text="▶ Phát")
        self.video_details.set(str(message))
        self.status.set(str(message))
        self._log(str(message))

    def _video_position(self, seconds, playing):
        self.video_timeline.set_position(seconds)
        if self.video_marker_mapping is not None:
            output = float(self.video_marker_mapping.output_time(seconds))
            self.video_time.set(f"Gốc {time_label(seconds)} · Mới {time_label(output)}")
        else:
            self.video_time.set(f"{time_label(seconds)} / {time_label(self.video_duration)}")
        self.video_play_button.configure(text="Ⅱ Tạm dừng" if playing else "▶ Phát")

    def toggle_video_play(self):
        if self.busy:
            raise RuntimeError("Hãy chờ xuất video hoàn tất trước khi xem trước.")
        if self.video_preview.playing:
            self.video_preview.pause()
        else:
            self.video_preview.play()

    def _video_seek(self, seconds):
        self.video_preview.seek(seconds)

    def _video_select(self, start, end, index):
        self.video_selected = index
        self.video_start.set(f"{start:.6f}")
        self.video_end.set(f"{end:.6f}")
        self.video_selection_label.set(f"Đoạn {index + 1}" if index is not None else "Đoạn mới · bấm Thêm đoạn")
        previous_sync = self._video_sync
        self._video_sync = True
        try:
            if index is not None:
                segment = self.video_timeline.segments[index]
                self.video_speed.set(f"{segment.speed:g}")
                self.video_ramp.set(f"{segment.ramp_seconds:g}")
                self.video_speed_slider.set(segment.speed)
                self.video_segment_list.selection_set(str(index))
            else:
                self.video_segment_list.selection_remove(*self.video_segment_list.selection())
                try:
                    self.video_ramp.set(f"{min(float(self.video_ramp.get()), (end - start) / 2):g}")
                except ValueError:
                    self.video_ramp.set("0")
                self.video_preview.seek(start)
        finally:
            self._video_sync = previous_sync

    def _video_segments_changed(self, segments, index):
        self._ensure_video_manual_edit()
        self.video_segments = list(segments)
        self.video_selected = index
        self._refresh_video_segments()

    def _video_speed_drag(self, value):
        if self._video_sync or self.video_marker_mapping is not None:
            return
        speed = round(float(value), 2)
        self.video_speed.set(f"{speed:.2f}")
        if self.video_selected is not None and not self.busy:
            # Keep exact timeline endpoints, including adjacent fractional ranges.
            original = self.video_segments[self.video_selected]
            self.video_segments[self.video_selected] = SpeedSegment(
                original.start, original.end, speed, original.ramp_seconds)
            self._refresh_video_segments()

    def _video_list_select(self, event=None):
        if self._video_sync:
            return
        selection = self.video_segment_list.selection()
        if selection:
            index = int(selection[0])
            if index == self.video_selected:
                return
            self.video_timeline.set_segments(self.video_segments, index)
            segment = self.video_segments[index]
            self._video_select(segment.start, segment.end, index)

    def _video_form_segment(self):
        self._ensure_video_manual_edit()
        if not self.video_duration:
            raise ValueError("Hãy mở video trước khi thêm đoạn tốc độ.")
        variables = (self.video_start, self.video_end, self.video_speed, self.video_ramp)
        values = [float(variable.get()) for variable in variables]
        if self.video_selected is not None:
            original = self.video_segments[self.video_selected]
            exact = (original.start, original.end, original.speed, original.ramp_seconds)
            displayed = (f"{exact[0]:.6f}", f"{exact[1]:.6f}", f"{exact[2]:g}", f"{exact[3]:g}")
            values = [old if variable.get() == label else value
                      for variable, value, old, label in zip(variables, values, exact, displayed)]
        start, end, speed, ramp = values
        segment = SpeedSegment(start, end, speed, ramp)
        build_multi_speed_time_map(self.video_duration, [segment])
        return segment

    def add_video_segment(self):
        segment = self._video_form_segment()
        segments = [*self.video_segments, segment]
        build_multi_speed_time_map(self.video_duration, segments)
        self.video_segments = sorted(segments, key=lambda item: item.start)
        self.video_selected = self.video_segments.index(segment)
        self._refresh_video_segments()

    def apply_video_segment(self):
        if self.video_selected is None:
            raise ValueError("Chọn một đoạn đã thêm trước khi áp dụng thay đổi.")
        segment = self._video_form_segment()
        segments = list(self.video_segments)
        segments[self.video_selected] = segment
        build_multi_speed_time_map(self.video_duration, segments)
        self.video_segments = sorted(segments, key=lambda item: item.start)
        self.video_selected = self.video_segments.index(segment)
        self._refresh_video_segments()

    def delete_video_segment(self):
        self._ensure_video_manual_edit()
        if self.video_selected is None:
            raise ValueError("Chọn đoạn muốn xóa trên timeline hoặc trong danh sách.")
        self.video_segments.pop(self.video_selected)
        self.video_selected = None
        self._refresh_video_segments()

    def clear_video_segments(self):
        self._ensure_video_manual_edit()
        self.video_segments = []
        self.video_selected = None
        self._refresh_video_segments()

    def _refresh_video_segments(self):
        self.video_timeline.set_segments(self.video_segments, self.video_selected)
        self.video_timeline.set_markers(self.video_old_markers, self.video_new_markers)
        self._update_video_edit_mode()
        if self.video_marker_mapping is not None:
            self.video_preview.set_time_map(self.video_marker_mapping)
        else:
            self.video_preview.set_segments(tuple(self.video_segments))
        self._video_sync = True
        try:
            self.video_segment_list.delete(*self.video_segment_list.get_children())
            for index, segment in enumerate(self.video_segments):
                self.video_segment_list.insert("", "end", iid=str(index),
                                               values=(f"{segment.start:.2f}–{segment.end:.2f}", f"{segment.speed:g}x"))
            if self.video_selected is not None:
                segment = self.video_segments[self.video_selected]
                self._video_select(segment.start, segment.end, self.video_selected)
            else:
                self.video_selection_label.set("Chọn đoạn trên timeline")
        finally:
            self._video_sync = False
        if self.video_duration:
            mapping = self.video_marker_mapping or build_multi_speed_time_map(self.video_duration, self.video_segments)
            label = f"{len(self.video_old_markers)} mốc đã căn" if self.video_marker_mapping else f"{len(self.video_segments)} đoạn"
            self.video_output_duration.set(f"{label} · video xuất {mapping.output_duration:.3f}s")

    def _ensure_video_manual_edit(self):
        if self.busy:
            raise RuntimeError("Hãy chờ tác vụ hiện tại hoàn tất trước khi chỉnh đoạn.")
        if self.video_marker_mapping is not None:
            raise ValueError("Bấm Bỏ căn trước khi chỉnh tốc độ bằng tay để giữ đúng các mốc mới.")

    def _update_video_edit_mode(self):
        locked = self.video_marker_mapping is not None
        self.video_timeline.set_editable(not locked)
        state = "disabled" if locked or self.busy else "normal"
        for widget in [*self.video_manual_fields, self.video_speed_control, *self.video_manual_buttons]:
            widget.configure(state=state)

    def _reset_video_marker_state(self):
        self.video_old_markers = ()
        self.video_new_markers = ()
        self.video_marker_mapping = None
        self._video_manual_segments = None
        self.video_old_marks_path.set("")
        self.video_new_marks_path.set("")
        self.video_marker_table.delete(*self.video_marker_table.get_children())
        self.video_marker_status.set("Nhập hai file số thứ tự + HH:MM:SS,mmm")
        self.video_timeline.set_markers(())
        self._update_video_edit_mode()

    def import_video_markers(self, kind="old"):
        if self.busy:
            raise RuntimeError("Hãy chờ tác vụ hiện tại hoàn tất trước khi nhập mốc.")
        if not self.video_duration:
            raise ValueError("Hãy mở video trước khi nhập các mốc thời gian.")
        if kind not in ("old", "new"):
            raise ValueError("Chọn file mốc cũ hoặc mốc mới.")
        label = "cũ" if kind == "old" else "mới"
        path = filedialog.askopenfilename(parent=self.root, title=f"Chọn file mốc {label}",
                                         filetypes=[("File mốc thời gian", "*.txt *.srt"), ("Tất cả", "*.*")])
        if not path:
            return
        markers = read_marker_file(path)
        if kind == "old" and markers[-1].time_ms / 1000 > self.video_duration:
            raise ValueError("Mốc cũ vượt thời lượng video đang mở.")
        # Parse and validate before changing an already applied alignment.
        if self.video_marker_mapping is not None:
            self.clear_video_marker_alignment()
        if kind == "old":
            self.video_old_markers = markers
            self.video_old_marks_path.set(str(path))
        else:
            self.video_new_markers = markers
            self.video_new_marks_path.set(str(path))
        self.video_timeline.set_markers(self.video_old_markers, self.video_new_markers)
        self._refresh_video_marker_rows()
        self.status.set(f"Đã nhập {len(markers)} mốc {label}. Nhập đủ hai file rồi bấm Căn theo mốc.")

    def _refresh_video_marker_rows(self):
        old_by_id = {marker.index: marker for marker in self.video_old_markers}
        new_by_id = {marker.index: marker for marker in self.video_new_markers}
        self.video_marker_table.delete(*self.video_marker_table.get_children())
        rows = list(self.video_old_markers or self.video_new_markers)
        def label(marker):
            if marker is None:
                return "—"
            return f"{marker.time_ms // 1000}.{marker.time_ms % 1000:03d}"
        for marker in rows:
            self.video_marker_table.insert("", "end", iid=str(marker.index),
                                           values=(marker.index, label(old_by_id.get(marker.index)),
                                                   label(new_by_id.get(marker.index))))
        old_name = Path(self.video_old_marks_path.get()).name or "chưa chọn"
        new_name = Path(self.video_new_marks_path.get()).name or "chưa chọn"
        applied = "Đã căn theo mốc · đuôi giữ 1x" if self.video_marker_mapping else "Chưa áp dụng căn mốc"
        self.video_marker_status.set(f"Cũ: {old_name} ({len(self.video_old_markers)})\n"
                                     f"Mới: {new_name} ({len(self.video_new_markers)})\n{applied}")

    def _video_marker_selected(self, event=None):
        selected = self.video_marker_table.selection()
        if selected:
            index = int(selected[0])
            marker = next((item for item in self.video_old_markers if item.index == index), None)
            if marker is not None:
                self.video_preview.seek(marker.time_ms / 1000)

    def align_video_markers(self):
        if self.busy:
            raise RuntimeError("Hãy chờ tác vụ hiện tại hoàn tất trước khi căn mốc.")
        if not self.video_duration:
            raise ValueError("Hãy mở video trước khi căn các mốc thời gian.")
        if not self.video_preview.ready:
            raise ValueError("Video chưa sẵn sàng để xem trước. Hãy mở lại video rồi căn mốc.")
        mapping = build_marker_time_map(self.video_duration, self.video_old_markers, self.video_new_markers)
        if self.video_marker_mapping is None:
            self._video_manual_segments = tuple(self.video_segments)
        self.video_marker_mapping = mapping
        self.video_segments = list(mapping.segments)
        self.video_selected = None
        self._refresh_video_segments()
        self._refresh_video_marker_rows()
        self._video_position(self.video_preview.position, self.video_preview.playing)
        self.status.set(f"Đã căn {len(mapping.old_markers)} mốc · phần đuôi giữ 1x. Phát để xem trước rồi Xuất video.")

    def clear_video_marker_alignment(self):
        if self.busy:
            raise RuntimeError("Hãy chờ tác vụ hiện tại hoàn tất trước khi bỏ căn mốc.")
        if self.video_marker_mapping is not None:
            self.video_segments = list(self._video_manual_segments or ())
            self.video_marker_mapping = None
            self._video_manual_segments = None
            self.video_selected = None
            self._refresh_video_segments()
            self._video_position(self.video_preview.position, self.video_preview.playing)
        self._refresh_video_marker_rows()
        self.status.set("Đã bỏ căn mốc và khôi phục các đoạn chỉnh tay; các file mốc vẫn được giữ.")

    def export_video(self):
        path = self.video_path.get().strip()
        if not path:
            raise ValueError("Hãy chọn video đầu vào.")
        if not self.video_segments:
            raise ValueError("Hãy thêm ít nhất một đoạn tốc độ trên timeline.")
        segments = tuple(self.video_segments)
        quality, preset = int(self.video_quality.get()), self.video_preset.get()
        self.video_preview.pause()
        started = time.monotonic()
        last_update, last_stage = 0.0, ""

        def progress(fraction):
            nonlocal last_update, last_stage
            now = time.monotonic()
            stage = ("Chuẩn bị xuất video" if fraction == 0 else
                     "Mã hóa video" if fraction < 1 else "Xuất video hoàn tất")
            if stage != last_stage or fraction >= 1 or now - last_update >= 0.2:
                self._post("status", f"{stage} · {fraction * 100:.0f}% · đã chạy {now - started:.0f}s")
                last_update, last_stage = now, stage
        if self.video_marker_mapping is not None:
            old, new = tuple(self.video_old_markers), tuple(self.video_new_markers)
            self._job(lambda: self.tool.edit_video_markers(path, old, new, keep_audio=False,
                                                          quality=quality, preset=preset, progress=progress), self._result)
        else:
            self._job(lambda: self.tool.edit_video_segments(path, segments, keep_audio=False,
                                                           quality=quality, preset=preset, progress=progress), self._result)


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
        self._refresh_inline_history()

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
        self.events.put((kind, value))

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _drain(self):
        if self.closed:
            return
        queue_history_dirty = False
        for _ in range(100):
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status.set(value)
            elif kind == "log":
                self._log(value)
            elif kind == "queue":
                page, identifier, state, payload = value
                self.queue_panels[page].update_task(identifier, state, payload)
                if state in ("done", "error", "cancelled"):
                    queue_history_dirty = True
                if state == "error":
                    self._log(str(payload))
                elif state == "done" and isinstance(payload, tuple) and len(payload) > 1:
                    self._log(f"{payload[0]}: {payload[1]}")
                elif state == "done" and getattr(payload, "timings", None):
                    timing = payload.timings
                    self._log(f"{payload.audio.name}: total={timing['total']:.2f}s, "
                              f"model={timing['model']:.2f}s, effects/export={timing['other']:.2f}s")
            elif kind == "done":
                self.busy = False
                self.video_timeline.enabled = True
                self.progress.stop()
                for button in self.action_buttons:
                    button.configure(state="normal")
                self._update_video_edit_mode()
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
        if queue_history_dirty:
            self.refresh_history()
        self._drain_handle = self.root.after(80, self._drain)

    def _job(self, function, callback=lambda result: None):
        if self.busy:
            raise RuntimeError("Đang có tác vụ chạy. Hãy chờ hoặc bấm Dừng.")
        self.busy = True
        self.video_preview.pause()
        self.video_timeline.enabled = False
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
            var = self.vars.get(name)
            default = getattr(defaults, name)
            values[name] = default if var is None else bool(var.get()) if isinstance(default, bool) else type(default)(var.get())
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
        if self._voice_model is self.tool.tts:
            return
        names = self.tool.voice_names()
        default = getattr(self.tool.tts, "_default_voice", "") if self.tool.tts else ""
        for variable, box in self.voice_boxes:
            box.configure(values=names)
            if variable.get() not in names:
                variable.set(default if default in names else (names[0] if names else ""))
        self._voice_model = self.tool.tts
        self.model_state.set(f"{self.tool.tts.backend.upper()} · {len(names)} giọng" if self.tool.tts else "Model chưa tải")

    def _queue_section(self, parent, page):
        tabs = ttk.Notebook(parent)
        tabs.pack(fill="both", expand=True)
        queue_page = ttk.Frame(tabs, padding=10)
        tabs.add(queue_page, text="Hàng đợi")
        actions = ttk.Frame(queue_page)
        actions.pack(fill="x", pady=(0, 7))
        self._button(actions, "Chạy hàng đợi", lambda: self.start_queue(page), primary=True).configure(style="CompactPrimary.TButton")
        self._button(actions, "Bỏ mục chọn", lambda: self.remove_queue_items(page)).configure(style="Compact.TButton")
        ttk.Button(actions, text="Dừng", style="Compact.TButton", command=self.stop).pack(side="left", padx=3)
        ttk.Label(queue_page, text="Audio + SRT từng câu" if page == "Văn bản" else "Audio khớp mốc SRT",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 8))
        panel = QueuePanel(queue_page)
        panel.pack(fill="both", expand=True)
        self.queue_panels[page] = panel
        history_page = ttk.Frame(tabs, padding=10)
        tabs.add(history_page, text="Lịch sử tạo")
        row = ttk.Frame(history_page)
        row.pack(fill="x", pady=(0, 8))
        self._button(row, "Phát", lambda: self.play_inline_history(page)).configure(style="CompactPrimary.TButton")
        ttk.Button(row, text="Dừng", style="Compact.TButton", command=self.stop).pack(side="left", padx=3)
        self._button(row, "Xóa", lambda: self.delete_inline_history(page)).configure(style="Compact.TButton")
        self._button(row, "Làm mới", self.refresh_history).configure(style="Compact.TButton")
        self._button(row, "Thư mục", lambda: self.open_inline_history(page, folder=True)).configure(style="Compact.TButton")
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
            table.delete(*table.get_children())
            paths.clear()
            for record in self.history_records:
                path = record.path
                if path.suffix.lower() not in (".wav", ".flac", ".mp3"):
                    continue
                wanted = path.name.startswith("srt_") if page == "SRT" else path.name.startswith(("speech_", "stream_"))
                if wanted:
                    identifier = table.insert("", "end", values=(path.name,
                        datetime.fromtimestamp(record.modified).strftime("%d/%m/%Y %H:%M")))
                    paths[identifier] = path

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
        self.video_preview.pause()
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


    def close(self):
        if self.closed:
            return
        self.closed = True
        for panel in self.queue_panels.values():
            panel.stop_bars()
        self.video_preview.close()
        self.root.after_cancel(self._drain_handle)
        self.stop()
        if not self.busy and self.tool.tts is not None:
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
