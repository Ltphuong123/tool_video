"""File selection, an always-visible queue, progress and external result actions."""
from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

from apps.desktop_theme import COLORS
from apps.vietsub_queue_panel import VietsubQueuePanel


class VietsubPanel(ttk.Frame):
    def __init__(self, parent, *, start, stop, guard, open_file, make_task):
        super().__init__(parent)
        self._guard, self._open_file, self._make_task = guard, open_file, make_task
        self.busy = False
        self._resetting_inputs = False
        self._default_voice = ''
        self._path_displays = []
        self.source_video, self.source_srt = tk.StringVar(), tk.StringVar()
        self.output_video, self.output_srt, self.output_audio = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.voice, self.speed = tk.StringVar(), tk.StringVar(value='1.0')
        self.stage, self.detail = tk.StringVar(value='Sẵn sàng'), tk.StringVar()
        self.percent = tk.StringVar(value='0%')
        self.result_name = tk.StringVar(value='Chưa có kết quả')
        self.input_controls, self.result_controls = [], []
        style = ttk.Style(self)
        style.configure('Vietsub.Horizontal.TProgressbar', background=COLORS['purple'],
                        troughcolor='#ded7fb', borderwidth=0, thickness=8)
        style.configure('Vietsub.Status.TLabel', background='#eee9ff', foreground=COLORS['purple'],
                        font=('Segoe UI', 10, 'bold'))
        style.configure('Vietsub.StatusDetail.TLabel', background='#eee9ff', foreground=COLORS['muted'],
                        font=('Segoe UI', 9))
        style.configure('Vietsub.Success.TLabel', background='#eee9ff', foreground=COLORS['teal'],
                        font=('Segoe UI', 10, 'bold'))
        style.configure('Vietsub.Error.TLabel', background='#eee9ff', foreground='#c74360',
                        font=('Segoe UI', 10, 'bold'))
        style.configure('Vietsub.Results.TFrame', background='#eaf8f4')
        style.configure('Vietsub.ResultTitle.TLabel', background='#eaf8f4', foreground=COLORS['teal'],
                        font=('Segoe UI', 10, 'bold'))
        style.configure('Vietsub.ResultName.TLabel', background='#eaf8f4', foreground=COLORS['ink'],
                        font=('Segoe UI', 9))
        for widget_style in ('Vietsub.TEntry', 'Vietsub.TCombobox', 'Vietsub.TSpinbox'):
            style.configure(widget_style, padding=(5, 3))
        style.configure('Vietsub.Success.TButton', padding=(8, 6), font=('Segoe UI', 9, 'bold'))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.inputs = ttk.Frame(self, padding=(12, 6))
        self.inputs.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        self.inputs.columnconfigure(0, weight=3)
        self.inputs.columnconfigure(1, weight=2)
        self._card_title(self.inputs, '01', 'Video & giọng đọc', 'Accent.TLabel')
        paths = ttk.Frame(self.inputs)
        paths.grid(row=1, column=0, sticky='nsew', padx=(0, 18))
        self._path_row(paths, 'Video', self.source_video, lambda: self._choose('video'))
        self._path_row(paths, 'SRT', self.source_srt, lambda: self._choose('srt'))
        settings = ttk.Frame(self.inputs)
        settings.grid(row=1, column=1, sticky='nsew')
        settings.columnconfigure(1, weight=1)
        ttk.Label(settings, text='Giọng đọc').grid(row=0, column=0, sticky='w', padx=(0, 8), pady=3)
        self.voice_box = ttk.Combobox(settings, textvariable=self.voice, state='readonly', width=16,
                                     style='Vietsub.TCombobox')
        self.voice_box.grid(row=0, column=1, sticky='ew', pady=3)
        ttk.Label(settings, text='Tốc độ').grid(row=1, column=0, sticky='w', padx=(0, 8), pady=3)
        speed_row = ttk.Frame(settings)
        speed_row.grid(row=1, column=1, sticky='w', pady=3)
        self.speed_box = ttk.Spinbox(speed_row, textvariable=self.speed, from_=0.5, to=2, increment=.05, width=5,
                                    style='Vietsub.TSpinbox')
        self.speed_box.pack(side='left')
        ttk.Label(speed_row, text='×  (0.5–2.0)', style='Muted.TLabel').pack(side='left', padx=7)
        adding = ttk.Frame(self.inputs)
        adding.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(5, 0))
        for text, command, button_style in (('+ Thêm vào hàng đợi', self.add_current, 'CompactPrimary.TButton'),
                                             ('Thêm nhiều video…', self.add_videos, 'Compact.TButton')):
            button = ttk.Button(adding, text=text, style=button_style, command=lambda fn=command: guard(fn))
            button.pack(side='left', padx=(0, 7))
            self.input_controls.append((button, 'normal'))
        ttk.Label(adding, text='SRT cùng tên → tự ghép.', style='Muted.TLabel').pack(side='left', padx=3)

        self.queue_card = ttk.Frame(self, padding=(12, 8, 12, 8))
        self.queue_card.grid(row=1, column=0, sticky='nsew')
        self._card_title(self.queue_card, '02', 'Hàng đợi xử lý', 'Blue.TLabel', pack=True)
        self.queue = VietsubQueuePanel(self.queue_card, select=self._show_selected, guard=guard)
        self.queue.pack(fill='both', expand=True)
        self.queue.on_change = self._refresh_buttons
        self.start_button = ttk.Button(self.queue.actions, text='Chạy hàng đợi', style='CompactPrimary.TButton',
                                       command=lambda: guard(start))
        self.start_button.pack(side='left', padx=(0, 7), before=self.queue.buttons[0])
        self.stop_button = ttk.Button(self.queue.actions, text='Dừng', style='Danger.TButton',
                                      command=stop, state='disabled')
        self.stop_button.pack(side='left', padx=(0, 12), before=self.queue.buttons[0])

        self.progress_area = ttk.Frame(self, padding=(12, 4), style='Hero.TFrame')
        self.progress_area.grid(row=2, column=0, sticky='ew', pady=8)
        progress_header = ttk.Frame(self.progress_area, style='Hero.TFrame')
        progress_header.pack(fill='x')
        self.stage_label = ttk.Label(progress_header, textvariable=self.stage, style='Vietsub.Status.TLabel')
        self.stage_label.pack(side='left', fill='x', expand=True)
        ttk.Label(progress_header, textvariable=self.percent, style='Vietsub.Status.TLabel').pack(side='right', padx=(8, 0))
        self.progress = ttk.Progressbar(self.progress_area, mode='determinate', maximum=100,
                                        style='Vietsub.Horizontal.TProgressbar')
        self.progress.pack(fill='x', pady=(4, 3))
        self.detail_label = ttk.Label(self.progress_area, textvariable=self.detail,
                                     style='Vietsub.StatusDetail.TLabel', wraplength=700)
        self.detail_label.pack(anchor='w')

        self.results = ttk.Frame(self, padding=(10, 7), style='Vietsub.Results.TFrame')
        self.results.grid(row=3, column=0, sticky='ew')
        self.results.columnconfigure(1, weight=1)
        ttk.Label(self.results, text='03  Kết quả', style='Vietsub.ResultTitle.TLabel').grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.result_label = ttk.Label(self.results, textvariable=self.result_name, width=12, style='Vietsub.ResultName.TLabel')
        self.result_label.grid(row=0, column=1, sticky='ew')
        for column, (label, variable) in enumerate((('Video MP4', self.output_video), ('SRT', self.output_srt),
                                                   ('Giọng WAV', self.output_audio)), 2):
            button = ttk.Button(self.results, text=label, style='Compact.TButton',
                                command=lambda var=variable: guard(lambda: self._open(var)))
            button.grid(row=0, column=column, padx=(7, 0))
            self.result_controls.append((button, variable))
        self.folder_button = ttk.Button(self.results, text='Mở thư mục', style='Vietsub.Success.TButton',
                                        command=lambda: guard(self.open_result_folder))
        self.folder_button.grid(row=0, column=5, padx=(7, 0))
        self.bind('<Configure>', lambda event: self.detail_label.configure(wraplength=max(200, event.width - 24)))
        self.source_video.trace_add('write', self._input_changed)
        self.source_srt.trace_add('write', self._input_changed)
        for variable in (self.output_video, self.output_srt, self.output_audio):
            variable.trace_add('write', lambda *_: self._refresh_buttons())
        try:
            data = json.loads(files('vieneu').joinpath('assets/voices_v3_turbo.json').read_text(encoding='utf-8'))
            names = sorted(data['presets'], key=lambda name: data['presets'][name].get('featured') or 1000)
            self.set_voices(names, data.get('default_voice', ''))
        except (OSError, ValueError, KeyError, ModuleNotFoundError):
            self.set_voices([], '')
        self.set_stage(None)
        self.set_busy(False)

    @staticmethod
    def _card_title(parent, number, title, style, *, pack=False):
        header = ttk.Frame(parent)
        ttk.Label(header, text=number, style=style, font=('Segoe UI', 10, 'bold')).pack(side='left', padx=(0, 8))
        ttk.Label(header, text=title, style='Section.TLabel').pack(side='left')
        if pack:
            header.pack(fill='x')
        else:
            header.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 4))

    def _path_row(self, parent, label, variable, action):
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=2)
        ttk.Label(row, text=label, width=5).pack(side='left')
        display = tk.StringVar(value='Chưa chọn')
        self._path_displays.append(display)
        variable.trace_add('write', lambda *_: display.set(Path(variable.get()).name if variable.get() else 'Chưa chọn'))
        entry = ttk.Entry(row, textvariable=display, state='readonly', width=18, style='Vietsub.TEntry')
        entry.pack(side='left', fill='x', expand=True, padx=(0, 6))
        entry.bind('<Double-Button-1>', lambda event: self._guard(lambda: self._open(variable)))
        button = ttk.Button(row, text='Chọn…', style='Compact.TButton', command=lambda: self._guard(action))
        button.pack(side='left')
        self.input_controls.extend(((entry, 'readonly'), (button, 'normal')))

    def _choose(self, kind):
        if self.busy:
            return
        types = [('Video', '*.mp4 *.mov *.mkv *.avi *.webm')] if kind == 'video' else [('Phụ đề SRT', '*.srt')]
        value = filedialog.askopenfilename(parent=self.winfo_toplevel(), filetypes=[*types, ('Tất cả', '*.*')])
        if value:
            (self.source_video if kind == 'video' else self.source_srt).set(value)
            if kind == 'video':
                # Auto-pair a neighbouring SRT to save a file-dialog round trip.
                neighbour = Path(value).with_suffix('.srt')
                self.source_srt.set(str(neighbour) if neighbour.is_file() else '')

    def add_current(self):
        task = self._make_task(self.source_video.get(), self.source_srt.get())
        self.queue.add(task)
        self.queue.tree.see(task.identifier)
        self._reset_inputs()
        return task

    def add_videos(self):
        if self.busy:
            return
        videos = filedialog.askopenfilenames(parent=self.winfo_toplevel(), title='Chọn nhiều video',
                    filetypes=[('Video', '*.mp4 *.mov *.mkv *.avi *.webm'), ('Tất cả', '*.*')])
        missing = []
        added = 0
        for value in videos:
            video = Path(value)
            srt = video.with_suffix('.srt')
            if not srt.is_file():
                missing.append(video.name)
                continue
            self.queue.add(self._make_task(video, srt))
            added += 1
        if added:
            self._reset_inputs()
        if missing:
            raise ValueError('Chưa có SRT cùng tên cho: ' + ', '.join(missing)
                             + '. Chọn video/SRT riêng rồi bấm Thêm vào hàng đợi để ghép thủ công.')

    def _reset_inputs(self):
        # Reset only the add form, after the queue has captured its settings.
        # Keep the currently displayed result/progress available for browsing.
        self._resetting_inputs = True
        try:
            self.source_video.set('')
            self.source_srt.set('')
            self.voice.set(self._default_voice)
            self.speed.set('1.0')
        finally:
            self._resetting_inputs = False
        self._refresh_buttons()

    def _input_changed(self, *_):
        if self._resetting_inputs:
            return
        self.output_video.set('')
        self.output_srt.set('')
        self.output_audio.set('')
        self.result_name.set('Chưa có kết quả')
        self.set_stage(None)

    def show_task(self, task, result=None):
        self.source_video.set(str(task.video))
        self.source_srt.set(str(task.subtitles))
        if task.voice:
            self.voice.set(task.voice)
        self.speed.set(f'{task.sampling.speed:g}')
        if result is not None:
            self.finish(result)

    def _show_selected(self, task, row):
        self.show_task(task, row['result'])
        if row['state'] in ('error', 'cancelled'):
            self.fail(row['message'], row['state'] == 'cancelled')

    def _open(self, variable):
        path = Path(variable.get())
        if not variable.get() or not path.is_file():
            raise ValueError('File không còn tồn tại trên máy.')
        self._open_file(path)

    def open_result_folder(self):
        if not self.output_video.get():
            raise ValueError('Chọn một mục đã xong trong hàng đợi.')
        folder = Path(self.output_video.get()).parent
        if not folder.is_dir():
            raise ValueError('Thư mục kết quả không còn tồn tại.')
        self._open_file(folder)

    def set_voices(self, names, default=''):
        self.voice_box.configure(values=names)
        self._default_voice = default if default in names else (names[0] if names else '')
        if self.voice.get() not in names:
            self.voice.set(self._default_voice)

    def set_busy(self, busy, running=False):
        self.busy = busy
        self.queue.set_busy(busy)
        for widget, state in self.input_controls:
            widget.configure(state='disabled' if busy else state)
        self.voice_box.configure(state='disabled' if busy else 'readonly')
        self.speed_box.configure(state='disabled' if busy else 'normal')
        self.stop_button.configure(state='normal' if busy and running else 'disabled')
        self._refresh_buttons()

    def _refresh_buttons(self):
        pending = self.queue.pending()
        single = not self.queue.tasks and self.source_video.get().strip() and self.source_srt.get().strip()
        retry = any(self.queue.rows[task.identifier]['state'] in ('error', 'cancelled') for task in pending)
        label = (f'Tiếp tục ({len(pending)})' if retry else f'Chạy {len(pending)} video') if pending else (
                'Chạy 1 video' if single else 'Chạy hàng đợi')
        self.start_button.configure(text=label, state='normal' if not self.busy and (pending or single) else 'disabled')
        for button, variable in self.result_controls:
            button.configure(state='normal' if not self.busy and variable.get() else 'disabled')
        self.folder_button.configure(state='normal' if not self.busy and self.output_video.get() else 'disabled')

    def set_stage(self, event):
        self.progress.stop()
        self.stage_label.configure(style='Vietsub.Status.TLabel')
        if event is None:
            self.stage.set('Sẵn sàng')
            self.detail.set('Chọn video + SRT, chọn giọng và tốc độ. Mỗi video sẽ có một thư mục kết quả riêng.')
            self.percent.set('0%')
            self.progress.configure(mode='determinate', value=0)
            return
        self.stage.set(f'Bước {event.step}/5 · {event.title}')
        self.detail.set(event.detail)
        if event.fraction is None:
            self.percent.set('…')
            self.progress.configure(mode='indeterminate', value=0)
            self.progress.start(12)
        else:
            fraction = max(0, min(1, event.fraction))
            self.percent.set(f'{fraction:.0%}')
            self.progress.configure(mode='determinate', value=100 * fraction)

    def finish(self, result):
        self.output_video.set(str(result.video))
        self.output_srt.set(str(result.subtitles))
        self.output_audio.set(str(result.audio))
        self.result_name.set(result.video.parent.name)
        self.progress.stop()
        self.progress.configure(mode='determinate', value=100)
        self.percent.set('100%')
        self.stage.set('Hoàn tất · MP4, SRT và WAV đã sẵn sàng')
        self.stage_label.configure(style='Vietsub.Success.TLabel')
        self.detail.set(f'{result.sentences} mục phụ đề · {result.elapsed:.1f} giây xử lý')

    def finish_queue(self, done, failed, stopped):
        self.progress.stop()
        self.progress.configure(mode='determinate', value=0 if stopped else 100)
        self.percent.set('—' if stopped else '100%')
        self.stage.set(('Đã dừng' if stopped else 'Hoàn tất') + f' · {done} video xong, {failed} lỗi')
        self.stage_label.configure(style='Vietsub.Status.TLabel' if stopped else (
                                   'Vietsub.Error.TLabel' if failed else 'Vietsub.Success.TLabel'))
        self.detail.set('Chọn một mục đã xong trong bảng để mở MP4, SRT, WAV hoặc thư mục kết quả.'
                        + (' Bấm Tiếp tục để chạy các mục còn lại.' if stopped else ''))

    def fail(self, message, cancelled=False):
        self.progress.stop()
        self.stage_label.configure(style='Vietsub.Status.TLabel' if cancelled else 'Vietsub.Error.TLabel')
        self.progress.configure(mode='determinate')
        self.percent.set('—')
        self.stage.set('Đã dừng' if cancelled else 'Chưa hoàn tất')
        self.detail.set(str(message))

    def close(self):
        self.progress.stop()
