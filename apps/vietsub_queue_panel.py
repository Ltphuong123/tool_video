"""Compact video queue table; all mutations stay on the Tk thread."""
from tkinter import ttk

from apps.desktop_theme import COLORS


class VietsubQueuePanel(ttk.Frame):
    def __init__(self, parent, *, select, guard):
        super().__init__(parent)
        self.tasks, self.rows = [], {}
        self.busy = False
        self._select, self._guard = select, guard
        actions = self.actions = ttk.Frame(self)
        actions.pack(fill='x', pady=(4, 5))
        self.buttons = []
        for text, command in (("Bỏ mục chọn", self.remove_selected), ("Bỏ mục đã xong", self.clear_done)):
            button = ttk.Button(actions, text=text, style='Compact.TButton', command=lambda fn=command: guard(fn))
            button.pack(side='left', padx=(0, 6))
            self.buttons.append(button)
        self.summary = ttk.Label(actions, text='0 video', style='Blue.TLabel')
        self.summary.pack(side='right')
        table = ttk.Frame(self)
        table.pack(fill='both', expand=True)
        ttk.Style(self).configure('Vietsub.Treeview', rowheight=28,
                                 background=COLORS['surface'], fieldbackground=COLORS['surface'],
                                 foreground=COLORS['ink'], borderwidth=0)
        ttk.Style(self).configure('Vietsub.Treeview.Heading', padding=(7, 5),
                                 font=('Segoe UI', 9, 'bold'))
        ttk.Style(self).map('Vietsub.Treeview', background=[('selected', '#e5e0ff')],
                           foreground=[('selected', COLORS['purple'])])
        self.tree = ttk.Treeview(table, columns=('video', 'srt', 'voice', 'speed', 'state'),
                                 show='headings', selectmode='extended', style='Vietsub.Treeview', height=4)
        self.tree.tag_configure('alternate', background='#f4f6ff')
        for state, color in (('pending', COLORS['muted']), ('running', COLORS['purple']), ('done', COLORS['teal']),
                             ('error', '#c74360'), ('cancelled', '#aa701c')):
            self.tree.tag_configure(state, foreground=color)
        for key, title, width in (('video', 'Video', 155), ('srt', 'Phụ đề SRT', 135), ('voice', 'Giọng', 105),
                                  ('speed', 'Tốc độ', 62), ('state', 'Trạng thái', 170)):
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, minwidth=45, stretch=key in ('video', 'srt', 'state'))
        scroll = ttk.Scrollbar(table, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.tree.pack(fill='both', expand=True)
        self.tree.bind('<<TreeviewSelect>>', self._selected)
        self.tree.bind('<Delete>', lambda event: self._guard(self.remove_selected))
        self.tree.bind('<Control-a>', self._select_all)
        self._refresh_actions()

    def add(self, task):
        if self.busy:
            raise RuntimeError('Hãy chờ hoặc dừng hàng đợi trước khi thêm video.')
        if task.identifier in self.rows:
            raise ValueError('Video này đã có trong hàng đợi.')
        alternate = len(self.tasks) % 2
        self.tasks.append(task)
        self.rows[task.identifier] = dict(state='pending', result=None, message='Chờ')
        self.tree.insert('', 'end', iid=task.identifier, tags=('pending', 'alternate') if alternate else ('pending',),
                         values=(task.video.name, task.subtitles.name,
                         task.voice or 'Mặc định', f'{task.sampling.speed:g}×', 'Chờ'))
        self.refresh()

    def pending(self):
        return tuple(task for task in self.tasks if self.rows[task.identifier]['state']
                     in ('pending', 'error', 'cancelled'))

    def set_busy(self, busy):
        self.busy = busy
        self._refresh_actions()

    def update_task(self, identifier, state, value):
        row = self.rows[identifier]
        if state == 'progress':
            message = f'Bước {value.step}/5 · {value.title}'
            # Fractions are drawn by the progress card. Avoid rescanning and
            # redrawing every queue row for each generated sentence/frame.
            if row['message'] == message:
                return
        else:
            row['state'] = state
            message = {'running': 'Đang chạy', 'done': 'Xong', 'cancelled': 'Đã dừng', 'error': 'Lỗi'}.get(state, state)
            if state == 'done':
                row['result'] = value
            elif state in ('error', 'cancelled'):
                message += ': ' + str(value)
        row['message'] = message
        alternating = tuple(tag for tag in self.tree.item(identifier, 'tags') if tag == 'alternate')
        self.tree.item(identifier, tags=(row['state'], *alternating))
        self.tree.set(identifier, 'state', message)
        if state == 'running':
            self.tree.see(identifier)
        if state != 'progress':
            self.refresh()

    def _selected(self, event=None):
        self._refresh_actions()
        if self.busy:
            return
        selected = self.tree.selection()
        if len(selected) == 1:
            task = next(task for task in self.tasks if task.identifier == selected[0])
            self._guard(lambda: self._select(task, self.rows[task.identifier]))

    def _select_all(self, event=None):
        self.tree.selection_set(self.tree.get_children())
        self._refresh_actions()
        return 'break'

    def _refresh_actions(self):
        self.buttons[0].configure(state='normal' if not self.busy and self.tree.selection() else 'disabled')
        self.buttons[1].configure(state='normal' if not self.busy and any(
            row['state'] == 'done' for row in self.rows.values()) else 'disabled')

    def remove_selected(self):
        if not self.busy:
            self._remove(set(self.tree.selection()))

    def clear_done(self):
        if not self.busy:
            self._remove({key for key, row in self.rows.items() if row['state'] == 'done'})

    def _remove(self, identifiers):
        # Removing rows does not delete exported files.
        for identifier in identifiers:
            self.tree.delete(identifier)
            self.rows.pop(identifier)
        self.tasks[:] = [task for task in self.tasks if task.identifier not in identifiers]
        for index, task in enumerate(self.tasks):
            state = self.rows[task.identifier]['state']
            self.tree.item(task.identifier, tags=(state, 'alternate') if index % 2 else (state,))
        self.refresh()

    def refresh(self):
        done = sum(row['state'] == 'done' for row in self.rows.values())
        self.summary.configure(text=f'{done}/{len(self.tasks)} xong')
        self._refresh_actions()
        if hasattr(self, 'on_change'):
            self.on_change()
