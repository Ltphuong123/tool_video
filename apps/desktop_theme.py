"""Shared Studio colors and native Tk styles; no image assets are required."""
from tkinter import ttk


COLORS = {
    "background": "#f2f4ff", "surface": "#ffffff", "ink": "#192441",
    "muted": "#687793", "purple": "#7357ff", "blue": "#3478f6",
    "teal": "#13a894", "orange": "#ef9b38", "border": "#dfe5f5",
    "navy": "#14213e", "lavender": "#eee9ff",
}


def apply_theme(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    c = COLORS
    root.configure(background=c["background"])
    style.configure("TFrame", background=c["surface"])
    style.configure("Shell.TFrame", background=c["background"])
    style.configure("Hero.TFrame", background=c["lavender"])
    style.configure("Sidebar.TFrame", background=c["navy"])
    style.configure("TLabel", background=c["surface"], foreground=c["ink"], font=("Segoe UI", 10))
    style.configure("Muted.TLabel", foreground=c["muted"])
    style.configure("Shell.TLabel", background=c["background"], foreground=c["muted"])
    style.configure("Hero.TLabel", background=c["lavender"], foreground=c["ink"])
    style.configure("HeroMuted.TLabel", background=c["lavender"], foreground="#675c93", font=("Segoe UI", 9))
    style.configure("Title.TLabel", background=c["lavender"], font=("Segoe UI", 20, "bold"), foreground=c["ink"])
    style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"))
    for name, color in (("Accent", c["purple"]), ("Blue", c["blue"]), ("Success", "#0c8675")):
        style.configure(f"{name}.TLabel", foreground=color, font=("Segoe UI", 10, "bold"))
    style.configure("Sidebar.TLabel", background=c["navy"], foreground="#ffffff", font=("Segoe UI", 22, "bold"))
    style.configure("SidebarMuted.TLabel", background=c["navy"], foreground="#aebddd", font=("Segoe UI", 9))
    for name, color in (("Purple", c["purple"]), ("Blue", c["blue"]), ("Teal", "#0c8675"), ("Orange", "#b06a15")):
        style.configure(f"{name}.TLabelframe", background=c["surface"], bordercolor=c["border"], borderwidth=1)
        style.configure(f"{name}.TLabelframe.Label", background=c["surface"], foreground=color,
                        font=("Segoe UI", 11, "bold"))
    style.configure("TLabelframe", background=c["surface"], bordercolor=c["border"], borderwidth=1)
    style.configure("TLabelframe.Label", background=c["surface"], foreground=c["ink"], font=("Segoe UI", 11, "bold"))
    style.configure("TButton", font=("Segoe UI", 10), padding=(10, 7), background="#edf1ff",
                    foreground=c["ink"], borderwidth=0, focusthickness=1, focuscolor=c["purple"])
    style.map("TButton", background=[("disabled", "#edf0f6"), ("active", "#dfe7ff")],
              foreground=[("disabled", "#929cb0")])
    for name, color, active in (("Primary", c["purple"], "#5c40e6"), ("Success", "#0c8675", "#077568")):
        style.configure(f"{name}.TButton", background=color, foreground="#ffffff", font=("Segoe UI", 10, "bold"))
        style.map(f"{name}.TButton", background=[("disabled", "#e1dcfa"), ("active", active)],
                  foreground=[("disabled", "#9290aa")])
    style.configure("Compact.TButton", font=("Segoe UI", 9), padding=(8, 6))
    style.configure("CompactPrimary.TButton", font=("Segoe UI", 9, "bold"), padding=(9, 6),
                    background=c["purple"], foreground="#ffffff")
    style.map("CompactPrimary.TButton", background=[("disabled", "#e1dcfa"), ("active", "#5c40e6")],
              foreground=[("disabled", "#9290aa")])
    style.configure("Danger.TButton", background="#fff0f3", foreground="#b73355", padding=(9, 6))
    style.map("Danger.TButton", background=[("disabled", "#f3edf0"), ("active", "#ffe1e9")])
    style.configure("Nav.TButton", background=c["navy"], foreground="#b9c7e7", anchor="w",
                    padding=(12, 12), font=("Segoe UI", 10), focusthickness=0)
    style.map("Nav.TButton", background=[("selected", "#7357ff"), ("active", "#253658")],
              foreground=[("selected", "#ffffff"), ("active", "#ffffff")])
    for name, color in (("Blue", "#276de4"), ("Teal", "#098675"), ("Orange", "#aa681d"), ("Slate", "#425680")):
        style.configure(f"{name}.Nav.TButton")
        style.map(f"{name}.Nav.TButton", background=[("selected", color), ("active", "#253658")],
                  foreground=[("selected", "#ffffff"), ("active", "#ffffff")])
    style.configure("TEntry", fieldbackground="#f8f9ff", bordercolor=c["border"], padding=6)
    style.map("TEntry", bordercolor=[("focus", c["purple"])])
    style.configure("TCombobox", fieldbackground="#f8f9ff", padding=5, arrowsize=13)
    style.map("TCombobox", fieldbackground=[("readonly", "#f8f9ff")], foreground=[("readonly", c["ink"])])
    style.configure("TSpinbox", fieldbackground="#f8f9ff", padding=5, arrowsize=12)
    style.configure("TCheckbutton", background=c["surface"], font=("Segoe UI", 9))
    style.map("TCheckbutton", background=[("active", c["surface"])])
    style.configure("Treeview", background=c["surface"], fieldbackground=c["surface"], foreground=c["ink"],
                    rowheight=32, font=("Segoe UI", 9), borderwidth=0)
    style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), background="#edf1ff",
                    foreground="#58658b", padding=(8, 8), relief="flat")
    style.map("Treeview", background=[("selected", "#e7e0ff")], foreground=[("selected", "#423086")])
    style.configure("Queue.Treeview", rowheight=34)
    style.configure("Studio.TNotebook", background=c["background"], borderwidth=0, tabmargins=0)
    style.layout("Studio.TNotebook.Tab", [])
    style.configure("TNotebook", background=c["surface"], borderwidth=0, tabmargins=(0, 0, 0, 6))
    style.configure("TNotebook.Tab", font=("Segoe UI", 9, "bold"), padding=(14, 7), background="#edf1ff", borderwidth=0)
    style.map("TNotebook.Tab", background=[("selected", "#e7e0ff"), ("active", "#eef0ff")],
              foreground=[("selected", "#6244c5")])
    style.configure("Horizontal.TProgressbar", background=c["purple"], troughcolor="#edeefa", borderwidth=0, thickness=5)
    for name, background, foreground in (("Ready", "#dcf7ef", "#097760"), ("Loading", "#fff0d4", "#9d6615"),
                                         ("Idle", "#e5e8f1", "#687793")):
        style.configure(f"{name}.Badge.TLabel", background=background, foreground=foreground,
                        font=("Segoe UI", 9, "bold"), padding=(10, 5))
    return style
