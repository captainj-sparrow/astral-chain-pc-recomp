"""The installer window: python -m installer.gui

A front end over the CLI (docs/installer.md). The first screen asks what the player wants: a first install (the base
game, the update, the keys file, the game language and an empty install folder) or an update of an installed folder
(the folder alone; the game files only when that installation needs them again). It runs `python -m installer ...` as
a subprocess and shows its JSON events. Decisions live in guilogic.py; this file only draws. The window text is in
installer/text/<en|ru|pt-BR|zh-Hans>.json under "ui.*" and "launcher.*" ids.

    --selftest [--shots DIR]   build every screen in every language and close by itself (PNG per screen with --shots)
    --lang en|ru|pt-BR|zh-Hans window language (default: the Windows display language)
    --rebuild DIR              rebuild the install folder DIR for its mods (astralchain.exe starts this, build-cache/rebuild.txt):
                               the run starts at once with that folder, and the game starts again when it succeeds
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import ImageGrab   # only the screenshots of --selftest --shots need it; the release has no PIL
except ImportError:
    ImageGrab = None

from . import ini
from . import layout as layouts
from . import state as statefile
from .layout import SUPPORTED
from .guilogic import (DONE, FAILED, INSTALL, PENDING, RUNNING, SKIPPED, UI_LANGUAGE_LABELS, UI_LANGUAGES, UPDATE,
                       Catalog, Form, Model, Process, amount, default_game_language, describe_event, find_keys,
                       inspect_folder, install_subfolder, parse_line, sibling_game_files, windows_ui_language)

POLL_MS = 100
GLYPH = {PENDING: '\u25cb', RUNNING: '\u25b6', DONE: '\u2714', SKIPPED: '\u2013', FAILED: '\u2716'}
COLOR = {PENDING: '#808080', RUNNING: '#1a5fb4', DONE: '#26a269', SKIPPED: '#808080', FAILED: '#c01c28'}
OK_COLOR, BAD_COLOR, MUTED = '#26a269', '#c01c28', '#5e5c64'
CARD, CARD_HOVER, CARD_BORDER, ACCENT = '#ffffff', '#eef4fc', '#c8c8c8', '#1a5fb4'
PAD = 14
WRAP = 620
FILES = (('base', 'ui.base', 'ui.pick_game'), ('update', 'ui.update', 'ui.pick_update'), ('keys', 'ui.keys', 'ui.pick_keys'))


def fix_tcl_paths():
    """The embeddable Python keeps Tcl/Tk next to python.exe (python/tcl); point Tcl at it when it is not found."""
    tcl = Path(sys.executable).resolve().parent / 'tcl'
    if tcl.is_dir() and 'TCL_LIBRARY' not in os.environ:
        for lib, var in (('tcl9.0', 'TCL_LIBRARY'), ('tk9.0', 'TK_LIBRARY')):
            if (tcl / lib).is_dir():
                os.environ[var] = str(tcl / lib)


def dpi_aware():
    """Draw at the screen's real resolution on Windows. Python's exe declares no DPI awareness, so at 125 % or 150 %
    Windows stretches the window as a bitmap and every letter blurs. Must run before Tk opens its first window; Tk then
    reads the real DPI and sizes its point-sized fonts to it (App.px scales the pixel sizes)."""
    if os.name != 'nt':
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)   # system DPI aware
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()     # before Windows 8.1
        except (AttributeError, OSError):
            pass


# A failed check the player may carry on past: the window offers the same run again with --accept-unknown-version, and
# the CLI then reports it as a warning. The game may not run; that is the player's call.
ACCEPT_ANYWAY = frozenset({'check.base_wrong_version', 'extract.exefs_mismatch', 'extract.romfs_mismatch'})


class App:
    def __init__(self, root, ui_lang, layout, run_args=None, rebuild=None):
        self.root, self.layout = root, layout
        self.rebuild = rebuild           # the install folder of --rebuild, or None
        self.cat = Catalog(ui_lang)
        self.model = Model()
        self.process = None
        self.closing = False
        self.run_args = run_args or {}   # extra CLI options for tests: temp, verify
        self.log_entries = []            # events and raw lines of the run, rendered by render_log
        self.bound = []                  # (widget, option, id, params) retranslated on a language switch
        # base, update, keys: shared by both forms; dest: the first install's folder; update_dest: the installed folder
        self.vars = {n: tk.StringVar() for n in ('base', 'update', 'keys', 'dest', 'update_dest')}
        self.lang_vars = {INSTALL: tk.StringVar(), UPDATE: tk.StringVar()}
        self.lang_codes = {INSTALL: default_game_language(ui_lang), UPDATE: default_game_language(ui_lang)}
        self.lang_boxes = {}
        self.lang_touched = {INSTALL: False, UPDATE: False}
        self.accept_version = False      # the next run carries on past the version check
        self.mode = INSTALL              # the form the last run came from
        self.files_wanted = False        # the update form shows the game files (the CLI asked for them)
        self.run_dest = ''               # the install folder of the last run
        self.screen = None
        self.fitted = None               # the (width, height) fit last gave the window
        self.scale = max(1.0, root.winfo_fpixels('1i') / 96.0)

        root.title(self.cat('ui.title'))
        icon = layout.runtime / 'astralchain.ico'
        png = layout.runtime / 'astralchain.png'   # Tk loads no .ico off Windows; the Linux release converts it
        if os.name == 'nt' and icon.is_file():
            root.iconbitmap(default=str(icon))
        elif png.is_file():
            self.icon = tk.PhotoImage(file=str(png))
            root.iconphoto(True, self.icon)
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        self.ui_font = 'Segoe UI' if os.name == 'nt' else 'DejaVu Sans'
        self.mono_font = 'Consolas' if os.name == 'nt' else 'DejaVu Sans Mono'
        style = ttk.Style(root)
        style.configure('Title.TLabel', font=(self.ui_font, 15, 'bold'))
        style.configure('Heading.TLabel', font=(self.ui_font, 12, 'bold'))
        style.configure('Section.TLabel', font=(self.ui_font, 10, 'bold'))
        style.configure('Notice.TLabel', background='#fff3cd', foreground='#664d03', padding=self.px(8))
        style.configure('Err.TLabel', foreground=BAD_COLOR)
        style.configure('Muted.TLabel', foreground=MUTED)
        style.configure('Version.TLabel', foreground='#808080')
        style.configure('Info.TLabel', background='#e7f0fb', foreground='#1c3d6e', padding=self.px(8))
        style.configure('Warn.TLabel', background='#fff3cd', foreground='#664d03', padding=self.px(8))
        style.configure('Big.TButton', padding=(self.px(14), self.px(5)))

        top = ttk.Frame(root, padding=(self.px(PAD), self.px(PAD), self.px(PAD), 0))
        top.pack(fill='x')
        self.title = self.label(top, 'ui.title', style='Title.TLabel')
        self.title.pack(side='left')
        version = ttk.Label(top, style='Version.TLabel')
        self.bind_text(version, 'text', 'ui.version', version=layout.release)
        version.pack(side='left', padx=(self.px(10), 0), pady=(self.px(6), 0))
        sw = ttk.Frame(top)
        sw.pack(side='right')
        self.ui_var = tk.StringVar(value=self.cat.lang)   # the window language code; the box shows its name
        labels = [UI_LANGUAGE_LABELS[c] for c in UI_LANGUAGES]
        self.ui_box = ttk.Combobox(sw, values=labels, state='readonly', width=max(map(len, labels)) + 1)
        self.ui_box.set(UI_LANGUAGE_LABELS[self.cat.lang])
        self.ui_box.bind('<<ComboboxSelected>>', self.pick_language)
        self.ui_box.pack(side='left')
        self.body = ttk.Frame(root, padding=self.px(PAD))
        self.body.pack(fill='both', expand=True)
        self.frames = {'welcome': self.build_welcome(), INSTALL: self.build_install(), UPDATE: self.build_update(),
                       'run': self.build_run(), 'finish': self.build_finish()}
        for mode in (INSTALL, UPDATE):
            self.lang_vars[mode].set(self.lang_label(self.lang_codes[mode]))
        for name in ('base', 'update', 'keys', 'dest'):
            self.vars[name].trace_add('write', lambda *_: self.refresh_install())
        for name in ('base', 'update', 'keys'):
            self.vars[name].trace_add('write', lambda *_: self.refresh_update())
        self.vars['update_dest'].trace_add('write', lambda *_: self.on_update_dest())
        self.refresh_install()
        self.refresh_update()
        if rebuild:
            self.bind_text(self.title, 'text', 'ui.rebuild_title')
            self.vars['update_dest'].set(str(rebuild))
            self.mode = UPDATE
            self.show(UPDATE)
            root.after(POLL_MS, self.start)
        else:
            self.show('welcome')

    def px(self, n):
        """A size in pixels at 96 DPI, scaled to the screen (Tk scales fonts by itself, not pixel sizes)."""
        return int(round(n * self.scale))

    # ---- text -----------------------------------------------------------------------------------------------------

    def bind_text(self, widget, option, id, **params):
        self.bound = [b for b in self.bound if not (b[0] is widget and b[1] == option)]
        self.bound.append((widget, option, id, params))
        widget.configure(**{option: self.cat(id, **params)})

    def label(self, parent, id, **kw):
        w = ttk.Label(parent, **kw)
        self.bind_text(w, 'text', id)
        return w

    def wrapped(self, parent, id=None, width=WRAP, **kw):
        w = ttk.Label(parent, wraplength=self.px(width), justify='left', **kw)
        if id:
            self.bind_text(w, 'text', id)
        return w

    def button(self, parent, id, command, **kw):
        w = ttk.Button(parent, command=command, **kw)
        self.bind_text(w, 'text', id)
        return w

    def retext(self):
        self.root.title(self.cat('ui.title'))
        for widget, option, id, params in self.bound:
            widget.configure(**{option: self.cat(id, **params)})
        self.fill_languages()
        self.refresh_install()
        self.refresh_update()
        self.refresh_run()
        self.refresh_finish()
        self.render_log()

    def pick_language(self, _event=None):
        self.ui_var.set(UI_LANGUAGES[self.ui_box.current()])
        self.switch_language()

    def switch_language(self):
        codes = {m: self.selected_lang(m) for m in (INSTALL, UPDATE)}
        self.cat.set_language(self.ui_var.get())
        self.ui_box.set(UI_LANGUAGE_LABELS[self.ui_var.get()])
        self.retext()
        for mode, code in codes.items():
            self.lang_vars[mode].set(self.lang_label(code))
        self.fit()

    def lang_label(self, code):
        return self.cat('lang.' + code)

    def fill_languages(self):
        for mode, box in self.lang_boxes.items():
            code = self.selected_lang(mode)
            box.configure(values=[self.lang_label(c) for c in ini.LANGUAGES])
            self.lang_vars[mode].set(self.lang_label(code))

    def selected_lang(self, mode):
        """The game language code behind a form's combobox text (kept when it is not one of the labels)."""
        shown = self.lang_vars[mode].get()
        for code in ini.LANGUAGES:
            for lang in UI_LANGUAGES:
                if shown == Catalog(lang)('lang.' + code):
                    return code
        return self.lang_codes[mode]

    # ---- screens --------------------------------------------------------------------------------------------------

    def show(self, name):
        for f in self.frames.values():
            f.pack_forget()
        self.frames[name].pack(fill='both', expand=True)
        self.screen = name
        self.fit()

    def fit(self):
        """Size the window to its largest screen (an unpacked frame still reports the size its children ask for), so
        a screen or language switch neither cuts text nor moves the buttons. The window only grows: opening the log
        makes it taller, and a size the user dragged it to stays. Every text change that can add lines calls it again,
        or pack takes the missing height from the button bars."""
        self.root.update_idletasks()
        sizes = [(f.winfo_reqwidth(), f.winfo_reqheight()) for f in self.frames.values()]
        chrome = self.root.winfo_reqheight() - self.body.winfo_reqheight()   # the title row above the body
        w = max(self.root.winfo_reqwidth(), max(s[0] for s in sizes) + 2 * self.px(PAD))
        h = chrome + max(s[1] for s in sizes) + 2 * self.px(PAD)
        if (w, h) == self.fitted:   # refresh_run calls it on every poll: leave the window alone while nothing grew
            return
        self.fitted = (w, h)
        self.root.minsize(w, h)
        self.root.geometry(f'{max(w, self.root.winfo_width())}x{max(h, self.root.winfo_height())}')

    def notice(self, parent):
        w = ttk.Label(parent, style='Notice.TLabel', wraplength=self.px(WRAP), justify='left')
        self.bind_text(w, 'text', 'ui.notice')
        return w

    def screen_frame(self, heading=None, back=True):
        """A form screen: its button bar (packed first, so pack shrinks the text above it and never the buttons), an
        optional heading and Back to the first screen. Returns (frame, bar)."""
        f = ttk.Frame(self.body)
        bar = ttk.Frame(f)
        bar.pack(side='bottom', fill='x', pady=(self.px(12), 0))
        if back and not self.rebuild:
            self.button(bar, 'ui.back', lambda: self.show('welcome')).pack(side='left')
        if heading:
            self.label(f, heading, style='Heading.TLabel').pack(anchor='w', pady=(0, self.px(6)))
        return f, bar

    def choice(self, parent, glyph, title_id, text_id, command):
        """A large clickable card: a glyph, a bold title and a line of explanation. Click, Enter or Space runs
        command; the mouse and the keyboard focus highlight it."""
        card = tk.Frame(parent, bg=CARD, highlightthickness=self.px(1), highlightbackground=CARD_BORDER,
                        highlightcolor=ACCENT, cursor='hand2', takefocus=1, padx=self.px(16), pady=self.px(12))
        # a fixed width, so the titles of both cards start at the same place whatever the glyph's width
        icon = tk.Label(card, text=glyph, bg=CARD, fg=ACCENT, font=(self.ui_font, 22), width=2, anchor='center')
        icon.grid(row=0, column=0, rowspan=2, sticky='n', padx=(0, self.px(14)))
        title = tk.Label(card, bg=CARD, fg='#1c1c1c', font=(self.ui_font, 12, 'bold'), anchor='w', justify='left')
        self.bind_text(title, 'text', title_id)
        title.grid(row=0, column=1, sticky='w')
        text = tk.Label(card, bg=CARD, fg=MUTED, font=(self.ui_font, 10), anchor='w', justify='left',
                        wraplength=self.px(WRAP - 90))
        self.bind_text(text, 'text', text_id)
        text.grid(row=1, column=1, sticky='w', pady=(self.px(3), 0))
        card.columnconfigure(1, weight=1)
        parts = (card, icon, title, text)

        def lit(on):
            for w in parts:
                w.configure(bg=CARD_HOVER if on else CARD)
            card.configure(highlightbackground=ACCENT if on else CARD_BORDER)

        for w in parts:
            w.bind('<Button-1>', lambda _e: command())
            w.bind('<Enter>', lambda _e: lit(True))
            w.bind('<Leave>', lambda _e: lit(card.focus_get() is card))
        card.bind('<FocusIn>', lambda _e: lit(True))
        card.bind('<FocusOut>', lambda _e: lit(False))
        for key in ('<Return>', '<space>'):
            card.bind(key, lambda _e: command())
        return card

    def build_welcome(self):
        f = ttk.Frame(self.body)
        self.wrapped(f, 'ui.intro').pack(anchor='w', pady=(0, self.px(12)))
        self.label(f, 'ui.welcome', style='Heading.TLabel').pack(anchor='w', pady=(0, self.px(8)))
        self.install_card = self.choice(f, '\u2193', 'ui.choice_install', 'ui.choice_install_text',
                                        lambda: self.open_form(INSTALL))
        self.install_card.pack(fill='x', pady=(0, self.px(10)))
        self.update_card = self.choice(f, '\u21bb', 'ui.choice_update', 'ui.choice_update_text',
                                       lambda: self.open_form(UPDATE))
        self.update_card.pack(fill='x')
        self.notice(f).pack(fill='x', pady=(self.px(14), 0))
        return f

    def file_rows(self, parent, start_row=0):
        """The base game, update and keys rows of a grid: label, entry, Browse and a mark that says whether the file
        exists, with a line under each that says what the file is and where it comes from. Returns {name: mark}."""
        marks = {}
        for i, (name, label_id, pick_id) in enumerate(FILES):
            row = start_row + 2 * i
            self.label(parent, label_id).grid(row=row, column=0, sticky='w', pady=(self.px(4), 0),
                                              padx=(0, self.px(10)))
            ttk.Entry(parent, textvariable=self.vars[name]).grid(row=row, column=1, sticky='ew', pady=(self.px(4), 0))
            self.button(parent, 'ui.browse', lambda n=name, p=pick_id: self.pick_file(n, p)).grid(
                row=row, column=2, padx=(self.px(8), 0), pady=(self.px(4), 0))
            marks[name] = ttk.Label(parent, width=2, anchor='center')
            marks[name].grid(row=row, column=3, padx=(self.px(6), 0), pady=(self.px(4), 0))
            self.wrapped(parent, f'ui.{name}_hint', width=WRAP - 180, style='Muted.TLabel').grid(
                row=row + 1, column=1, columnspan=3, sticky='w', pady=(self.px(1), self.px(2)))
        return marks

    def lang_row(self, parent, row, mode):
        self.label(parent, 'ui.game_language').grid(row=row, column=0, sticky='w', pady=self.px(3),
                                                    padx=(0, self.px(10)))
        box = ttk.Combobox(parent, textvariable=self.lang_vars[mode], state='readonly', width=30)
        box.grid(row=row, column=1, sticky='w', pady=self.px(3))
        box.bind('<<ComboboxSelected>>', lambda _e: self.lang_touched.__setitem__(mode, True))
        self.lang_boxes[mode] = box
        box.configure(values=[self.lang_label(c) for c in ini.LANGUAGES])

    def build_install(self):
        f, bar = self.screen_frame('ui.choice_install')
        self.install_button = self.button(bar, 'ui.do_install', self.start, style='Big.TButton', default='active')
        self.install_button.pack(side='right')
        self.wrapped(f, 'ui.install_intro').pack(anchor='w', pady=(0, self.px(10)))

        # One grid for both sections, so the entries of the files and of the folder line up.
        grid = ttk.Frame(f)
        grid.pack(fill='x')
        grid.columnconfigure(1, weight=1)
        self.label(grid, 'ui.section_files', style='Section.TLabel').grid(row=0, column=0, columnspan=4, sticky='w')
        self.install_marks = self.file_rows(grid, start_row=1)
        self.wrapped(grid, 'ui.keys_private', style='Muted.TLabel').grid(row=7, column=0, columnspan=4, sticky='w',
                                                                         pady=(self.px(4), 0))
        self.label(grid, 'ui.section_target', style='Section.TLabel').grid(row=8, column=0, columnspan=4, sticky='w',
                                                                           pady=(self.px(14), 0))
        self.label(grid, 'ui.dest').grid(row=9, column=0, sticky='w', pady=(self.px(4), 0), padx=(0, self.px(10)))
        ttk.Entry(grid, textvariable=self.vars['dest']).grid(row=9, column=1, sticky='ew', pady=(self.px(4), 0))
        self.button(grid, 'ui.browse', self.pick_dest).grid(row=9, column=2, padx=(self.px(8), 0), pady=(self.px(4), 0))
        self.dest_mark = ttk.Label(grid, width=2, anchor='center')
        self.dest_mark.grid(row=9, column=3, padx=(self.px(6), 0), pady=(self.px(4), 0))
        self.dest_hint = self.wrapped(grid, width=WRAP - 180, style='Muted.TLabel')
        self.dest_hint.grid(row=10, column=1, columnspan=3, sticky='w', pady=(self.px(1), self.px(2)))
        self.lang_row(grid, 11, INSTALL)

        # A problem the player can fix in one click (the folder holds an installation: update it) gets a button.
        self.install_error_row = ttk.Frame(f)
        self.install_error_row.pack(fill='x', pady=(self.px(8), 0))
        self.install_error = self.wrapped(self.install_error_row, width=WRAP - 160, style='Err.TLabel')
        self.install_error.pack(side='left', anchor='w')
        self.to_update_button = self.button(self.install_error_row, 'ui.to_update', self.install_to_update)
        return f

    def build_update(self):
        f, bar = self.screen_frame('ui.choice_update')
        self.update_button = self.button(bar, 'ui.do_update', self.start, style='Big.TButton', default='active')
        self.update_button.pack(side='right')
        self.wrapped(f, 'ui.update_intro').pack(anchor='w', pady=(0, self.px(10)))
        grid = ttk.Frame(f)
        grid.pack(fill='x')
        grid.columnconfigure(1, weight=1)
        self.label(grid, 'ui.install_folder').grid(row=0, column=0, sticky='w', pady=self.px(3), padx=(0, self.px(10)))
        ttk.Entry(grid, textvariable=self.vars['update_dest']).grid(row=0, column=1, sticky='ew', pady=self.px(3))
        self.button(grid, 'ui.browse', self.pick_update_dest).grid(row=0, column=2, padx=(self.px(8), 0))
        self.update_dest_mark = ttk.Label(grid, width=2, anchor='center')
        self.update_dest_mark.grid(row=0, column=3, padx=(self.px(6), 0))

        # What the folder holds: an installation (its version, release and state), nothing, or something else.
        self.found_box = ttk.Label(f, style='Info.TLabel', wraplength=self.px(WRAP), justify='left')
        self.found_box.pack(fill='x', pady=(self.px(8), 0))

        self.update_lang_frame = ttk.Frame(f)
        self.update_lang_frame.pack(fill='x', pady=(self.px(8), 0))
        self.update_lang_frame.columnconfigure(1, weight=1)
        self.lang_row(self.update_lang_frame, 0, UPDATE)

        # The game files, shown only when the installation must extract the game again.
        self.update_files = ttk.Frame(f)
        self.wrapped(self.update_files, 'ui.files_needed', style='Warn.TLabel').pack(fill='x', pady=(0, self.px(6)))
        grid = ttk.Frame(self.update_files)
        grid.pack(fill='x')
        grid.columnconfigure(1, weight=1)
        self.update_marks = self.file_rows(grid)

        self.update_error_row = ttk.Frame(f)
        self.update_error_row.pack(side='bottom', fill='x', pady=(self.px(8), 0))
        self.update_error = self.wrapped(self.update_error_row, width=WRAP - 160, style='Err.TLabel')
        self.update_error.pack(side='left', anchor='w')
        self.to_install_button = self.button(self.update_error_row, 'ui.to_install', self.update_to_install)
        return f

    def build_run(self):
        f = ttk.Frame(self.body)
        bar = ttk.Frame(f)
        bar.pack(side='bottom', fill='x', pady=(self.px(8), 0))
        self.cancel_button = self.button(bar, 'ui.cancel', self.cancel)
        self.cancel_button.pack(side='right')
        self.log_toggle = self.button(bar, 'ui.show_log', self.toggle_log)
        self.log_toggle.pack(side='left')
        self.run_header = ttk.Label(f, style='Heading.TLabel')
        self.run_header.pack(anchor='w')
        self.run_hint = self.wrapped(f, 'ui.run_hint', style='Muted.TLabel')
        self.run_hint.pack(anchor='w', pady=(self.px(2), 0))
        self.step_rows = {}
        self.steps_box = ttk.Frame(f)
        self.steps_box.pack(fill='x', pady=(self.px(8), self.px(8)))
        self.phase_label = self.wrapped(f)
        self.phase_label.pack(anchor='w')
        self.bar = ttk.Progressbar(f, mode='determinate', maximum=1000)
        self.bar.pack(fill='x', pady=(self.px(4), self.px(6)))
        self.space_label = self.wrapped(f, style='Muted.TLabel')
        self.space_label.pack(anchor='w')
        self.warn_label = ttk.Label(f, wraplength=self.px(WRAP), justify='left', foreground='#9a6700')
        self.warn_label.pack(anchor='w')
        self.log_frame = ttk.Frame(f)
        self.log = tk.Text(self.log_frame, height=10, wrap='word', state='disabled', font=(self.mono_font, 9))
        scroll = ttk.Scrollbar(self.log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.log.pack(side='left', fill='both', expand=True)
        self.log_open = False
        return f

    def build_finish(self):
        f = ttk.Frame(self.body)
        bar = ttk.Frame(f)
        bar.pack(side='bottom', fill='x', pady=(self.px(12), 0))
        self.close_button = self.button(bar, 'ui.close', self.on_close)
        self.close_button.pack(side='right')
        self.open_button = self.button(bar, 'ui.open_folder', self.open_folder)
        self.open_button.pack(side='right', padx=(0, self.px(8)))
        # Shown by refresh_finish when they apply: the game after a success (or a failed rebuild, with the code it
        # has), the log after a failure, the same run past the version check, and the update form with its game files
        # after the CLI asked for them.
        self.play_button = self.button(bar, 'ui.play', self.start_game, style='Big.TButton', default='active')
        self.log_button = self.button(bar, 'ui.open_log', self.open_log)
        self.retry_button = self.button(bar, 'ui.continue_anyway', self.continue_anyway)
        self.files_button = self.button(bar, 'ui.choose_files', self.choose_files)
        self.back_button = self.button(bar, 'ui.back', lambda: self.show(self.mode))
        self.back_button.pack(side='left')
        self.finish_title = ttk.Label(f, style='Title.TLabel', wraplength=self.px(WRAP), justify='left')
        self.finish_title.pack(anchor='w')
        self.finish_text = self.wrapped(f)
        self.finish_text.pack(anchor='w', pady=(self.px(8), self.px(8)))
        self.finish_steps = ttk.Frame(f)
        self.finish_steps.pack(fill='x', pady=(0, self.px(8)))
        self.finish_notice = self.notice(f)
        self.finish_notice.pack(fill='x', pady=(self.px(8), 0))
        return f

    # ---- forms ----------------------------------------------------------------------------------------------------

    def open_form(self, mode):
        self.mode = mode
        if mode == INSTALL and not self.vars['keys'].get().strip():
            found = find_keys([self.vars['base'].get(), self.vars['update'].get()])
            if found:
                self.vars['keys'].set(found)
        self.show(mode)

    def form(self, mode=None):
        mode = mode or self.mode
        if mode == UPDATE:
            files = self.update_files_shown()
            return Form(base=self.vars['base'].get() if files else '', update=self.vars['update'].get() if files else '',
                        keys=self.vars['keys'].get() if files else '', lang=self.selected_lang(UPDATE),
                        dest=self.vars['update_dest'].get(), mode=UPDATE)
        return Form(base=self.vars['base'].get(), update=self.vars['update'].get(), keys=self.vars['keys'].get(),
                    lang=self.selected_lang(INSTALL), dest=self.vars['dest'].get(), mode=INSTALL)

    def update_files_shown(self):
        folder = inspect_folder(self.vars['update_dest'].get())
        return folder.kind == 'install' and (folder.needs_game_files or self.files_wanted)

    @staticmethod
    def mark(widget, value, ok):
        """✔ when ok, ✖ when a value was entered but is not ok, nothing for an empty entry."""
        if not value.strip():
            widget.configure(text='')
        else:
            widget.configure(text='\u2714' if ok else '\u2716', foreground=OK_COLOR if ok else BAD_COLOR)

    def refresh_marks(self, marks):
        for name, w in marks.items():
            value = self.vars[name].get()
            self.mark(w, value, bool(value.strip()) and Path(value.strip()).is_file())

    def refresh_install(self):
        if not hasattr(self, 'install_marks'):
            return
        self.refresh_marks(self.install_marks)
        folder = inspect_folder(self.vars['dest'].get())
        ok = folder.kind == 'new' or (folder.kind == 'install' and not folder.finished)
        self.mark(self.dest_mark, self.vars['dest'].get(), ok)
        if folder.kind == 'install' and not folder.finished:
            self.bind_text(self.dest_hint, 'text', 'ui.dest_resume')
            self.dest_hint.grid()
        elif folder.kind == 'new' and not Path(folder.path).exists():
            self.bind_text(self.dest_hint, 'text', 'ui.dest_created')
            self.dest_hint.grid()
        else:   # no line at all, rather than an empty one that leaves a gap above the language
            self.bound = [b for b in self.bound if b[0] is not self.dest_hint]
            self.dest_hint.grid_remove()
        self.show_problems(INSTALL, only_folder=True)

    def on_update_dest(self):
        """The installed folder changed: take its recorded game language unless the player chose one here."""
        folder = inspect_folder(self.vars['update_dest'].get())
        if folder.kind == 'install' and folder.lang in ini.LANGUAGES and not self.lang_touched[UPDATE]:
            self.lang_vars[UPDATE].set(self.lang_label(folder.lang))
        self.files_wanted = False
        self.refresh_update()

    def refresh_update(self):
        if not hasattr(self, 'update_marks'):
            return
        folder = inspect_folder(self.vars['update_dest'].get())
        self.mark(self.update_dest_mark, folder.path, folder.kind == 'install')
        if folder.kind == 'install':
            version = folder.version or '1.0.0'
            lines = [('ui.install_found', {'version': version, 'release': folder.release or '?'})]
            if folder.version not in SUPPORTED:
                lines.append(('ui.install_old', {}))
            elif not folder.finished:
                lines.append(('ui.install_unfinished', {}))
            elif folder.release == self.layout.release:
                lines.append(('ui.install_same_release', {}))
            else:
                lines.append(('ui.install_kept', {}))
            self.found_box.configure(text='\n'.join(self.cat(i, **p) for i, p in lines))
            if not self.found_box.winfo_manager():
                self.found_box.pack(fill='x', pady=(self.px(8), 0), after=self.found_anchor())
            if not self.update_lang_frame.winfo_manager():
                self.update_lang_frame.pack(fill='x', pady=(self.px(8), 0), after=self.found_box)
        else:
            self.found_box.pack_forget()
            self.update_lang_frame.pack_forget()
        if self.update_files_shown():
            if not self.update_files.winfo_manager():
                self.update_files.pack(fill='x', pady=(self.px(10), 0), after=self.update_lang_frame)
        else:
            self.update_files.pack_forget()
        self.refresh_marks(self.update_marks)
        self.show_problems(UPDATE, only_folder=True)

    def found_anchor(self):
        """The widget the found box follows: the folder row's grid (the frame packed after the intro)."""
        return self.update_dest_mark.master

    def show_problems(self, mode, only_folder=False, problems=None):
        """The problems of a form under it. only_folder (while the player types) shows those about the folder alone,
        so an empty field is not an error before Install or Update is pressed."""
        if problems is None:
            problems = self.form(mode).problems()
        folder_ids = {'ui.dest_is_file', 'ui.dest_not_empty', 'ui.dest_has_install', 'ui.install_unreadable',
                      'ui.no_install'}
        if only_folder:
            problems = [p for p in problems if p in folder_ids]
        label = self.install_error if mode == INSTALL else self.update_error
        button = self.to_update_button if mode == INSTALL else self.to_install_button
        dest = self.vars['dest' if mode == INSTALL else 'update_dest'].get()
        params = {}
        folder = inspect_folder(dest)
        if folder.kind == 'broken':
            params = {'path': str(Path(folder.path) / statefile.NAME)}
        label.configure(text=' '.join(self.cat(p, **params) for p in problems))
        offer = ('ui.dest_has_install' in problems) if mode == INSTALL else (
            'ui.no_install' in problems and folder.kind in ('new',))
        if offer and not button.winfo_manager():
            button.pack(side='right', anchor='n')
        elif not offer and button.winfo_manager():
            button.pack_forget()
        self.fit()

    def install_to_update(self):
        self.vars['update_dest'].set(self.vars['dest'].get())
        self.open_form(UPDATE)

    def update_to_install(self):
        self.vars['dest'].set(self.vars['update_dest'].get())
        self.open_form(INSTALL)

    def pick_file(self, name, pick_id):
        types = [(self.cat('ui.type_game'), '*.nsp *.xci'), (self.cat('ui.type_all'), '*.*')]
        if name == 'update':
            types = [(self.cat('ui.type_update'), '*.nsp'), (self.cat('ui.type_all'), '*.*')]
        if name == 'keys':
            types = [(self.cat('ui.type_keys'), '*.keys'), (self.cat('ui.type_all'), '*.*')]
        start = self.vars[name].get().strip() or self.vars['base'].get().strip() or self.vars['update'].get().strip()
        kw = {'initialdir': str(Path(start).parent)} if start and Path(start).parent.is_dir() else {}
        path = filedialog.askopenfilename(title=self.cat(pick_id), filetypes=types, **kw)
        if path:
            self.set_file(name, os.path.normpath(path))

    def set_file(self, name, path):
        """A chosen file, and the ones next to it that fill the other empty fields: the update beside the base game
        (and the other way round) and a prod.keys beside either."""
        self.vars[name].set(path)
        if name in ('base', 'update'):
            for other, found in sibling_game_files(path).items():
                if other != name and found and not self.vars[other].get().strip():
                    self.vars[other].set(os.path.normpath(found))
            if not self.vars['keys'].get().strip():
                found = find_keys([path])
                if found:
                    self.vars['keys'].set(os.path.normpath(found))

    def pick_dest(self):
        path = filedialog.askdirectory(title=self.cat('ui.pick_dest'), mustexist=False)
        if path:
            self.vars['dest'].set(os.path.normpath(install_subfolder(os.path.normpath(path))))

    def pick_update_dest(self):
        path = filedialog.askdirectory(title=self.cat('ui.pick_install_folder'), mustexist=True)
        if path:
            self.vars['update_dest'].set(os.path.normpath(path))

    # ---- run ------------------------------------------------------------------------------------------------------

    def start(self, accept=False):
        if self.screen in (INSTALL, UPDATE):
            self.mode = self.screen
        self.accept_version = accept
        form = self.form()
        problems = form.problems()
        if problems:
            self.show_problems(self.mode, problems=problems)
            return
        self.show_problems(self.mode, problems=[])
        # in a repository the Python running this window: the tools' embeddable Python there has no ._pth entry for
        # the repository, so `-m installer` would not find the package
        python = Path(sys.executable) if self.layout.dev else self.layout.python
        command = form.command(python, root=self.layout.root if self.layout.dev else None,
                               temp=self.run_args.get('temp'), verify_hashes=self.run_args.get('verify', False),
                               accept_version=accept)
        self.run_dest = form.dest.strip()
        self.begin(command, self.layout.root)

    def begin(self, command, cwd):
        self.model = Model()
        self.model.dest = self.run_dest
        self.clear_log()
        self.build_step_rows(self.model.steps)
        self.cancel_button.configure(state='normal')
        self.bind_text(self.cancel_button, 'text', 'ui.cancel')
        self.show('run')
        self.refresh_run()
        self.process = Process(command, cwd=cwd, helper_python=self.layout.python)
        self.root.after(POLL_MS, self.poll)

    def build_step_rows(self, steps):
        for w in self.steps_box.winfo_children():
            w.destroy()
        self.step_rows = {}
        for s in steps:
            row = ttk.Frame(self.steps_box)
            row.pack(fill='x', pady=1)
            glyph = ttk.Label(row, width=3)
            glyph.pack(side='left')
            name = ttk.Label(row)
            name.pack(side='left')
            self.step_rows[s] = (glyph, name)

    def poll(self):
        if self.process is None:
            return
        try:
            while True:
                line = self.process.lines.get_nowait()
                if line is not None:
                    self.feed(line)
        except Exception:
            pass
        code = self.process.poll()
        self.refresh_run()
        if code is None:
            self.root.after(POLL_MS, self.poll)
            return
        self.model.finish(code)
        self.process = None
        if self.rebuild and self.model.result == 'ok' and not self.closing:
            self.start_game()
            return
        self.refresh_run()
        self.refresh_finish()
        self.show('finish')
        if self.closing:
            self.root.destroy()

    def feed(self, line):
        """One stdout line of the CLI: an event updates the model and the log, anything else goes to the log as is."""
        event = parse_line(line)
        if event is None:
            if line.strip():
                self.log_entries.append(line.rstrip())
                self.append_log(line.rstrip())
            return
        before = list(self.model.steps)
        self.model.apply(event)
        if self.model.steps != before:
            self.build_step_rows(self.model.steps)
        self.log_entries.append(event)
        self.append_log(describe_event(self.cat, event))

    def refresh_run(self):
        m = self.model
        for s, (glyph, name) in self.step_rows.items():
            state = m.states.get(s, PENDING)
            glyph.configure(text=GLYPH[state], foreground=COLOR[state])
            name.configure(text=self.cat('step.' + s) + (f'  ({self.cat("ui.state_" + state)})'
                                                          if state in (SKIPPED, FAILED) else ''),
                           font=(self.ui_font, 10, 'bold') if state == RUNNING else (self.ui_font, 10))
        mode = m.mode or ('update' if self.mode == UPDATE else 'install')
        self.run_header.configure(text=self.cat('ui.running_' + mode if mode in ('install', 'resume', 'update')
                                                else 'ui.running_install'))
        if m.current and m.phase:
            text = self.cat(f'phase.{m.current}.{m.phase}')
            if m.total:
                text += '  ' + amount(self.cat, m.done, m.total, m.unit)
        elif m.current:
            text = self.cat('step.' + m.current)
        else:
            text = ''
        self.phase_label.configure(text=text)
        frac = m.fraction()
        if frac is None:
            if str(self.bar.cget('mode')) != 'indeterminate':
                self.bar.configure(mode='indeterminate')
                self.bar.start(15)
        else:
            if str(self.bar.cget('mode')) == 'indeterminate':
                self.bar.stop()
                self.bar.configure(mode='determinate')
            self.bar.configure(value=frac * 1000)
        self.space_label.configure(text=m.need_text(self.cat))
        self.warn_label.configure(text='\n'.join(self.cat(i, **p) for i, p in m.warnings))
        self.fit()   # new step rows and warnings make the screen taller

    def toggle_log(self):
        self.log_open = not self.log_open
        if self.log_open:
            self.log_frame.pack(fill='both', expand=True, pady=(self.px(8), 0))
        else:
            self.log_frame.pack_forget()
        self.bind_text(self.log_toggle, 'text', 'ui.hide_log' if self.log_open else 'ui.show_log')
        self.fit()

    def clear_log(self):
        self.log_entries = []
        self.render_log()

    def render_log(self):
        """The whole log from the kept events in the current language (raw CLI lines stay as they are)."""
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        for entry in self.log_entries:
            text = entry if isinstance(entry, str) else describe_event(self.cat, entry)
            if text:
                self.log.insert('end', text + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def append_log(self, text):
        if text:
            self.log.configure(state='normal')
            self.log.insert('end', text + '\n')
            self.log.see('end')
            self.log.configure(state='disabled')

    def cancel(self):
        if self.process:
            self.cancel_button.configure(state='disabled')
            self.bind_text(self.cancel_button, 'text', 'ui.cancelling')
            self.process.cancel()

    def on_close(self):
        if self.process:
            if not messagebox.askyesno(self.cat('ui.title'), self.cat('ui.confirm_cancel')):
                return
            self.closing = True
            self.cancel()
            return
        self.root.destroy()

    # ---- finish ---------------------------------------------------------------------------------------------------

    def game_exe(self):
        dest = Path(self.model.dest or self.run_dest or '.')
        return dest / ('astralchain.exe' if os.name == 'nt' else 'astralchain')

    def refresh_finish(self):
        m = self.model
        result = m.result or 'ok'
        updated = m.mode == 'update' or (self.mode == UPDATE and m.mode != 'install')
        title = 'ui.result_updated' if result == 'ok' and updated else 'ui.result_' + result
        self.finish_title.configure(text=self.cat(title))
        error = None
        if result == 'ok':
            text = self.cat('ui.updated_text' if updated else 'ui.done_text', dest=m.dest)
        elif result == 'cancelled':
            text = self.cat('ui.cancelled_text')
        else:
            error = m.error or ('ui.process_died', {'code': m.exit_code}, None)
            text = self.cat(error[0], **error[1])
            if error[0] in ACCEPT_ANYWAY:
                text += '\n\n' + self.cat('ui.continue_anyway_help')
            if m.log_path:
                text += '\n\n' + self.cat('ui.log_file', path=m.log_path)
        self.finish_text.configure(text=text)
        for w in self.finish_steps.winfo_children():
            w.destroy()
        if result != 'ok':   # a success needs no step list: the title says it all
            for s in m.steps:
                state = m.states.get(s, PENDING)
                ttk.Label(self.finish_steps, text=f'{GLYPH[state]}  {self.cat("step." + s)}',
                          foreground=COLOR[state]).pack(anchor='w')
        self.open_button.state(['!disabled' if Path(m.dest or '.').is_dir() and m.dest else 'disabled'])
        wanted = {
            self.play_button: (result == 'ok' or bool(self.rebuild)) and self.game_exe().is_file(),
            self.log_button: result == 'failed' and bool(m.log_path) and Path(m.log_path).is_file(),
            self.retry_button: error is not None and error[0] in ACCEPT_ANYWAY,
            self.files_button: error is not None and error[0] == 'check.inputs_needed' and self.mode == UPDATE,
        }
        for w in (self.play_button, self.log_button, self.retry_button, self.files_button):
            w.pack_forget()
        # Play, the main action, goes rightmost (packed before Close); the others go left of Open folder.
        if wanted[self.play_button]:
            self.play_button.pack(side='right', padx=(self.px(8), 0), before=self.close_button)
        for w in (self.files_button, self.retry_button, self.log_button):   # right to left
            if wanted[w]:
                w.pack(side='right', padx=(0, self.px(8)), before=self.open_button)
        if result == 'ok' and not self.rebuild:
            self.back_button.pack_forget()
        elif not self.back_button.winfo_manager():
            self.back_button.pack(side='left')
        if wanted[self.play_button]:
            self.play_button.focus_set()
        self.fit()

    def continue_anyway(self):
        """The version check failed: run the setup again, past it, with the warning the CLI then prints."""
        self.show(self.mode)
        self.start(accept=True)

    def choose_files(self):
        """The CLI must extract the game again (the kept game data failed its check): the update form, with the game
        files to fill in."""
        self.files_wanted = True
        self.refresh_update()
        self.open_form(UPDATE)

    def start_game(self):
        """Starts the game of the install folder and closes the window."""
        exe = self.game_exe()
        flags = subprocess.DETACHED_PROCESS if os.name == 'nt' else 0
        if exe.is_file():
            subprocess.Popen([str(exe)], cwd=str(exe.parent), creationflags=flags, close_fds=True)
        self.root.destroy()

    @staticmethod
    def open_path(path):
        if os.name == 'nt':
            os.startfile(path)
        else:
            subprocess.Popen(['xdg-open', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)

    def open_folder(self):
        dest = self.model.dest
        if dest and Path(dest).is_dir():
            self.open_path(dest)

    def open_log(self):
        if self.model.log_path and Path(self.model.log_path).is_file():
            self.open_path(self.model.log_path)


# ---- self test ---------------------------------------------------------------------------------------------------

SAMPLE_START = {'type': 'start', 'release': 'selftest', 'mode': 'install', 'dest': 'Z:/selftest/Astral Chain', 'log': 'D:\\log.txt',
                'steps': ['check', 'extract', 'lift', 'compile', 'prepare', 'cleanup']}
SAMPLE_RUN = [
    SAMPLE_START,
    {'type': 'plan', 'run': ['check', 'extract'], 'skip': {}},
    {'type': 'step', 'step': 'check', 'state': 'started'},
    {'type': 'progress', 'step': 'check', 'phase': 'inputs'},
    {'type': 'space', 'dest': 'Z:/selftest/Astral Chain', 'need': 36_400_000_000, 'free': 210_000_000_000, 'temp': 'C:\\t',
     'temp_need': 0, 'temp_free': 0, 'same_disk': True},
    {'type': 'step', 'step': 'check', 'state': 'done', 'seconds': 12.0},
    {'type': 'step', 'step': 'extract', 'state': 'started'},
    {'type': 'progress', 'step': 'extract', 'phase': 'copy', 'done': 6_000_000_000, 'total': 16_000_000_000,
     'unit': 'bytes'},
]


SAMPLE_OK = [SAMPLE_START, {'type': 'plan', 'run': ['check', 'extract'], 'skip': {'lift': 'current',
                                                                       'compile': 'current',
                                                                       'prepare': 'current',
                                                                       'cleanup': 'not_needed'}},
             {'type': 'step', 'step': 'check', 'state': 'started'},
             {'type': 'step', 'step': 'check', 'state': 'done', 'seconds': 1.0},
             {'type': 'step', 'step': 'extract', 'state': 'started'},
             {'type': 'step', 'step': 'extract', 'state': 'done', 'seconds': 1.0},
             {'type': 'step', 'step': 'lift', 'state': 'skipped', 'reason': 'current'},
             {'type': 'step', 'step': 'compile', 'state': 'skipped', 'reason': 'current'},
             {'type': 'step', 'step': 'prepare', 'state': 'skipped', 'reason': 'current'},
             {'type': 'step', 'step': 'cleanup', 'state': 'done', 'seconds': 0.1},
             {'type': 'end', 'result': 'ok', 'seconds': 3}]


def grab(root, path):
    if ImageGrab is None:
        raise SystemExit('--shots needs Pillow')
    root.attributes('-topmost', True)
    root.update()
    x, y, w, h = root.winfo_rootx(), root.winfo_rooty(), root.winfo_width(), root.winfo_height()
    ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True).save(path)


def selftest(layout, shots=None):
    """Every screen in every language; the window closes itself. Returns 0 when nothing raised."""
    dpi_aware()
    fix_tcl_paths()
    root = tk.Tk()
    root.geometry('+60+40')
    app = App(root, 'en', layout)
    checks = []
    mixed = []
    work = tempfile.TemporaryDirectory(prefix='ac-gui-selftest-')
    installed, old = Path(work.name) / 'Astral Chain', Path(work.name) / 'Astral Chain 1.0.0'
    built = {s: {'key': 'k'} for s in ('extract', 'lift', 'compile', 'prepare')}
    for folder, version in ((installed, '1.0.1'), (old, '1.0.0')):
        folder.mkdir()
        statefile.save(folder, statefile.empty('1.13.0') | {'version': version, 'lang': 'fr', 'steps': dict(built)})
        (folder / ('astralchain.exe' if os.name == 'nt' else 'astralchain')).write_bytes(b'')   # Play is offered

    def shot(name):
        root.update()
        if shots:
            Path(shots).mkdir(parents=True, exist_ok=True)
            grab(root, Path(shots) / f'{name}.png')

    def run_all():
        for lang in UI_LANGUAGES:
            app.ui_var.set(lang)
            app.switch_language()
            tag = lang
            app.show('welcome')
            shot(f'welcome-{tag}')
            app.open_form(INSTALL)
            app.vars['dest'].set('Z:/selftest/Astral Chain')
            app.vars['base'].set('Z:/selftest/astralchain.xci')
            app.vars['update'].set('Z:/selftest/astralchain-1.0.1.nsp')
            app.vars['keys'].set('Z:/selftest/prod.keys')
            app.start()   # missing files: shows the problems
            checks.append(bool(app.install_error.cget('text')))
            shot(f'install-{tag}')
            app.vars['dest'].set(str(installed))   # a finished installation: Update offered
            checks.append(app.to_update_button.winfo_manager())
            shot(f'install-has-install-{tag}')
            app.install_to_update()
            checks.append(app.screen == UPDATE and app.found_box.winfo_manager()
                          and not app.update_files.winfo_manager())
            shot(f'update-{tag}')
            app.vars['update_dest'].set(str(old))   # made before update support: the game files again
            checks.append(app.update_files.winfo_manager())
            shot(f'update-old-{tag}')
            app.vars['update_dest'].set(str(Path(work.name) / 'nothing here'))
            checks.append(app.to_install_button.winfo_manager())
            shot(f'update-none-{tag}')
            app.vars['update_dest'].set('')
            app.model = Model()
            app.clear_log()
            app.build_step_rows(app.model.steps)
            app.show('run')
            for e in SAMPLE_RUN:
                app.feed(json.dumps(e))
            app.log_open = False
            app.toggle_log()
            app.refresh_run()
            shot(f'run-{tag}')
            if lang == 'en':   # the log follows the window language: no English line may remain after the switch
                app.ui_var.set('ru')
                app.switch_language()
                text = app.log.get('1.0', 'end')
                mixed.append('Checking' in text or 'Extracting' not in text and 'Распаковка' not in text)
                app.ui_var.set('en')
                app.switch_language()
            for mode, name in ((INSTALL, 'finish-ok'), (UPDATE, 'finish-updated')):
                app.mode = mode
                app.model = Model()
                for e in SAMPLE_OK:
                    app.model.apply(dict(e, mode='update') if mode == UPDATE and e['type'] == 'start' else e)
                app.model.dest = str(installed)
                app.model.finish(0)
                app.refresh_finish()
                app.show('finish')
                shot(f'{name}-{tag}')
            app.mode = INSTALL
            app.model = Model()
            for e in SAMPLE_RUN[:3] + [{'type': 'error', 'id': 'check.disk_space', 'step': 'check',
                                        'params': {'path': 'Z:/selftest/Astral Chain', 'need_gb': 36.4, 'free_gb': 20.1}},
                                       {'type': 'end', 'result': 'failed', 'seconds': 3}]:
                app.model.apply(e)
            app.model.finish(1)
            app.refresh_finish()
            shot(f'finish-failed-{tag}')
            app.model = Model()
            app.model.finish(3)
            app.refresh_finish()
            shot(f'finish-cancelled-{tag}')
            app.toggle_log()
        root.destroy()

    root.after(200, run_all)
    root.after(90000, root.destroy)   # never leaves a window open
    root.mainloop()
    work.cleanup()
    names = ('install problems shown', 'update offered for an installed folder', 'installation found',
             'game files asked for an old installation', 'install offered for an empty folder')
    failed = [f'{names[i % len(names)]} ({UI_LANGUAGES[i // len(names)]})' for i, ok in enumerate(checks) if not ok]
    if len(checks) != len(names) * len(UI_LANGUAGES):
        failed.append(f'{len(checks)} checks ran, {len(names) * len(UI_LANGUAGES)} expected')
    if mixed != [False]:
        failed.append('the log kept English lines after a language switch')
    for f in failed:
        print(f'selftest: FAILED: {f}', file=sys.stderr)
    return 1 if failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--shots', type=Path)
    ap.add_argument('--lang', choices=UI_LANGUAGES)
    ap.add_argument('--root', type=Path)
    ap.add_argument('--rebuild', type=Path)
    args = ap.parse_args(argv)
    layout = layouts.find(args.root)
    if args.selftest:
        return selftest(layout, args.shots)
    dpi_aware()
    fix_tcl_paths()
    root = tk.Tk()
    App(root, args.lang or windows_ui_language(), layout, rebuild=args.rebuild)
    root.mainloop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
