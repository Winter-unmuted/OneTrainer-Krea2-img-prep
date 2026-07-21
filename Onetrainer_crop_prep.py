#!/usr/bin/env python3
"""
Krea2 Crop Tool
---------------
Preprocess images for Krea2 training with OneTrainer.

- Prompts for a list of target resolutions (e.g. "512, 768, 1024") and a folder.
- Generates the exact aspect-ratio bucket list OneTrainer's AspectBucketing
  produces at quantization 64 (union across all targets).
- Lets you place pixel-exact crop boxes (no resampling at crop time) on each
  image, with optional per-image downscaling (session-only clones).
- Hard rule: no upscaling anywhere.
- Exports JPEG quality 100, 4:4:4 chroma subsampling, with the original
  filename + scale info in the JPEG comment and EXIF ImageDescription.
- Output: <folder>/crops/<target>/NNNN.jpg  (one folder per target-resolution
  family, e.g. 512 / 768 / 1024, each holding crops of every aspect bucket in
  that family at their exact pixel sizes. Each folder = one OneTrainer concept
  with resolution set to that single target value and aspect bucketing on, so
  training does not resize. Serial counters are per target folder, resume.)
- Session state (boxes, clones) saved to <folder>/krea2_crop_session.json.

Hotkeys:
  PgDn / PgUp ......... next / previous image
  Mouse wheel ......... cycle crop size (over the image)
  Left-click .......... place crop box at cursor
  Right-click ......... select / deselect a placed box under cursor
  Arrow keys .......... nudge selected box 1 px  (Shift = 16 px)
  Delete / Backspace .. delete selected box
  C  or  Ctrl+E ....... export (crop) current image's boxes
  Ctrl+D .............. clone current image for downscaling
  Escape .............. deselect box
"""

import json
import math
import os
import re
import sys
import colorsys
import tkinter as tk

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
from tkinter import filedialog, messagebox, simpledialog, ttk

from PIL import Image, ImageOps, ImageTk

Image.MAX_IMAGE_PIXELS = None  # allow big images

# ----------------------------------------------------------------------------
# Bucket generation (mirrors mgds AspectBucketing at quantization 64)
# ----------------------------------------------------------------------------

QUANT = 64

# (h, w) relative aspects, as in mgds/pipelineModules/AspectBucketing.py
ALL_POSSIBLE_INPUT_ASPECTS = [
    (1.0, 1.0), (1.0, 1.25), (1.0, 1.5), (1.0, 1.75), (1.0, 2.0),
    (1.0, 2.5), (1.0, 3.0), (1.0, 3.5), (1.0, 4.0),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

RESAMPLE_METHODS = {
    "Lanczos": Image.Resampling.LANCZOS,
    "Bicubic": Image.Resampling.BICUBIC,
    "Area (box)": Image.Resampling.BOX,
    "Bilinear": Image.Resampling.BILINEAR,
    "Hamming": Image.Resampling.HAMMING,
}

SESSION_FILENAME = "krea2_crop_session.json"
OUTPUT_DIRNAME = "crops"


def generate_buckets(targets: list[int], quant: int = QUANT
                     ) -> list[tuple[int, int, int]]:
    """Return sorted unique list of (w, h, target) bucket resolutions, exactly
    as OneTrainer/mgds computes them (round each dim to nearest multiple of
    quant). `target` is the resolution bucket that produced this (w, h). If the
    same (w, h) is produced by more than one target, the smallest target wins
    (deterministic; overlaps do not occur for typical target lists)."""
    owner: dict[tuple[int, int], int] = {}
    for t in sorted(targets):
        new = [(h / math.sqrt(h * w) * t, w / math.sqrt(h * w) * t)
               for (h, w) in ALL_POSSIBLE_INPUT_ASPECTS]
        new = new + [(w, h) for (h, w) in new]
        for (h, w) in new:
            qh = round(h / quant) * quant
            qw = round(w / quant) * quant
            if qh > 0 and qw > 0:
                owner.setdefault((qw, qh), t)   # smallest target wins
    return sorted(((w, h, t) for (w, h), t in owner.items()),
                  key=lambda r: (r[0] * r[1], r[0] / r[1]))


def nice_ratio(w: int, h: int) -> str:
    g = math.gcd(w, h)
    rw, rh = w // g, h // g
    if rw <= 32 and rh <= 32:
        return f"{rw}:{rh}"
    return f"{w / h:.2f}:1" if w >= h else f"1:{h / w:.2f}"


# Sort modes for the crop-size list. Each bucket is (w, h, target).
SORT_BY_BUCKET = "By bucket, portrait\u2192landscape"
SORT_BY_ASPECT = "By aspect (portrait\u2192landscape)"
SORT_BY_DIMS = "By max X, then max Y"
SORT_MODES = [SORT_BY_BUCKET, SORT_BY_ASPECT, SORT_BY_DIMS]

# File-list sort modes
FILE_SORT_NAME = "Name (A\u2192Z)"
FILE_SORT_MP = "Megapixels (high\u2192low)"
FILE_SORT_CROPS = "Crop boxes (most\u2192few)"
FILE_SORT_SIM = "Similarity (clustered)"
FILE_SORT_MODES = [FILE_SORT_NAME, FILE_SORT_MP, FILE_SORT_CROPS, FILE_SORT_SIM]


def sort_buckets(buckets: list[tuple[int, int, int]], mode: str
                 ) -> list[tuple[int, int, int]]:
    """Sort a list of (w, h, target) buckets by the given mode.
    'portrait' = tall = small w/h; 'landscape' = wide = large w/h."""
    if mode == SORT_BY_BUCKET:
        # group by target ascending, within group aspect asc, ties by pixels
        return sorted(buckets, key=lambda r: (r[2], r[0] / r[1], r[0] * r[1]))
    if mode == SORT_BY_ASPECT:
        # aspect asc (portrait first), ties broken by total size asc
        return sorted(buckets, key=lambda r: (r[0] / r[1], r[0] * r[1]))
    # SORT_BY_DIMS: max X dim asc, subsorted by max Y dim asc
    return sorted(buckets, key=lambda r: (r[0], r[1]))


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------

class Box:
    def __init__(self, w, h, x, y, exported=False):
        self.w, self.h, self.x, self.y = int(w), int(h), int(x), int(y)
        self.exported = exported

    def to_dict(self):
        return {"w": self.w, "h": self.h, "x": self.x, "y": self.y,
                "exported": self.exported}

    @staticmethod
    def from_dict(d):
        return Box(d["w"], d["h"], d["x"], d["y"], d.get("exported", False))


class ImgItem:
    """One entry in the browser: an original image or a downscale clone."""
    _next_uid = 1

    def __init__(self, src_name, scale=1.0, method="Lanczos", is_clone=False,
                 rot=0, flip_h=False, flip_v=False):
        self.uid = ImgItem._next_uid
        ImgItem._next_uid += 1
        self.src_name = src_name          # path relative to source folder
        self.scale = float(scale)         # <= 1.0 always (no upscaling)
        self.method = method
        self.is_clone = is_clone
        self.rot = int(rot) % 360         # CCW degrees: 0/90/180/270
        self.flip_h = bool(flip_h)        # applied BEFORE rotation
        self.flip_v = bool(flip_v)
        self.locked = False               # scale/orient locked (boxes / applied)
        self.excluded = False             # sinks to bottom of every sort
        self.boxes: list[Box] = []        # coords in WORKING (oriented+scaled) space

    def _orient_tag(self):
        bits = []
        if self.rot:
            bits.append(f"rot{self.rot}")
        if self.flip_h:
            bits.append("flipH")
        if self.flip_v:
            bits.append("flipV")
        return " ".join(bits)

    def label(self):
        base = os.path.basename(self.src_name)
        extra = []
        if self.scale != 1.0:
            extra.append(f"x{self.scale:.3f} {self.method}")
        ot = self._orient_tag()
        if ot:
            extra.append(ot)
        if self.is_clone and not extra:
            extra.append("clone")
        return f"{base}  [{', '.join(extra)}]" if extra else base

    def to_dict(self):
        return {"src_name": self.src_name, "scale": self.scale,
                "method": self.method, "is_clone": self.is_clone,
                "rot": self.rot, "flip_h": self.flip_h, "flip_v": self.flip_v,
                "locked": self.locked, "excluded": self.excluded,
                "boxes": [b.to_dict() for b in self.boxes]}

    @staticmethod
    def from_dict(d):
        it = ImgItem(d["src_name"], d.get("scale", 1.0),
                     d.get("method", "Lanczos"), d.get("is_clone", False),
                     d.get("rot", 0), d.get("flip_h", False), d.get("flip_v", False))
        it.locked = d.get("locked", False)
        it.excluded = d.get("excluded", False)
        it.boxes = [Box.from_dict(b) for b in d.get("boxes", [])]
        return it


# ----------------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------------

class CropApp:
    def __init__(self, root: tk.Tk, folder: str, targets: list[int]):
        self.root = root
        self.dark = False
        self.include_subdirs = False

        # canvas display state
        self.disp_scale = 1.0
        self.disp_ox = 0
        self.disp_oy = 0
        self.canvas_photo = None
        self.work_dims = (0, 0)

        self.fit_options: list[tuple[int, int, int]] = []
        self.row_to_opt: list = []
        self.opt_to_row: list = []
        self.sel_size_idx = 0
        self.selected_box: Box | None = None
        self.ghost_xy = None
        self._scale_job = None
        self._snap_dropdown_map = []

        self._build_ui()
        self._apply_theme()
        self._load_folder(folder, targets)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_folder(self, folder: str, targets: list[int]):
        """(Re)initialize all folder-dependent state. Safe to call repeatedly."""
        self.folder = folder
        self.targets = targets
        self.buckets = generate_buckets(targets)
        self._bucket_target = {(w, h): t for (w, h, t) in self.buckets}
        self._target_colors = self._build_target_colors(targets)
        self.session_path = os.path.join(folder, SESSION_FILENAME)

        self.items = []
        self.cur_idx = 0
        self.orig_cache = {}
        self.thumb_cache = {}
        self.dim_cache = {}
        self._feat_cache = {}
        self.selected_box = None
        self.ghost_xy = None
        self.sel_size_idx = 0
        self._pending_resort = False

        self.root.title(f"Krea2 Crop Tool - {self.folder}")
        self._scan_folder()
        self._load_session()
        self.items = self._ordered_items()
        self._populate_tree()
        if self.items:
            self._select_item(0)
        else:
            self.canvas.delete("all")
            self.size_list.delete(0, tk.END)
            self.dims_label.configure(text="(no images found)")

    # ------------------------------------------------------------- reload ----

    def _reload_current(self):
        """Start over on the current folder (re-scan, reload saved session)."""
        self._save_session()
        self._load_folder(self.folder, self.targets)
        self._set_status(f"Reloaded {self.folder}")

    def _open_folder(self):
        """Pick a different folder, keeping the same target resolutions."""
        self._save_session()
        new = filedialog.askdirectory(title="Select image folder",
                                      parent=self.root)
        if not new:
            return
        self._load_folder(new, self.targets)
        if not self.items:
            messagebox.showinfo("Krea2 Crop Tool",
                                "No images found in that folder.")

    def _toggle_subdirs(self):
        self.include_subdirs = bool(self.subdirs_var.get())
        self._save_session()
        self._load_folder(self.folder, self.targets)
        self._set_status(
            f"Subdirectories {'included' if self.include_subdirs else 'excluded'}.")

    def _output_rel(self) -> str:
        """Current output path from the field. May be a nested subpath
        ('crops/train'), a parent-relative path ('../crops/train'), or an
        absolute path. Falls back to the default folder name if empty."""
        raw = self.outname_var.get().strip() if hasattr(self, "outname_var") else ""
        raw = raw.strip('"').strip("'").strip()
        return raw or OUTPUT_DIRNAME

    @property
    def out_root(self) -> str:
        rel = self._output_rel()
        if os.path.isabs(rel):
            return os.path.normpath(rel)
        return os.path.normpath(os.path.join(self.folder, rel))

    # ----------------------------------------------------- colors / theme ---

    def _build_target_colors(self, targets):
        """Map each target -> a text color. Blue = smallest target, red =
        largest, gradient (blue->magenta->red) between. Tuned dark enough to
        read on the white crop-size list."""
        ts = sorted(set(targets))
        colors = {}
        n = len(ts)
        for i, t in enumerate(ts):
            frac = 0.0 if n == 1 else i / (n - 1)
            hue = (240 + frac * 120) / 360.0          # 240deg blue -> 360deg red
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.62)
            colors[t] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        return colors

    def _apply_theme(self):
        dark = self.dark
        if dark:
            bg, fg = "#2b2b2b", "#e6e6e6"
            field, tree_bg = "#3c3c3c", "#333333"
            sel = "#4a6ea9"
            self.canvas_bg = "#1a1a1a"
        else:
            bg, fg = "#f0f0f0", "#000000"
            field, tree_bg = "#ffffff", "#ffffff"
            sel = "#3a7bd5"
            self.canvas_bg = "#202020"

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.root.configure(bg=bg)
        style.configure(".", background=bg, foreground=fg,
                        fieldbackground=field)
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TButton", background=field, foreground=fg)
        style.map("TButton", background=[("active", sel)])
        style.configure("TCheckbutton", background=bg, foreground=fg)
        style.map("TCheckbutton", background=[("active", bg)])
        style.configure("TEntry", fieldbackground=field, foreground=fg)
        style.configure("TCombobox", fieldbackground=field, foreground=fg,
                        background=field, arrowcolor=fg)
        # readonly is the state our comboboxes use; without these the text is
        # gray-on-gray in dark mode
        style.map("TCombobox",
                  fieldbackground=[("readonly", field), ("disabled", field)],
                  foreground=[("readonly", fg), ("disabled", "#888888")],
                  selectbackground=[("readonly", field)],
                  selectforeground=[("readonly", fg)],
                  arrowcolor=[("readonly", fg)])
        # the combobox drop-down list is a separate Tk listbox
        self.root.option_add("*TCombobox*Listbox.background", field)
        self.root.option_add("*TCombobox*Listbox.foreground", fg)
        self.root.option_add("*TCombobox*Listbox.selectBackground", sel)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        style.configure("TMenubutton", background=field, foreground=fg,
                        arrowcolor=fg)
        style.map("TMenubutton", background=[("active", sel)])
        style.configure("TScale", background=bg)
        style.configure("Treeview", background=tree_bg, fieldbackground=tree_bg,
                        foreground=fg)
        style.map("Treeview", background=[("selected", sel)],
                  foreground=[("selected", "#ffffff")])
        style.configure("Treeview.Heading", background=field, foreground=fg)

        # tk (non-ttk) widgets
        if hasattr(self, "canvas"):
            self.canvas.configure(bg=self.canvas_bg)
        # crop-size list stays WHITE in both modes (gradient is tuned for white)
        if hasattr(self, "size_list"):
            self.size_list.configure(bg="#ffffff", fg="#000000",
                                     selectbackground="#c8dcff",
                                     selectforeground="#000000")
        if hasattr(self, "status"):
            self.status.configure(background=bg, foreground=fg)

    def _toggle_dark(self):
        self.dark = not self.dark
        self._apply_theme()
        if self.items:
            self._redraw(full=True)
        self._set_status(f"{'Dark' if self.dark else 'Light'} mode.")

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        self.root.title("Krea2 Crop Tool")
        self.root.geometry("1500x900")

        style = ttk.Style(self.root)
        style.configure("Treeview", rowheight=54)

        # ---- file / view bar ----
        top0 = ttk.Frame(self.root, padding=(4, 4, 4, 0))
        top0.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(top0, text="Open folder\u2026",
                   command=self._open_folder).pack(side=tk.LEFT)
        ttk.Button(top0, text="Reload / start over",
                   command=self._reload_current).pack(side=tk.LEFT, padx=4)
        self.subdirs_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top0, text="Include subdirectories",
                        variable=self.subdirs_var,
                        command=self._toggle_subdirs).pack(side=tk.LEFT, padx=8)
        ttk.Button(top0, text="Toggle dark mode",
                   command=self._toggle_dark).pack(side=tk.LEFT, padx=4)
        ttk.Label(top0, text="Output subdir:").pack(side=tk.LEFT, padx=(12, 2))
        self.outname_var = tk.StringVar(value=OUTPUT_DIRNAME)
        self.outname_entry = ttk.Entry(top0, width=16,
                                       textvariable=self.outname_var)
        self.outname_entry.pack(side=tk.LEFT)
        self.dims_label = ttk.Label(top0, text="")
        self.dims_label.pack(side=tk.RIGHT, padx=8)

        # ---- top toolbar (rows so nothing overflows off-screen) ----
        top = ttk.Frame(self.root, padding=(4, 4, 4, 0))
        top.pack(side=tk.TOP, fill=tk.X)
        top2 = ttk.Frame(self.root, padding=(4, 2, 4, 4))
        top2.pack(side=tk.TOP, fill=tk.X)

        # row 1: downscale + snap
        ttk.Label(top, text="Downscale:").pack(side=tk.LEFT)
        ttk.Button(top, text="◀", width=2,
                   command=lambda: self._snap_step(-1)).pack(side=tk.LEFT)
        self.scale_var = tk.DoubleVar(value=100.0)
        self.scale_slider = ttk.Scale(top, from_=10.0, to=100.0, length=220,
                                      variable=self.scale_var,
                                      command=self._on_slider_move)
        self.scale_slider.pack(side=tk.LEFT, padx=2)
        self.scale_slider.bind("<ButtonPress-1>", self._on_slider_press)
        ttk.Button(top, text="▶", width=2,
                   command=lambda: self._snap_step(+1)).pack(side=tk.LEFT)

        self.scale_entry = ttk.Entry(top, width=6)
        self.scale_entry.pack(side=tk.LEFT, padx=(6, 0))
        self.scale_entry.bind("<Return>", self._on_scale_entry)
        ttk.Label(top, text="%").pack(side=tk.LEFT)

        self.snap_label = ttk.Label(top, text="", width=24, anchor=tk.W,
                                    font=("Consolas", 9))
        self.snap_label.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(top, text="Method:").pack(side=tk.LEFT, padx=(8, 0))
        self.method_var = tk.StringVar(value="Lanczos")
        self.method_combo = ttk.Combobox(top, textvariable=self.method_var,
                                         values=list(RESAMPLE_METHODS.keys()),
                                         width=10, state="readonly")
        self.method_combo.pack(side=tk.LEFT, padx=2)
        self.method_combo.bind("<<ComboboxSelected>>", lambda e: self._method_changed())

        # snap-to-bucket: a menu so entries can be colored per bucket
        ttk.Label(top, text="Snap to bucket:").pack(side=tk.LEFT, padx=(8, 0))
        self.snap_menu = tk.Menu(self.root, tearoff=0, bg="#ffffff",
                                 fg="#000000", activebackground="#c8dcff",
                                 activeforeground="#000000")
        self.snap_mb = ttk.Menubutton(top, text="(choose) \u25be",
                                      menu=self.snap_menu)
        self.snap_mb.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # row 2: transforms + actions
        ttk.Button(top2, text="\u21ba CCW", width=6,
                   command=lambda: self._apply_transform("ccw")).pack(side=tk.LEFT)
        ttk.Button(top2, text="\u21bb CW", width=6,
                   command=lambda: self._apply_transform("cw")).pack(side=tk.LEFT, padx=2)
        ttk.Button(top2, text="Flip H", width=6,
                   command=lambda: self._apply_transform("fliph")).pack(side=tk.LEFT, padx=2)
        ttk.Button(top2, text="Flip V", width=6,
                   command=lambda: self._apply_transform("flipv")).pack(side=tk.LEFT, padx=2)
        ttk.Separator(top2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(top2, text="Apply Downscaling",
                   command=self._apply_downscale).pack(side=tk.LEFT)
        ttk.Button(top2, text="Clone + Downscale (Ctrl+D)",
                   command=self._clone_current).pack(side=tk.LEFT, padx=4)
        ttk.Button(top2, text="CROP this image (C)",
                   command=self._export_current).pack(side=tk.LEFT, padx=(16, 2))
        ttk.Button(top2, text="CROP ALL images",
                   command=self._export_all).pack(side=tk.LEFT, padx=2)

        # ---- status bar ----
        self.status = ttk.Label(
            self.root, anchor=tk.W, padding=(6, 2),
            text="Wheel: cycle size | L-click: place box | R-click: select box | "
                 "Del: delete | Arrows: nudge | PgUp/PgDn: images | C: crop | "
                 "Ctrl+D: clone | \u25c0\u25b6 by slider: snap to best-fit bucket | "
                 "Ctrl+\u2191/\u2193: jump to next image with crops | "
                 "Ctrl+T: exclude | Esc: deselect")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # ---- main panes ----
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # left: crop size options
        left = ttk.Frame(main, padding=4)
        left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="Crop sizes (fit current image)").pack(anchor=tk.W)
        self.sort_var = tk.StringVar(value=SORT_BY_BUCKET)
        sort_combo = ttk.Combobox(left, textvariable=self.sort_var,
                                  values=SORT_MODES, width=28, state="readonly")
        sort_combo.pack(anchor=tk.W, pady=(2, 4))
        sort_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_fit_options())
        self.size_list = tk.Listbox(left, width=30, exportselection=False,
                                    font=("Consolas", 10))
        self.size_list.pack(fill=tk.Y, expand=True)
        self.size_list.bind("<<ListboxSelect>>", self._on_size_click)

        # right: image browser
        right = ttk.Frame(main, padding=4)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(right, text="Images  (Ctrl+\u2191/\u2193: jump to next with crops)"
                  ).pack(side=tk.TOP, anchor=tk.W)
        sort_row = ttk.Frame(right)
        sort_row.pack(side=tk.TOP, fill=tk.X, pady=(2, 2))
        ttk.Label(sort_row, text="Sort:").pack(side=tk.LEFT)
        self.file_sort_var = tk.StringVar(value=FILE_SORT_NAME)
        fsort = ttk.Combobox(sort_row, textvariable=self.file_sort_var,
                             values=FILE_SORT_MODES, width=22, state="readonly")
        fsort.pack(side=tk.LEFT, padx=4)
        fsort.bind("<<ComboboxSelected>>", lambda e: self._sort_items())

        # buttons reserved at the bottom FIRST so they can't be clipped off
        btns = ttk.Frame(right)
        btns.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(btns, text="Exclude / include (Ctrl+T)",
                   command=self._toggle_exclude_current).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btns, text="Reset exports (selected image)",
                   command=self._reset_exports_current).pack(fill=tk.X, pady=(2, 0))
        ttk.Button(btns, text="RESET ALL EXPORTS",
                   command=self._reset_exports_all).pack(fill=tk.X, pady=(2, 0))
        ttk.Button(btns, text="Tally output buckets",
                   command=self._tally_buckets).pack(fill=tk.X, pady=(2, 0))

        tree_wrap = ttk.Frame(right)
        tree_wrap.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(tree_wrap, columns=("mp", "crops"),
                                 selectmode="browse", height=10)
        self.tree.heading("#0", text="File")
        self.tree.heading("mp", text="Mp")
        self.tree.heading("crops", text="Crops")
        self.tree.column("#0", width=320)
        self.tree.column("mp", width=64, anchor=tk.CENTER)
        self.tree.column("crops", width=70, anchor=tk.CENTER)
        vsb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<MouseWheel>", self._on_tree_wheel)
        self.tree.bind("<Button-4>", lambda e: self.tree.yview_scroll(-1, "units"))
        self.tree.bind("<Button-5>", lambda e: self.tree.yview_scroll(1, "units"))
        self.tree.tag_configure("excluded", foreground="#999999")

        # center: canvas
        self.canvas = tk.Canvas(main, bg="#202020", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._redraw(full=True))
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        # mouse wheel (Windows/mac + Linux)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._cycle_size(-1))
        self.canvas.bind("<Button-5>", lambda e: self._cycle_size(+1))

        # global keys
        self.root.bind("<Next>", lambda e: self._nav(+1))       # PgDn
        self.root.bind("<Prior>", lambda e: self._nav(-1))      # PgUp
        self.root.bind("<Delete>", lambda e: self._delete_selected_box())
        self.root.bind("<BackSpace>", lambda e: self._delete_selected_box())
        self.root.bind("<Escape>", lambda e: self._deselect_box())
        self.root.bind("<Key-c>", lambda e: self._export_current())
        self.root.bind("<Control-e>", lambda e: self._export_current())
        self.root.bind("<Control-d>", lambda e: self._clone_current())
        self.root.bind("<Control-Up>", lambda e: self._nav_with_crops(-1))
        self.root.bind("<Control-Down>", lambda e: self._nav_with_crops(+1))
        self.root.bind("<Control-Left>", lambda e: self._snap_step(-1))
        self.root.bind("<Control-Right>", lambda e: self._snap_step(+1))
        self.root.bind("<Control-t>", lambda e: self._toggle_exclude_current())
        for key, dx, dy in (("<Left>", -1, 0), ("<Right>", 1, 0),
                            ("<Up>", 0, -1), ("<Down>", 0, 1)):
            self.root.bind(key, lambda e, dx=dx, dy=dy: self._nudge(dx, dy, 1))
            self.root.bind(key.replace("<", "<Shift-"),
                           lambda e, dx=dx, dy=dy: self._nudge(dx, dy, 16))

    # ------------------------------------------------------- folder/session --

    def _scan_folder(self):
        if not self.include_subdirs:
            names = sorted(
                f for f in os.listdir(self.folder)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                and os.path.isfile(os.path.join(self.folder, f))
            )
            self.src_names = names
            return
        # recursive: relative paths, but never descend into the output dir
        out_abs = os.path.normpath(self.out_root)
        names = []
        for root, dirs, files in os.walk(self.folder):
            keep = []
            for d in dirs:
                if d.startswith(".") or d == OUTPUT_DIRNAME:
                    continue
                if os.path.normpath(os.path.join(root, d)) == out_abs:
                    continue
                keep.append(d)
            dirs[:] = keep
            for f in files:
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                    rel = os.path.relpath(os.path.join(root, f), self.folder)
                    names.append(rel.replace("\\", "/"))
        self.src_names = sorted(names)

    def _load_session(self):
        loaded = {}
        if os.path.isfile(self.session_path):
            try:
                with open(self.session_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                saved_name = data.get("output_name")
                if saved_name and hasattr(self, "outname_var"):
                    self.outname_var.set(saved_name)
                for d in data.get("items", []):
                    loaded.setdefault(d["src_name"], []).append(ImgItem.from_dict(d))
            except Exception as e:
                messagebox.showwarning("Session", f"Could not read session file:\n{e}")

        self.items = []
        scanned = set(self.src_names)
        for name in self.src_names:
            if name in loaded:
                self.items.extend(loaded[name])
            else:
                self.items.append(ImgItem(name))
        # keep session entries for files not in the current scan (e.g. because
        # "include subdirectories" is off) so their boxes aren't lost on save
        self._orphan_items = [it for name, lst in loaded.items()
                              if name not in scanned for it in lst]

    def _save_session(self):
        items = list(self.items) + list(getattr(self, "_orphan_items", []))
        data = {
            "targets": self.targets,
            "output_name": self._output_rel(),
            "items": [it.to_dict() for it in items],
        }
        try:
            tmp = self.session_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1)
            os.replace(tmp, self.session_path)
        except Exception as e:
            messagebox.showwarning("Session", f"Could not save session:\n{e}")

    def _on_close(self):
        self._save_session()
        self.root.destroy()

    # ------------------------------------------------------------- images --

    def _get_original(self, src_name) -> Image.Image:
        img = self.orig_cache.get(src_name)
        if img is None:
            img = Image.open(os.path.join(self.folder, src_name))
            img = ImageOps.exif_transpose(img)
            img.load()
            if len(self.orig_cache) > 6:      # small cache
                self.orig_cache.pop(next(iter(self.orig_cache)))
            self.orig_cache[src_name] = img
        return img

    def _oriented_image(self, item: ImgItem) -> Image.Image:
        """Original with flips then rotation applied (no scaling). WYSIWYG."""
        img = self._get_original(item.src_name)
        if item.flip_h:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if item.flip_v:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if item.rot == 90:
            img = img.transpose(Image.Transpose.ROTATE_90)    # CCW
        elif item.rot == 180:
            img = img.transpose(Image.Transpose.ROTATE_180)
        elif item.rot == 270:
            img = img.transpose(Image.Transpose.ROTATE_270)
        return img

    def _oriented_size(self, item: ImgItem) -> tuple[int, int]:
        ow, oh = self._get_original(item.src_name).size
        return (oh, ow) if item.rot in (90, 270) else (ow, oh)

    def _working_dims(self, item: ImgItem) -> tuple[int, int]:
        ow, oh = self._oriented_size(item)
        if item.scale >= 0.9995:
            return ow, oh
        return max(1, round(ow * item.scale)), max(1, round(oh * item.scale))

    def _working_image(self, item: ImgItem) -> Image.Image:
        img = self._oriented_image(item)
        if item.scale >= 0.9995:
            return img
        w, h = self._working_dims(item)
        return img.resize((w, h), RESAMPLE_METHODS[item.method])

    def _min_scale(self, item: ImgItem) -> float:
        """Smallest scale at which at least one bucket still fits. No upscaling."""
        ow, oh = self._oriented_size(item)
        reqs = [max(bw / ow, bh / oh) for (bw, bh, _t) in self.buckets]
        reqs = [r for r in reqs if r <= 1.0]
        if not reqs:
            return 1.0
        return min(reqs)

    # -------------------------------------------------------------- tree ----

    def _thumb(self, src_name) -> ImageTk.PhotoImage:
        ph = self.thumb_cache.get(src_name)
        if ph is None:
            img = self._get_original(src_name).copy()
            img.thumbnail((72, 48))
            ph = ImageTk.PhotoImage(img)
            self.thumb_cache[src_name] = ph
        return ph

    def _img_pixels(self, src_name) -> int:
        """Total pixels of the source image (header-only read), cached."""
        px = self.dim_cache.get(src_name)
        if px is None:
            try:
                with Image.open(os.path.join(self.folder, src_name)) as im:
                    w, h = im.size
                px = w * h
            except Exception:
                px = 0
            self.dim_cache[src_name] = px
        return px

    def _mp_str(self, src_name) -> str:
        return f"{self._img_pixels(src_name) / 1e6:.1f}"

    def _file_sort_key(self):
        mode = self.file_sort_var.get() if hasattr(self, "file_sort_var") \
            else FILE_SORT_NAME
        if mode == FILE_SORT_MP:
            return lambda it: (-self._img_pixels(it.src_name),
                               it.src_name.lower(), it.uid)
        if mode == FILE_SORT_CROPS:
            return lambda it: (-len(it.boxes), it.src_name.lower(), it.uid)
        return lambda it: (it.src_name.lower(), it.uid)

    def _feature(self, src_name):
        """Size/crop-tolerant signature: L2-normalized HSV color histogram.
        Position-independent, so moderate crops stay similar and size is
        irrelevant. Cached per source file."""
        f = self._feat_cache.get(src_name)
        if f is None:
            try:
                im = self._get_original(src_name).convert("HSV").resize(
                    (64, 64), Image.Resampling.BILINEAR)
                arr = np.asarray(im, dtype=np.float32)
                h = np.clip((arr[..., 0] / 256 * 12).astype(int), 0, 11)
                s = np.clip((arr[..., 1] / 256 * 4).astype(int), 0, 3)
                v = np.clip((arr[..., 2] / 256 * 4).astype(int), 0, 3)
                idx = (h * 16 + s * 4 + v).ravel()
                hist = np.bincount(idx, minlength=192).astype(np.float32)
                n = float(np.linalg.norm(hist))
                f = hist / n if n > 0 else hist
            except Exception:
                f = np.zeros(192, dtype=np.float32)
            self._feat_cache[src_name] = f
        return f

    def _similarity_order(self, items):
        """Greedy nearest-neighbor chain seeded by the largest-Mp image, so
        similar images sit next to each other (clusters = runs of neighbors)."""
        if not items:
            return []
        feats = {it.uid: self._feature(it.src_name) for it in items}
        remaining = list(items)
        seed = max(remaining, key=lambda it: self._img_pixels(it.src_name))
        order = [seed]
        remaining.remove(seed)
        last = seed
        while remaining:
            lf = feats[last.uid]
            best = max(remaining,
                       key=lambda it: (float(np.dot(lf, feats[it.uid])),
                                       self._img_pixels(it.src_name)))
            order.append(best)
            remaining.remove(best)
            last = best
        return order

    def _ordered_items(self):
        """Full display order: active images sorted by the chosen mode, then
        excluded images (always) at the bottom in name order."""
        excluded = [it for it in self.items if getattr(it, "excluded", False)]
        active = [it for it in self.items if not getattr(it, "excluded", False)]
        mode = self.file_sort_var.get() if hasattr(self, "file_sort_var") \
            else FILE_SORT_NAME
        if mode == FILE_SORT_SIM and HAS_NUMPY:
            active = self._similarity_order(active)
        else:
            active.sort(key=self._file_sort_key())
        excluded.sort(key=lambda it: (it.src_name.lower(), it.uid))
        return active + excluded

    def _sort_items(self):
        if not self.items:
            return
        if self.file_sort_var.get() == FILE_SORT_SIM and not HAS_NUMPY:
            messagebox.showinfo(
                "Similarity sort",
                "Similarity sorting needs numpy.\n\npip install numpy")
            self.file_sort_var.set(FILE_SORT_NAME)
            return
        cur_uid = self.items[self.cur_idx].uid \
            if 0 <= self.cur_idx < len(self.items) else None
        self.items = self._ordered_items()
        self._pending_resort = False
        if cur_uid is not None:
            for i, it in enumerate(self.items):
                if it.uid == cur_uid:
                    self.cur_idx = i
                    break
        self._populate_tree()
        if self.items:
            uid = str(self.items[self.cur_idx].uid)
            self.tree.selection_set(uid)
            self.tree.see(uid)

    def _toggle_exclude_current(self):
        if not self.items:
            return
        item = self.items[self.cur_idx]
        item.excluded = not item.excluded
        self._refresh_tree_row(item)          # grey it in place, don't move yet
        self._pending_resort = True           # settle when we leave this photo
        self._save_session()
        self._set_status(
            f"'{os.path.basename(item.src_name)}' "
            f"{'excluded' if item.excluded else 'included'} "
            f"(sinks when you move to another image).")

    def _populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        for it in self.items:
            n_exp = sum(1 for b in it.boxes if b.exported)
            prefix = "\u2717 " if it.excluded else ""
            self.tree.insert("", tk.END, iid=str(it.uid),
                             text=prefix + it.label(),
                             image=self._thumb(it.src_name),
                             tags=("excluded",) if it.excluded else (),
                             values=(self._mp_str(it.src_name),
                                     f"{n_exp}/{len(it.boxes)}"))

    def _refresh_tree_row(self, item: ImgItem):
        n_exp = sum(1 for b in item.boxes if b.exported)
        prefix = "\u2717 " if item.excluded else ""
        self.tree.item(str(item.uid), text=prefix + item.label(),
                       tags=("excluded",) if item.excluded else (),
                       values=(self._mp_str(item.src_name),
                               f"{n_exp}/{len(item.boxes)}"))

    def _on_tree_select(self, _e):
        sel = self.tree.selection()
        if not sel:
            return
        uid = int(sel[0])
        for it in self.items:
            if it.uid == uid:
                if it is not self.items[self.cur_idx]:
                    self._go_to(it, from_tree=True)
                return

    # ---------------------------------------------------------- selection ---

    def _go_to(self, target_item, from_tree=False):
        """Select target_item. If an excluded image is waiting to be settled
        and we're actually leaving it, move trashed items to the bottom now
        (preserving everyone else's order) and land on target_item by identity."""
        leaving = self.items[self.cur_idx] if self.items else None
        resorted = False
        if self._pending_resort and target_item is not leaving:
            # stable partition: keep current order, just sink excluded ones.
            # (Do NOT re-run the full sort here — that would reshuffle the
            # similarity chain and throw the view around.)
            active = [it for it in self.items if not it.excluded]
            excluded = [it for it in self.items if it.excluded]
            self.items = active + excluded
            self._populate_tree()
            self._pending_resort = False
            resorted = True
        idx = self.items.index(target_item)
        self._select_item(idx, from_tree=(from_tree and not resorted))

    def _select_item(self, idx, from_tree=False):
        self.cur_idx = max(0, min(idx, len(self.items) - 1))
        item = self.items[self.cur_idx]
        self.selected_box = None
        self.ghost_xy = None

        # slider range for this item
        mn = self._min_scale(item) * 100.0
        self.scale_slider.configure(from_=mn, to=100.0)
        self.scale_var.set(item.scale * 100.0)
        self._set_scale_entry(item.scale)
        self.method_var.set(item.method)

        self._refresh_snap_dropdown(item)
        self._update_snap_indicator(item)

        if not from_tree:
            self.tree.selection_set(str(item.uid))
            self.tree.see(str(item.uid))

        self._refresh_fit_options()
        self._redraw(full=True)

    def _nav(self, step):
        if not self.items:
            return
        idx = max(0, min(self.cur_idx + step, len(self.items) - 1))
        self._go_to(self.items[idx])

    def _nav_with_crops(self, step):
        """Jump to the next image (in the given direction) that has >0 boxes."""
        if not self.items:
            return
        i = self.cur_idx + step
        while 0 <= i < len(self.items):
            if self.items[i].boxes:
                self._go_to(self.items[i])
                return
            i += step
        self._set_status("No further images with crops in that direction.")

    def _on_tree_wheel(self, e):
        self.tree.yview_scroll(-1 if e.delta > 0 else 1, "units")
        return "break"

    def _reset_exports_current(self):
        if not self.items:
            return
        item = self.items[self.cur_idx]
        n = sum(1 for b in item.boxes if b.exported)
        if n == 0:
            self._set_status("This image has no exported crops to reset.")
            return
        if not messagebox.askyesno(
                "Reset exports",
                f"Mark all {n} exported crop(s) on '{item.src_name}' as "
                f"un-exported?\n\nThis only clears the flag so they can be "
                f"re-cropped. Files already written to disk are NOT deleted."):
            return
        for b in item.boxes:
            b.exported = False
        self._refresh_tree_row(item)
        self._save_session()
        self._redraw()
        self._set_status(f"Reset {n} export flag(s) on this image.")

    def _reset_exports_all(self):
        if not self.items:
            return
        n = sum(1 for it in self.items for b in it.boxes if b.exported)
        if n == 0:
            self._set_status("No exported crops anywhere to reset.")
            return
        if not messagebox.askyesno(
                "Reset ALL exports",
                f"Mark all {n} exported crop(s) across every image as "
                f"un-exported?\n\nThis only clears the flags so they can be "
                f"re-cropped. Files already written to disk are NOT deleted."):
            return
        for it in self.items:
            for b in it.boxes:
                b.exported = False
            self._refresh_tree_row(it)
        self._save_session()
        self._redraw()
        self._set_status(f"Reset {n} export flag(s) across all images.")

    def _tally_buckets(self):
        """Scan the output folders and count how many images of each exact
        size live in each bucket (target) folder."""
        if not os.path.isdir(self.out_root):
            messagebox.showinfo("Tally", f"No output folder yet:\n{self.out_root}")
            return
        data = {}          # {target_folder: {(w, h): count}}
        errors = 0
        for entry in sorted(os.listdir(self.out_root)):
            sub = os.path.join(self.out_root, entry)
            if not os.path.isdir(sub):
                continue
            counts = {}
            for f in os.listdir(sub):
                if os.path.splitext(f)[1].lower() not in IMAGE_EXTS:
                    continue
                try:
                    with Image.open(os.path.join(sub, f)) as im:
                        wh = im.size
                except Exception:
                    errors += 1
                    continue
                counts[wh] = counts.get(wh, 0) + 1
            if counts:
                data[entry] = counts
        if not data:
            messagebox.showinfo("Tally", "No images found in the output folders.")
            return
        self._open_tally_window(data, errors)

    def _open_tally_window(self, data, errors):
        win = tk.Toplevel(self.root)
        win.title("Output bucket tally")
        win.geometry("560x640")

        top = ttk.Frame(win, padding=6)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Batch size:").pack(side=tk.LEFT)
        bs_var = tk.StringVar(value="1")
        bs_entry = ttk.Entry(top, width=6, textvariable=bs_var)
        bs_entry.pack(side=tk.LEFT, padx=4)
        summary = ttk.Label(top, text="", font=("Consolas", 9))
        summary.pack(side=tk.LEFT, padx=10)

        wrap = ttk.Frame(win, padding=(6, 0, 6, 6))
        wrap.pack(fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(wrap, columns=("count", "batches", "drop"))
        tree.heading("#0", text="Bucket / size")
        tree.heading("count", text="Images")
        tree.heading("batches", text="Batches")
        tree.heading("drop", text="Dropped/epoch")
        tree.column("#0", width=230)
        tree.column("count", width=80, anchor=tk.CENTER)
        tree.column("batches", width=90, anchor=tk.CENTER)
        tree.column("drop", width=120, anchor=tk.CENTER)
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree.tag_configure("warn", foreground="#c00000")

        def refresh():
            try:
                bs = max(1, int(bs_var.get()))
            except ValueError:
                bs = 1
            tree.delete(*tree.get_children())
            grand_imgs = grand_steps = grand_drop = 0
            for target in sorted(data, key=lambda k: (len(k), k)):
                counts = data[target]
                t_imgs = sum(counts.values())
                t_steps = sum(n // bs for n in counts.values())
                t_drop = sum(n % bs for n in counts.values())
                grand_imgs += t_imgs
                grand_steps += t_steps
                grand_drop += t_drop
                parent = tree.insert("", tk.END, text=f"crops/{target}",
                                     values=(t_imgs, t_steps, t_drop), open=True)
                for wh in sorted(counts, key=lambda s: (s[0] * s[1], s[0] / s[1])):
                    n = counts[wh]
                    tag = "warn" if n // bs == 0 else ""
                    tree.insert(parent, tk.END, text=f"    {wh[0]}x{wh[1]}",
                                values=(n, n // bs, n % bs), tags=(tag,))
            msg = (f"total: {grand_imgs} imgs  |  {grand_steps} steps/epoch  "
                   f"|  {grand_drop} dropped/epoch")
            if errors:
                msg += f"  |  {errors} unreadable"
            summary.configure(text=msg)

        bs_entry.bind("<Return>", lambda e: refresh())
        ttk.Button(top, text="Refresh", command=refresh).pack(side=tk.LEFT, padx=4)
        refresh()

    def _refresh_fit_options(self):
        item = self.items[self.cur_idx]
        ww, wh = self._working_dims(item)
        self.work_dims = (ww, wh)

        # remember currently selected (w,h) so we can keep it across re-sorts
        prev = None
        if self.fit_options and 0 <= self.sel_size_idx < len(self.fit_options):
            prev = self.fit_options[self.sel_size_idx][:2]

        fit = [b for b in self.buckets if b[0] <= ww and b[1] <= wh]
        mode = self.sort_var.get()
        self.fit_options = sort_buckets(fit, mode)

        # build listbox rows (with headers in bucket mode) + row<->option maps
        self.size_list.delete(0, tk.END)
        self.row_to_opt = []
        self.opt_to_row = [None] * len(self.fit_options)
        row = 0
        cur_group = None
        for opt_idx, (bw, bh, t) in enumerate(self.fit_options):
            if mode == SORT_BY_BUCKET and t != cur_group:
                cur_group = t
                self.size_list.insert(tk.END, f"\u2500\u2500 bucket {t} \u2500\u2500")
                self.size_list.itemconfig(row, foreground="#888888")
                self.row_to_opt.append(None)
                row += 1
            if mode == SORT_BY_BUCKET:
                text = f"  {bw:>4} x {bh:<4}  ({nice_ratio(bw, bh)})"
            else:
                text = f"{bw:>4} x {bh:<4} ({nice_ratio(bw, bh)}) [{t}]"
            self.size_list.insert(tk.END, text)
            self.size_list.itemconfig(
                row, foreground=self._target_colors.get(t, "#000000"))
            self.row_to_opt.append(opt_idx)
            self.opt_to_row[opt_idx] = row
            row += 1

        # restore or clamp selection
        if self.fit_options:
            new_idx = 0
            if prev is not None:
                for i, (bw, bh, _t) in enumerate(self.fit_options):
                    if (bw, bh) == prev:
                        new_idx = i
                        break
            self.sel_size_idx = new_idx
            self._sync_size_selection()
        else:
            self.sel_size_idx = 0

        self.dims_label.configure(
            text=f"working: {ww} x {wh}   options: {len(self.fit_options)}")

    def _sync_size_selection(self):
        """Highlight the listbox row for the current sel_size_idx."""
        self.size_list.selection_clear(0, tk.END)
        if 0 <= self.sel_size_idx < len(self.opt_to_row):
            r = self.opt_to_row[self.sel_size_idx]
            if r is not None:
                self.size_list.selection_set(r)
                self.size_list.see(r)

    def _on_size_click(self, _e):
        sel = self.size_list.curselection()
        if not sel:
            return
        opt = self.row_to_opt[sel[0]]
        if opt is None:            # clicked a group header: ignore, restore
            self._sync_size_selection()
            return
        self.sel_size_idx = opt
        self._redraw()

    def _cycle_size(self, step):
        if not self.fit_options:
            return
        self.sel_size_idx = (self.sel_size_idx + step) % len(self.fit_options)
        self._sync_size_selection()
        self._redraw()

    def _on_wheel(self, e):
        self._cycle_size(-1 if e.delta > 0 else +1)


    # ------------------------------------------------------------ scaling ---

    # -- bucket snapping ------------------------------------------------------

    def _snap_candidates(self, item):
        """Buckets reachable by downscaling the oriented image (no upscaling).
        Each: dict(w, h, t, scale, loss). `scale` is the cover scale (whole
        image fills the bucket, minimal crop); `loss` is fraction cropped."""
        ow, oh = self._oriented_size(item)
        out = []
        for (bw, bh, t) in self.buckets:
            if bw <= ow and bh <= oh:
                s = min(1.0, max(bw / ow, bh / oh))
                ww, wh = max(1, round(ow * s)), max(1, round(oh * s))
                loss = 1.0 - (bw * bh) / (ww * wh)
                out.append({"w": bw, "h": bh, "t": t, "scale": s, "loss": loss})
        return out

    @staticmethod
    def _snap_rank(c):
        # group losses into 0.5% bands (rounding noise), then prefer the larger
        # bucket -> less downscaling, sharper result.
        return (round(c["loss"] / 0.005), -(c["w"] * c["h"]))

    def _refresh_snap_dropdown(self, item):
        cands = sorted(self._snap_candidates(item), key=self._snap_rank)
        self._snap_dropdown_map = cands
        # which candidate is the image currently sitting on (if any)?
        close = [c for c in cands if abs(c["scale"] - item.scale) <= 5e-4]
        on = min(close, key=self._snap_rank) if close else None
        self.snap_menu.delete(0, tk.END)
        for i, c in enumerate(cands):
            mark = "> " if c is on else "   "
            label = (f"{mark}{c['w']}x{c['h']}   {c['loss'] * 100:4.1f}% crop   "
                     f"(x{c['scale']:.3f})  [{c['t']}]")
            color = self._target_colors.get(c["t"], "#000000")
            self.snap_menu.add_command(label=label, foreground=color,
                                       activeforeground=color,
                                       command=lambda idx=i: self._pick_snap_bucket(idx))
        if on is not None:
            self.snap_mb.configure(text=f"> {on['w']}x{on['h']} \u25be")
        else:
            self.snap_mb.configure(text="(choose) \u25be")

    def _pick_snap_bucket(self, idx):
        item = self.items[self.cur_idx]
        if idx < 0 or idx >= len(self._snap_dropdown_map):
            return
        if item.locked or item.boxes:
            self._on_slider_press(None)
            return
        self._apply_snap(item, self._snap_dropdown_map[idx])

    def _snap_step(self, direction):
        """Arrows step to the next best-fit downscale (by scale value)."""
        item = self.items[self.cur_idx]
        if item.locked or item.boxes:
            self._on_slider_press(None)
            return
        cands = self._snap_candidates(item)
        if not cands:
            return
        # unique scales, each remembering its lowest-loss bucket
        by_scale = {}
        for c in cands:
            key = round(c["scale"], 5)
            if key not in by_scale or c["loss"] < by_scale[key]["loss"]:
                by_scale[key] = c
        ordered = sorted(by_scale.values(), key=lambda c: c["scale"])
        cur = item.scale
        if direction < 0:
            picks = [c for c in ordered if c["scale"] < cur - 1e-5]
            c = picks[-1] if picks else ordered[0]
        else:
            picks = [c for c in ordered if c["scale"] > cur + 1e-5]
            c = picks[0] if picks else ordered[-1]
        self._apply_snap(item, c)

    def _apply_snap(self, item, c):
        """Jump scale to candidate c, refresh, and pre-select its bucket."""
        item.scale = c["scale"]
        self.scale_var.set(c["scale"] * 100.0)
        self._set_scale_entry(c["scale"])
        self._scale_commit()
        # pre-select the snapped bucket in the crop-size list
        for i, (bw, bh, _t) in enumerate(self.fit_options):
            if (bw, bh) == (c["w"], c["h"]):
                self.sel_size_idx = i
                self._sync_size_selection()
                break
        self._redraw()
        self._set_status(
            f"Snapped to {c['w']}x{c['h']} at x{c['scale']:.3f} "
            f"({c['loss'] * 100:.1f}% cropped). Place the box to lock.")

    def _update_snap_indicator(self, item):
        """Show whether the current scale sits exactly on a bucket snap."""
        cands = self._snap_candidates(item)
        if not cands:
            self.snap_label.configure(text="", foreground="#888888")
            return
        close = [c for c in cands if abs(c["scale"] - item.scale) <= 5e-4]
        if close:
            on = min(close, key=self._snap_rank)   # best bucket at this scale
            self.snap_label.configure(
                text=f"\u25c9 fits {on['w']}x{on['h']} ({on['loss'] * 100:.1f}%)",
                foreground="#20a020")
        else:
            nearest = min(cands, key=self._snap_rank)
            self.snap_label.configure(
                text=f"best: {nearest['w']}x{nearest['h']} @ x{nearest['scale']:.3f}",
                foreground="#888888")

    # -- freeform scaling -----------------------------------------------------

    def _set_scale_entry(self, scale):
        self.scale_entry.delete(0, tk.END)
        self.scale_entry.insert(0, f"{scale * 100:.1f}")

    def _on_slider_press(self, _e):
        item = self.items[self.cur_idx]
        if item.locked or item.boxes:
            if messagebox.askyesno(
                    "Image has crops",
                    "This image already has crop boxes (scale is locked).\n"
                    "Clone it in the browser with a new scaled version?"):
                self._clone_current()
            return "break"
        return None

    def _on_slider_move(self, _val):
        item = self.items[self.cur_idx]
        if item.locked or item.boxes:
            return
        item.scale = min(1.0, self.scale_var.get() / 100.0)
        self._set_scale_entry(item.scale)
        if self._scale_job:
            self.root.after_cancel(self._scale_job)
        self._scale_job = self.root.after(120, self._scale_commit)

    def _scale_commit(self):
        self._scale_job = None
        item = self.items[self.cur_idx]
        self._refresh_fit_options()
        self._refresh_tree_row(item)
        self._update_snap_indicator(item)
        self._refresh_snap_dropdown(item)
        self._redraw(full=True)

    def _on_scale_entry(self, _e):
        item = self.items[self.cur_idx]
        if item.locked or item.boxes:
            self._on_slider_press(None)
            return
        try:
            v = float(self.scale_entry.get()) / 100.0
        except ValueError:
            return
        mn = self._min_scale(item)
        item.scale = max(mn, min(1.0, v))
        self.scale_var.set(item.scale * 100.0)
        self._set_scale_entry(item.scale)
        self._scale_commit()

    def _method_changed(self):
        item = self.items[self.cur_idx]
        if item.locked or item.boxes:
            self.method_var.set(item.method)
            self._on_slider_press(None)
            return
        item.method = self.method_var.get()
        self._refresh_tree_row(item)
        self._redraw(full=True)

    def _apply_downscale(self):
        item = self.items[self.cur_idx]
        item.locked = True
        self._refresh_tree_row(item)
        self._save_session()
        self._set_status(f"Downscaling locked at x{item.scale:.3f} ({item.method}).")

    def _apply_transform(self, kind):
        """Rotate/flip. Like downscaling: if the image already has boxes (or is
        locked), offer to clone and transform the clone instead."""
        if not self.items:
            return
        item = self.items[self.cur_idx]
        if item.locked or item.boxes:
            if not messagebox.askyesno(
                    "Image has crops",
                    "This image already has crop boxes (orientation is "
                    "locked).\nClone it with the new rotation/flip?"):
                return
            self._clone_current()
            item = self.items[self.cur_idx]   # now the clone
        if kind == "cw":
            item.rot = (item.rot - 90) % 360
        elif kind == "ccw":
            item.rot = (item.rot + 90) % 360
        elif kind == "fliph":
            item.flip_h = not item.flip_h
        elif kind == "flipv":
            item.flip_v = not item.flip_v
        # clamp scale to the new orientation's minimum, then refresh everything
        item.scale = min(1.0, max(item.scale, self._min_scale(item)))
        self.scale_var.set(item.scale * 100.0)
        self._set_scale_entry(item.scale)
        self._refresh_snap_dropdown(item)
        self._scale_commit()
        self._refresh_tree_row(item)
        self._save_session()
        self._set_status(f"Applied {kind}. Orientation: {item._orient_tag() or 'none'}.")

    def _clone_current(self):
        src = self.items[self.cur_idx]
        clone = ImgItem(src.src_name, scale=src.scale, method=src.method,
                        is_clone=True, rot=src.rot,
                        flip_h=src.flip_h, flip_v=src.flip_v)
        self.items.insert(self.cur_idx + 1, clone)
        self.tree.insert("", self.cur_idx + 1, iid=str(clone.uid),
                         text=clone.label(), image=self._thumb(clone.src_name),
                         values=(self._mp_str(clone.src_name), "0/0"))
        self._save_session()
        self._select_item(self.cur_idx + 1)
        self._set_status("Clone created. Adjust downscale/orientation, then place boxes.")

    # ------------------------------------------------------------- canvas ---

    def _redraw(self, full=False):
        if not self.items:
            return
        item = self.items[self.cur_idx]
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        if full or self.canvas_photo is None:
            work = self._working_image(item)
            ww, wh = work.size
            self.work_dims = (ww, wh)
            ds = min(cw / ww, ch / wh, 1.0)
            self.disp_scale = ds
            dw, dh = max(1, round(ww * ds)), max(1, round(wh * ds))
            self.disp_ox = (cw - dw) // 2
            self.disp_oy = (ch - dh) // 2
            disp_img = work if ds >= 0.9995 else work.resize(
                (dw, dh), Image.Resampling.BILINEAR)
            self.canvas_photo = ImageTk.PhotoImage(disp_img)

        self.canvas.delete("all")
        self.canvas.create_image(self.disp_ox, self.disp_oy, anchor=tk.NW,
                                 image=self.canvas_photo)

        # placed boxes
        for b in item.boxes:
            color = ("#40ff40" if b.exported else "#ffff30")
            if b is self.selected_box:
                color = "#ff4040"
            x0, y0 = self._img_to_canvas(b.x, b.y)
            x1, y1 = self._img_to_canvas(b.x + b.w, b.y + b.h)
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2)
            self.canvas.create_text(x0 + 4, y0 + 4, anchor=tk.NW,
                                    text=f"{b.w}x{b.h}", fill=color,
                                    font=("Consolas", 9, "bold"))

        # ghost box under cursor
        if self.ghost_xy and self.fit_options and self.selected_box is None:
            bw, bh, _t = self.fit_options[self.sel_size_idx]
            gx, gy = self._ghost_topleft(bw, bh)
            x0, y0 = self._img_to_canvas(gx, gy)
            x1, y1 = self._img_to_canvas(gx + bw, gy + bh)
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#30d0ff",
                                         width=2, dash=(5, 3))
            self.canvas.create_text(x0 + 4, y0 + 4, anchor=tk.NW,
                                    text=f"{bw}x{bh}", fill="#30d0ff",
                                    font=("Consolas", 9, "bold"))

    def _img_to_canvas(self, x, y):
        return (self.disp_ox + x * self.disp_scale,
                self.disp_oy + y * self.disp_scale)

    def _canvas_to_img(self, cx, cy):
        return ((cx - self.disp_ox) / self.disp_scale,
                (cy - self.disp_oy) / self.disp_scale)

    def _ghost_topleft(self, bw, bh):
        ix, iy = self._canvas_to_img(*self.ghost_xy)
        ww, wh = self.work_dims
        gx = int(round(ix - bw / 2))
        gy = int(round(iy - bh / 2))
        gx = max(0, min(gx, ww - bw))
        gy = max(0, min(gy, wh - bh))
        return gx, gy

    def _on_motion(self, e):
        self.ghost_xy = (e.x, e.y)
        self._redraw()

    def _on_leave(self, _e):
        self.ghost_xy = None
        self._redraw()

    # ------------------------------------------------------------- boxes ----

    def _on_left_click(self, e):
        self.canvas.focus_set()
        if not self.items or not self.fit_options:
            return
        item = self.items[self.cur_idx]
        if self.selected_box is not None:
            # click while a box is selected: deselect first
            self.selected_box = None
            self._redraw()
            return
        self.ghost_xy = (e.x, e.y)
        bw, bh, _t = self.fit_options[self.sel_size_idx]
        gx, gy = self._ghost_topleft(bw, bh)
        item.boxes.append(Box(bw, bh, gx, gy))
        item.locked = True   # placing a box locks the downscale
        self._refresh_tree_row(item)
        self._save_session()
        self._redraw()
        self._set_status(f"Placed {bw}x{bh} at ({gx},{gy}). Scale locked.")

    def _on_right_click(self, e):
        if not self.items:
            return
        item = self.items[self.cur_idx]
        ix, iy = self._canvas_to_img(e.x, e.y)
        hit = None
        for b in reversed(item.boxes):   # topmost (latest) first
            if b.x <= ix <= b.x + b.w and b.y <= iy <= b.y + b.h:
                hit = b
                break
        self.selected_box = None if hit is self.selected_box else hit
        self._redraw()

    def _deselect_box(self):
        self.selected_box = None
        self._redraw()

    def _delete_selected_box(self):
        if not self.items or self.selected_box is None:
            return
        item = self.items[self.cur_idx]
        if self.selected_box.exported:
            if not messagebox.askyesno(
                    "Delete box",
                    "This box was already exported. Delete it from the "
                    "session anyway? (The exported file is NOT removed.)"):
                return
        item.boxes.remove(self.selected_box)
        self.selected_box = None
        if not item.boxes and not item.is_clone:
            item.locked = False   # original with no boxes may be rescaled again
        self._refresh_tree_row(item)
        self._save_session()
        self._redraw()

    def _nudge(self, dx, dy, step):
        if self.selected_box is None or not self.items:
            return
        item = self.items[self.cur_idx]
        b = self.selected_box
        ww, wh = self.work_dims
        b.x = max(0, min(b.x + dx * step, ww - b.w))
        b.y = max(0, min(b.y + dy * step, wh - b.h))
        if b.exported:
            b.exported = False   # moved after export -> counts as new crop
            self._refresh_tree_row(item)
        self._redraw()

    # ------------------------------------------------------------- export ---

    def _next_serial(self, bucket_dir):
        mx = 0
        if os.path.isdir(bucket_dir):
            for f in os.listdir(bucket_dir):
                m = re.match(r"^(\d+)\.jpe?g$", f, re.IGNORECASE)
                if m:
                    mx = max(mx, int(m.group(1)))
        return mx + 1

    def _export_item(self, item) -> int:
        """Export all un-exported boxes on one item. Returns count written."""
        pending = [b for b in item.boxes if not b.exported]
        if not pending:
            return 0
        work = self._working_image(item)
        if work.mode != "RGB":
            work = work.convert("RGB")
        n_done = 0
        for b in pending:
            crop = work.crop((b.x, b.y, b.x + b.w, b.y + b.h))
            target = self._bucket_target.get((b.w, b.h), "misc")
            bucket_dir = os.path.join(self.out_root, str(target))
            os.makedirs(bucket_dir, exist_ok=True)
            serial = self._next_serial(bucket_dir)
            out_path = os.path.join(bucket_dir, f"{serial:04d}.jpg")
            comment = (f"src={item.src_name}; scale={item.scale:.4f}; "
                       f"method={item.method}; crop_xywh={b.x},{b.y},{b.w},{b.h}")
            exif = Image.Exif()
            exif[0x010E] = comment   # ImageDescription
            crop.save(out_path, format="JPEG", quality=100, subsampling=0,
                      comment=comment.encode("utf-8"), exif=exif)
            b.exported = True
            n_done += 1
        self._refresh_tree_row(item)
        return n_done

    def _export_current(self):
        if not self.items:
            return
        item = self.items[self.cur_idx]
        n = self._export_item(item)
        if n == 0:
            self._set_status("No unexported boxes on this image.")
            return
        self._save_session()
        self._redraw()
        self._set_status(f"Exported {n} crop(s) to {self.out_root}")

    def _export_all(self):
        if not self.items:
            return
        total = 0
        imgs = 0
        for item in self.items:
            n = self._export_item(item)
            if n:
                total += n
                imgs += 1
        if total == 0:
            self._set_status("No unexported boxes anywhere.")
            return
        self._save_session()
        self._redraw()
        self._set_status(
            f"Exported {total} crop(s) across {imgs} image(s) to {self.out_root}")

    # ------------------------------------------------------------- misc -----

    def _set_status(self, msg):
        self.status.configure(
            text=msg + "   |   Wheel: size, L-click: place, R-click: select, "
                       "Del: delete, C: crop, PgUp/PgDn: images")


# ----------------------------------------------------------------------------
# Startup
# ----------------------------------------------------------------------------

def main():
    root = tk.Tk()
    root.withdraw()

    res_str = simpledialog.askstring(
        "Krea2 Crop Tool",
        "Target resolution buckets (comma separated),\n"
        "same values you will use in OneTrainer:",
        initialvalue="512, 768, 1024", parent=root)
    if not res_str:
        sys.exit(0)
    try:
        targets = sorted({int(s.strip()) for s in res_str.split(",") if s.strip()})
        assert targets and all(t > 0 for t in targets)
    except (ValueError, AssertionError):
        messagebox.showerror("Krea2 Crop Tool",
                             f"Could not parse resolutions: {res_str!r}")
        sys.exit(1)

    folder = filedialog.askdirectory(title="Select image folder", parent=root)
    if not folder:
        sys.exit(0)

    root.deiconify()
    app = CropApp(root, folder, targets)
    if not app.items:
        messagebox.showinfo("Krea2 Crop Tool", "No images found in that folder.")
        sys.exit(0)
    root.mainloop()


if __name__ == "__main__":
    main()
