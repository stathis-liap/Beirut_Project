"""Plug-and-play GUI for the flood pipeline.

Two tabs:
  - Corridor Flood: pick a rain amount, a storm duration, and a terrain
    quality, hit Run - it drives build_dem.py -> flood_sim.py ->
    render_3d.py for you and shows live progress.
  - Stairs (SPH / Navier-Stokes): pick a scene (real Saint Nicolas Stairs
    or the synthetic demo corridor), a duration and flow rate, hit Run -
    it drives particle_sim.py and shows the rendered snapshots.

No command line needed.

Usage:
  python scripts/gui_demo.py
"""

import datetime
import glob
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from PIL import Image, ImageTk

import area_sim
import cad_sim
import map_point
import render_3d
from las_common import find_las_file, pixel_to_utm, utm_to_pixel

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)
DATA_DIR = os.path.join(ROOT, "data")
TERRAIN_DIR = os.path.join(ROOT, "terrain")
SIM_DIR = os.path.join(ROOT, "sim")
OUTPUT_DIR = os.path.join(ROOT, "output")

# (name, resolution in meters, cache dir for that resolution's DEM, estimated
# time for a FRESH build). Runtime grows roughly with 1/res^3 (more cells AND
# a smaller stable timestep), so the high-detail option is capped well short
# of what the raw point cloud could support - a live demo shouldn't sit for
# hours. These fresh-build estimates only apply the first time a given
# quality is used - build_quality_options() below detects when dem.npy is
# already cached and swaps in a realistic ~1-2 min estimate instead (build_dem.py
# is skipped entirely on a cache hit, so quoting the fresh-build time then is
# misleading - a "Medium" run that's actually just flood_sim.py + render on an
# already-built grid takes ~2 min, not the 15-20 min a fresh 1m build needs).
QUALITY_SPECS = [
    ("Low", 2.0, os.path.join(TERRAIN_DIR, "q_low"), "~2-5 min"),
    ("Medium", 1.0, TERRAIN_DIR, "~15-20 min"),
    ("High", 0.75, os.path.join(TERRAIN_DIR, "q_high"), "~45-60 min"),
]


def build_quality_options():
    """QUALITY_SPECS with a time estimate reflecting whether that
    resolution's terrain is already built (skips build_dem.py entirely) or
    would need a fresh build."""
    options = []
    for name, res, cache_dir, fresh_est in QUALITY_SPECS:
        cached = os.path.exists(os.path.join(cache_dir, "dem.npy"))
        time_est = "~1-2 min, cached terrain" if cached else f"{fresh_est} (first build)"
        label = f"{name}  -  {time_est}   ({res:.2g} m grid)"
        options.append((label, res, cache_dir))
    return options

# label -> particle_sim.py --scene value
STAIRS_SCENE_OPTIONS = [
    ("Real: Saint Nicolas Stairs, Beirut", "real_stairs"),
    ("Pick a point on the map", "map_point"),
    ("Synthetic demo corridor", "synthetic"),
]
MAP_POINT_LABEL = STAIRS_SCENE_OPTIONS[1][0]
MAP_CANVAS_W = 520

# relative time weights used to turn per-stage progress into one overall bar
BUILD_WEIGHT = 3
SIM_WEIGHT = 10
HEATMAP_WEIGHT = 1
VIDEO_WEIGHT = 3
STAIRS_SIM_WEIGHT = 10
STAIRS_RENDER_WEIGHT = 1
STAIRS_RENDER_N = 6  # matches particle_render.render_snapshots' default
AREA_TERRAIN_WEIGHT = 5  # streaming the .las to reconstruct local terrain -
                         # runs before either solver, bounded by the SOURCE
                         # file's size not the crop, so it's a real ~15-20s
                         # wait that needs its own share of the bar
AREA_2D_WEIGHT = 5   # the 2D grid solver is cheap regardless of duration -
                     # small share of the bar
AREA_3D_WEIGHT = 10  # the capped SPH run is where the time actually goes
CAD_TESSELLATE_WEIGHT = 6  # tessellating tens of thousands of CAD objects -
                           # the CAD-model equivalent of AREA_TERRAIN_WEIGHT
CAD_2D_WEIGHT = 5
CAD_3D_WEIGHT = 10


def parse_build_progress(line):
    m = re.search(r"pass (\d)/2:\s*([\d.]+)%", line)
    if not m:
        return None
    pass_num, pct = int(m.group(1)), float(m.group(2))
    return min(1.0, ((pass_num - 1) + pct / 100.0) / 2.0)


def parse_scan_progress(line):
    m = re.search(r"scanning:\s*([\d.]+)%", line)
    if not m:
        return None
    return min(1.0, float(m.group(1)) / 100.0)


def parse_tessellate_progress(line):
    m = re.search(r"tessellating:\s*([\d.]+)%", line)
    if not m:
        return None
    return min(1.0, float(m.group(1)) / 100.0)


def parse_snapshot_progress(line, n_snapshots=STAIRS_RENDER_N):
    m = re.search(r"snapshot_(\d+)\.png", line)
    if not m:
        return None
    return min(1.0, (int(m.group(1)) + 1) / n_snapshots)


def parse_sim_progress(line, duration_sec):
    m = re.search(r"t=\s*([\d.]+)s", line)
    if not m:
        return None
    return min(1.0, float(m.group(1)) / duration_sec)


def parse_video_progress(line):
    m = re.search(r"frame (\d+)/(\d+)", line)
    if not m:
        return None
    i, total = int(m.group(1)), int(m.group(2))
    return min(1.0, i / total)


# --------------------------------------------------------------- styling --
FONT_FAMILY = "Courier New"
COLOR_DARK = "#1d3156"     # primary text, log background
COLOR_MID = "#496894"      # secondary buttons, muted/helper text
COLOR_LIGHT = "#a4b5d1"    # app background
COLOR_LIGHTER = "#b0cbe6"  # input fields, inactive tabs, troughs
COLOR_ACCENT = "#fed6ce"   # primary action (Run buttons), progress fill, selected tab


def configure_style(root):
    """Flat, minimalistic look in Courier New over the given blue/peach
    palette. 'clam' is the base ttk theme because it's the one that actually
    honors background/foreground color overrides cross-platform - the
    Windows-native themes ('vista'/'winnative') silently ignore most of
    this and keep OS chrome colors regardless of what's configured."""
    root.configure(background=COLOR_LIGHT)
    root.option_add("*Font", (FONT_FAMILY, 10))

    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(".", background=COLOR_LIGHT, foreground=COLOR_DARK,
                    font=(FONT_FAMILY, 10))
    style.configure("TFrame", background=COLOR_LIGHT)
    style.configure("TLabel", background=COLOR_LIGHT, foreground=COLOR_DARK)
    style.configure("TCheckbutton", background=COLOR_LIGHT, foreground=COLOR_DARK)
    style.map("TCheckbutton", background=[("active", COLOR_LIGHT)])

    # clam theme's "Button.border" element - the one that actually paints
    # the button's visible fill - doesn't have "background" in its option
    # set at all (verified via style.element_options('Button.border'):
    # only bordercolor/lightcolor/darkcolor/relief/borderwidth). "background"
    # only affects Button.label (the text), so configuring just background=
    # left buttons rendering with clam's default gray/border colors instead
    # of the intended fill - bordercolor+lightcolor+darkcolor all have to
    # match background for a flat, evenly-colored button.
    style.configure("TButton", background=COLOR_MID, foreground="white",
                    bordercolor=COLOR_MID, lightcolor=COLOR_MID, darkcolor=COLOR_MID,
                    borderwidth=0, focuscolor=COLOR_MID, padding=6)
    style.map("TButton",
              background=[("active", COLOR_DARK), ("disabled", COLOR_LIGHTER)],
              bordercolor=[("active", COLOR_DARK), ("disabled", COLOR_LIGHTER)],
              lightcolor=[("active", COLOR_DARK), ("disabled", COLOR_LIGHTER)],
              darkcolor=[("active", COLOR_DARK), ("disabled", COLOR_LIGHTER)],
              foreground=[("disabled", COLOR_MID)])

    # primary Run buttons only - see App.build_flood_tab/build_stairs_tab
    style.configure("Accent.TButton", background=COLOR_ACCENT, foreground=COLOR_DARK,
                    bordercolor=COLOR_ACCENT, lightcolor=COLOR_ACCENT, darkcolor=COLOR_ACCENT,
                    borderwidth=0, focuscolor=COLOR_ACCENT, padding=6,
                    font=(FONT_FAMILY, 10, "bold"))
    style.map("Accent.TButton",
              background=[("active", COLOR_LIGHTER), ("disabled", COLOR_LIGHTER)],
              bordercolor=[("active", COLOR_LIGHTER), ("disabled", COLOR_LIGHTER)],
              lightcolor=[("active", COLOR_LIGHTER), ("disabled", COLOR_LIGHTER)],
              darkcolor=[("active", COLOR_LIGHTER), ("disabled", COLOR_LIGHTER)],
              foreground=[("disabled", COLOR_MID)])

    style.configure("TCombobox", fieldbackground="white", background=COLOR_LIGHTER,
                    foreground=COLOR_DARK, arrowcolor=COLOR_DARK, borderwidth=0)
    style.map("TCombobox", fieldbackground=[("readonly", "white")])
    style.configure("TSpinbox", fieldbackground="white", background=COLOR_LIGHTER,
                    foreground=COLOR_DARK, arrowcolor=COLOR_DARK, borderwidth=0)

    style.configure("TNotebook", background=COLOR_LIGHT, borderwidth=0)
    style.configure("TNotebook.Tab", background=COLOR_LIGHTER, foreground=COLOR_DARK,
                    padding=(14, 7), borderwidth=0, font=(FONT_FAMILY, 10))
    style.map("TNotebook.Tab", background=[("selected", COLOR_ACCENT)])

    style.configure("Horizontal.TProgressbar", background=COLOR_ACCENT,
                    troughcolor=COLOR_LIGHTER, borderwidth=0, lightcolor=COLOR_ACCENT,
                    darkcolor=COLOR_ACCENT)
    return style


def make_scrollable_tab(notebook, title):
    """Adds a tab to `notebook` and returns an inner ttk.Frame to build
    content into, transparently scrollable if that content ends up taller
    than the window - see the notebook-building code in App.__init__ for
    why this matters (a plain Frame just clips overflow with no scrollbar
    and no visible indication anything's missing)."""
    outer = ttk.Frame(notebook)
    notebook.add(outer, text=title)

    canvas = tk.Canvas(outer, background=COLOR_LIGHT, highlightthickness=0)
    vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    canvas.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

    inner = ttk.Frame(canvas)
    window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))

    # mousewheel only scrolls this canvas while the pointer is actually
    # over it - a global bind would also hijack scrolling over the Log
    # panel's own ScrolledText
    def on_wheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", on_wheel))
    canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

    inner.scroll_canvas = canvas  # so code that dynamically shows/hides
                                  # content (e.g. on_stairs_scene_change) can
                                  # re-anchor the scroll view to the top -
                                  # otherwise a stale scroll fraction can
                                  # leave a blank gap or hide the new content
    return inner


class App:
    APP_TITLE = "SURF-B: Simulating Urban Runoff and Floods - Beirut"

    def __init__(self, root):
        self.root = root
        root.title(self.APP_TITLE)
        root.geometry("1750x800")
        configure_style(root)

        self.log_q = queue.Queue()
        self.current_proc = None
        self.cancelled = False
        self.run_dir = None
        self.cache_dir = None
        self.heatmap_path = None
        self.video_path = None
        self.stairs_run_dir = None
        self.area_run_dir = None
        self.cad_run_dir = None

        # header strip: logo + app title, spanning the full window width
        # (packed side="top" before the two side="left" panels below, so it
        # sits above everything)
        header = ttk.Frame(root)
        header.pack(side="top", fill="x", padx=8, pady=(8, 0))
        self.logo_photo = None  # keep a reference or Tk garbage-collects the image
        logo_path = os.path.join(ROOT, "assets", "logo.png")
        if os.path.exists(logo_path):
            try:
                logo = Image.open(logo_path)
                target_h = 72
                logo = logo.resize((int(logo.width * target_h / logo.height), target_h),
                                   Image.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(logo)
                ttk.Label(header, image=self.logo_photo).pack(side="left", padx=(0, 10))
            except Exception as e:
                print(f"couldn't load {logo_path}: {e}")
        ttk.Label(header, text=self.APP_TITLE,
                  font=(FONT_FAMILY, 13, "bold")).pack(side="left", pady=4)

        # point-cloud source picker - global, not per-tab: every tab (flood,
        # stairs' map_point/area_rain, 3D area) ultimately reads whichever
        # .las find_las_file() discovers in data/, so browsing to a
        # different file belongs at the app level, not buried in one tab.
        data_frame = ttk.Frame(header)
        data_frame.pack(side="right", padx=(10, 0))
        self.data_file_var = tk.StringVar(value=self._describe_current_las())
        ttk.Label(data_frame, textvariable=self.data_file_var,
                 foreground=COLOR_MID).pack(anchor="e")
        ttk.Button(data_frame, text="Browse for point cloud (.las)...",
                  command=self.browse_for_las).pack(anchor="e", pady=(2, 0))

        # PanedWindow (not a plain side="left" pack) so the boundary between
        # the config panel and the log is a draggable sash, not a fixed
        # split - a wide map + config layout and a useful log width don't
        # always agree on one fixed proportion.
        paned = ttk.Panedwindow(root, orient="horizontal")
        paned.pack(side="top", fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(paned, width=980)
        paned.add(left, weight=0)

        log_frame = ttk.Frame(paned)
        paned.add(log_frame, weight=3)
        ttk.Label(log_frame, text="Log:").pack(anchor="w")
        # dark terminal-style panel - deliberate contrast against the light
        # control panel, using the same palette (peach cursor/selection tie
        # it back together instead of looking like an unrelated widget)
        self.log = scrolledtext.ScrolledText(
            log_frame, font=(FONT_FAMILY, 9), background=COLOR_DARK, foreground=COLOR_LIGHTER,
            insertbackground=COLOR_ACCENT, selectbackground=COLOR_ACCENT,
            selectforeground=COLOR_DARK, relief="flat", borderwidth=0, padx=6, pady=6)
        self.log.pack(fill="both", expand=True)
        self.log.configure(state="disabled")

        notebook = ttk.Notebook(left)
        notebook.pack(fill="both", expand=True)
        # each tab's content (map canvas + several spinbox rows + notes +
        # buttons, esp. on the Stairs/3D Area tabs) can be taller than a
        # reasonably-sized window - a plain Frame just silently clips
        # whatever doesn't fit, hiding the Run button and progress bar
        # entirely with no visible sign anything's wrong. Wrapping each tab
        # in a scrollable canvas fixes that regardless of window/screen
        # size, instead of guessing a tall-enough fixed height.
        flood_tab = make_scrollable_tab(notebook, "Corridor Flood")
        stairs_tab = make_scrollable_tab(notebook, "Stairs (SPH)")
        area_tab = make_scrollable_tab(notebook, "3D Area")
        cad_tab = make_scrollable_tab(notebook, "CAD Model")

        self.build_flood_tab(flood_tab)
        self.build_stairs_tab(stairs_tab)
        self.build_area_tab(area_tab)
        self.build_cad_tab(cad_tab)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.status_var, wraplength=780).pack(fill="x", padx=4, pady=6)

        prog_row = ttk.Frame(left)
        prog_row.pack(fill="x", padx=4, pady=6)
        self.progress = ttk.Progressbar(prog_row, mode="determinate", maximum=100)
        self.progress.pack(fill="x")
        self.progress_label = ttk.Label(prog_row, text="0%")
        self.progress_label.pack(anchor="e")

        self.root.after(100, self.poll_log)

    # -------------------------------------------------------- point cloud --
    def _describe_current_las(self):
        path = find_las_file(DATA_DIR)
        return f"Using: {os.path.basename(path)}" if path else "No .las file in data/ yet."

    def browse_for_las(self):
        path = filedialog.askopenfilename(
            title="Select a .las point cloud",
            filetypes=[("LAS point cloud", "*.las"), ("All files", "*.*")])
        if not path:
            return
        if os.path.normcase(os.path.abspath(os.path.dirname(path))) == \
                os.path.normcase(os.path.abspath(DATA_DIR)):
            self.append_log(f"\n{path} is already in data/ - using it as-is.")
            self.data_file_var.set(self._describe_current_las())
            return

        dest = os.path.join(DATA_DIR, os.path.basename(path))
        if os.path.exists(dest) and not messagebox.askyesno(
                "File exists", f"{os.path.basename(path)} already exists in data/. Overwrite?"):
            return

        self.status_var.set(f"Bringing {os.path.basename(path)} into data/ ...")
        self.append_log(f"\n> importing {path} -> {dest}")
        threading.Thread(target=self._import_las_file, args=(path, dest), daemon=True).start()

    def _import_las_file(self, src, dest):
        # A hardlink is instant and costs no extra disk space, but only
        # works within the same volume - this project's .las files run into
        # the tens of GB (see README), so falling back to shutil.copy2 (the
        # cross-volume case) can genuinely take minutes; both run off the
        # main thread so the GUI doesn't freeze either way.
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            same_drive = (os.path.splitdrive(os.path.abspath(src))[0].lower() ==
                         os.path.splitdrive(os.path.abspath(dest))[0].lower())
            if same_drive:
                try:
                    if os.path.exists(dest):
                        os.remove(dest)
                    os.link(src, dest)
                    self.log_q.put(("LOG", f"linked {dest} -> {src} (same drive, no copy needed)"))
                except OSError as e:
                    self.log_q.put(("LOG", f"hardlink failed ({e}), copying instead..."))
                    shutil.copy2(src, dest)
                    self.log_q.put(("LOG", f"copied {src} -> {dest}"))
            else:
                shutil.copy2(src, dest)
                self.log_q.put(("LOG", f"copied {src} -> {dest}"))
            self.log_q.put(("LAS_READY", None))
        except Exception as e:
            self.log_q.put(("ERROR", f"Couldn't bring {src} into data/: {e}"))

    # ------------------------------------------------------------ flood tab --
    def build_flood_tab(self, parent):
        pad = {"padx": 4, "pady": 6}
        form = ttk.Frame(parent)
        form.pack(fill="x", **pad)
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Rain intensity (mm/h):").grid(row=0, column=0, columnspan=2, sticky="w")
        self.rain_var = tk.DoubleVar(value=30.0)
        ttk.Spinbox(form, from_=5, to=150, increment=5, textvariable=self.rain_var,
                    width=10).grid(row=1, column=0, columnspan=2, sticky="w")

        ttk.Label(form, text="Storm duration (minutes):").grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.duration_var = tk.DoubleVar(value=60.0)
        ttk.Spinbox(form, from_=10, to=180, increment=10, textvariable=self.duration_var,
                    width=10).grid(row=3, column=0, columnspan=2, sticky="w")

        ttk.Label(form, text="Terrain quality:").grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.quality_options = build_quality_options()
        self.quality_var = tk.StringVar(value=self.quality_options[1][0])
        self.quality_combo = ttk.Combobox(form, textvariable=self.quality_var, state="readonly",
                                          values=[q[0] for q in self.quality_options])
        self.quality_combo.grid(row=5, column=0, columnspan=2, sticky="we")

        self.video_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="Also render 3D flyover video",
                        variable=self.video_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))

        btns = ttk.Frame(parent)
        btns.pack(fill="x", **pad)
        self.flood_run_btn = ttk.Button(btns, text="Run simulation", command=self.on_run_flood,
                                        style="Accent.TButton")
        self.flood_run_btn.pack(fill="x")
        self.flood_cancel_btn = ttk.Button(btns, text="Cancel", command=self.on_cancel, state="disabled")
        self.flood_cancel_btn.pack(fill="x", pady=(4, 0))

        out_btns = ttk.Frame(parent)
        out_btns.pack(fill="x", **pad)
        self.open_folder_btn = ttk.Button(out_btns, text="Open results folder",
                                          command=lambda: self.open_path(self.run_dir),
                                          state="disabled")
        self.open_folder_btn.pack(fill="x")
        self.open_heatmap_btn = ttk.Button(out_btns, text="Open heatmap",
                                           command=lambda: self.open_path(self.heatmap_path),
                                           state="disabled")
        self.open_heatmap_btn.pack(fill="x", pady=(4, 0))
        self.open_video_btn = ttk.Button(out_btns, text="Open video",
                                         command=lambda: self.open_path(self.video_path),
                                         state="disabled")
        self.open_video_btn.pack(fill="x", pady=(4, 0))

        radius_row = ttk.Frame(out_btns)
        radius_row.pack(fill="x", pady=(8, 0))
        ttk.Label(radius_row, text="Ctrl+Click rain-simulation radius (km):").pack(anchor="w")
        self.reconstruct_radius_var = tk.DoubleVar(value=render_3d.RECONSTRUCT_RADIUS_KM)
        ttk.Spinbox(radius_row, from_=0.1, to=10.0, increment=0.1,
                    textvariable=self.reconstruct_radius_var, width=8).pack(anchor="w", pady=(2, 0))
        ttk.Label(radius_row, text="In the 3D view: Ctrl+Click a spot to pop a rain-drop SPH "
                 "simulation of that area (uses the rain intensity above).",
                 wraplength=700, foreground=COLOR_MID).pack(anchor="w", pady=(2, 0))

        self.view_water_btn = ttk.Button(
            out_btns, text="Open 3D View (terrain + water)",
            command=lambda: self.open_3d_view(with_water=True), state="disabled")
        self.view_water_btn.pack(fill="x", pady=(4, 0))
        self.view_terrain_btn = ttk.Button(
            out_btns, text="Open 3D View (terrain only)",
            command=lambda: self.open_3d_view(with_water=False), state="disabled")
        self.view_terrain_btn.pack(fill="x", pady=(4, 0))

    # ----------------------------------------------------------- stairs tab --
    def build_stairs_tab(self, parent):
        pad = {"padx": 4, "pady": 6}
        columns = ttk.Frame(parent)
        columns.pack(fill="both", expand=True, **pad)

        # map | config, same three-column shape (map | parameters+run | log,
        # log being the app-wide third column) as the 3D Area tab - built
        # here but only packed in by on_stairs_scene_change() for the
        # map_point scene, since the other scenes have no point to pick.
        self.stairs_utm_x, self.stairs_utm_y = None, None
        self.map_thumb_meta, self.map_thumb_photo, self.map_thumb_scale = None, None, 1.0
        self.stairs_map_frame = ttk.Frame(columns)
        ttk.Label(self.stairs_map_frame, text="Click a point on the map:").pack(anchor="w")
        self.stairs_map_canvas = tk.Canvas(self.stairs_map_frame, width=MAP_CANVAS_W,
                                           height=int(MAP_CANVAS_W * 0.7), background="black",
                                           highlightthickness=0)
        self.stairs_map_canvas.pack(pady=(2, 4))
        self.stairs_map_canvas.bind("<Button-1>", self.on_map_click)
        self.stairs_point_var = tk.StringVar(value="No point selected yet.")
        ttk.Label(self.stairs_map_frame, textvariable=self.stairs_point_var,
                 foreground=COLOR_MID).pack(anchor="w")

        # exact-coordinate entry, kept in sync both ways with clicking the
        # map - typing here moves the marker, clicking the map fills these in
        ttk.Label(self.stairs_map_frame, text="...or enter UTM coordinates exactly:",
                 foreground=COLOR_MID).pack(anchor="w", pady=(6, 0))
        coord_row = ttk.Frame(self.stairs_map_frame)
        coord_row.pack(fill="x", pady=(2, 0))
        ttk.Label(coord_row, text="X:").pack(side="left")
        self.stairs_utm_x_var = tk.StringVar()
        ttk.Entry(coord_row, textvariable=self.stairs_utm_x_var, width=11).pack(
            side="left", padx=(2, 8))
        ttk.Label(coord_row, text="Y:").pack(side="left")
        self.stairs_utm_y_var = tk.StringVar()
        ttk.Entry(coord_row, textvariable=self.stairs_utm_y_var, width=11).pack(
            side="left", padx=(2, 0))
        ttk.Button(self.stairs_map_frame, text="Go to coordinates",
                  command=self.on_stairs_coord_entry).pack(fill="x", pady=(4, 0))

        crop_row = ttk.Frame(self.stairs_map_frame)
        crop_row.pack(fill="x", pady=(8, 0))
        ttk.Label(crop_row, text="Crop size (m):").pack(side="left")
        self.stairs_crop_size_var = tk.DoubleVar(value=30.0)
        ttk.Spinbox(crop_row, from_=10, to=100, increment=5, textvariable=self.stairs_crop_size_var,
                    width=8).pack(side="left", padx=(6, 0))
        # not packed yet - on_stairs_scene_change() shows/hides it
        self.stairs_scroll_canvas = getattr(parent, "scroll_canvas", None)

        config_col = ttk.Frame(columns)
        config_col.pack(side="left", fill="both", expand=True)

        form = ttk.Frame(config_col)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Scene:").grid(row=0, column=0, columnspan=2, sticky="w")
        self.stairs_scene_var = tk.StringVar(value=STAIRS_SCENE_OPTIONS[0][0])
        scene_combo = ttk.Combobox(form, textvariable=self.stairs_scene_var, state="readonly",
                                   values=[s[0] for s in STAIRS_SCENE_OPTIONS])
        scene_combo.grid(row=1, column=0, columnspan=2, sticky="we")
        scene_combo.bind("<<ComboboxSelected>>", self.on_stairs_scene_change)

        ttk.Label(form, text="Sim duration (seconds):").grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.stairs_duration_var = tk.DoubleVar(value=15.0)
        ttk.Spinbox(form, from_=2, to=120, increment=1, textvariable=self.stairs_duration_var,
                    width=10).grid(row=3, column=0, columnspan=2, sticky="w")

        ttk.Label(form, text="Flow rate (m3/s):").grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.stairs_rate_var = tk.DoubleVar(value=0.15)
        ttk.Spinbox(form, from_=0.01, to=1.0, increment=0.01, textvariable=self.stairs_rate_var,
                    width=10).grid(row=5, column=0, columnspan=2, sticky="w")

        self.stairs_render_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(form, text="Render snapshots (PNG)",
                        variable=self.stairs_render_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))

        note = ("Weakly-compressible SPH solver (numerical solution of the "
                "Navier-Stokes momentum equation) - GPU via Taichi, thousands "
                "of water particles falling down real stair geometry.")
        self.stairs_note_label = ttk.Label(config_col, text=note, wraplength=340, foreground=COLOR_MID)
        self.stairs_note_label.pack(fill="x", pady=(8, 6))

        btns = ttk.Frame(config_col)
        btns.pack(fill="x")
        self.stairs_run_btn = ttk.Button(btns, text="Run simulation", command=self.on_run_stairs,
                                         style="Accent.TButton")
        self.stairs_run_btn.pack(fill="x")
        self.stairs_cancel_btn = ttk.Button(btns, text="Cancel", command=self.on_cancel, state="disabled")
        self.stairs_cancel_btn.pack(fill="x", pady=(4, 0))

        out_btns = ttk.Frame(config_col)
        out_btns.pack(fill="x", pady=(6, 0))
        self.stairs_open_folder_btn = ttk.Button(
            out_btns, text="Open results folder",
            command=lambda: self.open_path(self.stairs_run_dir), state="disabled")
        self.stairs_open_folder_btn.pack(fill="x")
        self.stairs_open_snapshot_btn = ttk.Button(
            out_btns, text="Open last snapshot", command=self.open_last_snapshot, state="disabled")
        self.stairs_open_snapshot_btn.pack(fill="x", pady=(4, 0))
        self.stairs_view_btn = ttk.Button(
            out_btns, text="Open 3D View (interactive)",
            command=self.open_stairs_3d_view, state="disabled")
        self.stairs_view_btn.pack(fill="x", pady=(4, 0))

        self.stairs_config_col = config_col

    def on_stairs_scene_change(self, event=None):
        if self.stairs_scene_var.get() == MAP_POINT_LABEL:
            self.stairs_map_frame.pack(side="left", fill="y", padx=(0, 14),
                                       before=self.stairs_config_col)
            if self.map_thumb_meta is None:
                self.load_map_thumbnail()
        else:
            self.stairs_map_frame.pack_forget()
        self._reset_scroll(self.stairs_scroll_canvas)

    def _reset_scroll(self, canvas):
        """Re-anchors a scrollable tab's view to the top after its content
        changes size (e.g. the map picker being shown/hidden) - otherwise a
        scroll position computed for the OLD content height can leave a
        blank gap or hide whatever's now at the top."""
        if canvas is None:
            return
        canvas.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.yview_moveto(0)

    def load_map_thumbnail(self):
        try:
            path, meta = map_point.build_thumbnail()
        except FileNotFoundError as e:
            messagebox.showerror("Map data not found", str(e))
            self.stairs_scene_var.set(STAIRS_SCENE_OPTIONS[0][0])
            self.stairs_map_frame.pack_forget()
            return
        self.map_thumb_meta = meta
        img = Image.open(path)
        self.map_thumb_scale = MAP_CANVAS_W / img.width
        disp = img.resize((MAP_CANVAS_W, int(img.height * self.map_thumb_scale)), Image.LANCZOS)
        self.stairs_map_canvas.configure(height=disp.height)
        self.map_thumb_photo = ImageTk.PhotoImage(disp)
        self.stairs_map_canvas.delete("all")
        self.stairs_map_canvas.create_image(0, 0, anchor="nw", image=self.map_thumb_photo)

    def on_map_click(self, event):
        if not self.map_thumb_meta:
            return
        px, py = event.x / self.map_thumb_scale, event.y / self.map_thumb_scale
        x, y = map_point.thumb_pixel_to_utm(self.map_thumb_meta, px, py)
        self._set_stairs_point(x, y)

    def on_stairs_coord_entry(self):
        if not self.map_thumb_meta:
            messagebox.showinfo("Map not loaded", "Select the map_point scene first.")
            return
        try:
            x = float(self.stairs_utm_x_var.get())
            y = float(self.stairs_utm_y_var.get())
        except ValueError:
            messagebox.showerror("Invalid coordinates", "UTM X and Y must be numbers.")
            return
        self._set_stairs_point(x, y)

    def _set_stairs_point(self, x, y):
        # single path for both "clicked the map" and "typed exact
        # coordinates" - keeps the marker, the label, and the two entry
        # boxes all in sync regardless of which one the user just used
        self.stairs_utm_x, self.stairs_utm_y = x, y
        self.stairs_utm_x_var.set(f"{x:.2f}")
        self.stairs_utm_y_var.set(f"{y:.2f}")
        self.stairs_point_var.set(f"Selected: UTM ({x:.1f}, {y:.1f})  EPSG:32636")
        self.stairs_map_canvas.delete("marker")
        thumb_px, thumb_py = map_point.utm_to_thumb_pixel(self.map_thumb_meta, x, y)
        cx, cy = thumb_px * self.map_thumb_scale, thumb_py * self.map_thumb_scale
        w, h = self.stairs_map_canvas.winfo_width(), self.stairs_map_canvas.winfo_height()
        if 0 <= cx <= w and 0 <= cy <= h:
            r = 5
            self.stairs_map_canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                               outline=COLOR_ACCENT, width=2, tags="marker")
        else:
            self.append_log(f"\n({x:.1f}, {y:.1f}) is outside the visible map preview - "
                            "point is still set, just not shown on the thumbnail.")

    # ------------------------------------------------------------ 3D area tab --
    def build_area_tab(self, parent):
        pad = {"padx": 4, "pady": 6}
        columns = ttk.Frame(parent)
        columns.pack(fill="both", expand=True, **pad)

        map_col = ttk.Frame(columns)
        map_col.pack(side="left", fill="y", padx=(0, 14))
        config_col = ttk.Frame(columns)
        config_col.pack(side="left", fill="both", expand=True)

        # --- map column ---
        ttk.Label(map_col, text="Click a point on the corridor map:").pack(anchor="w")
        self.area_map_canvas = tk.Canvas(map_col, width=MAP_CANVAS_W,
                                         height=int(MAP_CANVAS_W * 0.7),
                                         background="black", highlightthickness=0)
        self.area_map_canvas.pack(pady=(2, 4))
        self.area_map_canvas.bind("<Button-1>", self.on_area_map_click)
        self.area_utm_x = self.area_utm_y = None
        self.area_transform = None
        self.area_thumb_photo = None
        self.area_thumb_scale = 1.0
        self.area_point_var = tk.StringVar(value="No point selected yet.")
        ttk.Label(map_col, textvariable=self.area_point_var, foreground=COLOR_MID).pack(anchor="w")

        ttk.Label(map_col, text="...or enter UTM coordinates exactly:",
                 foreground=COLOR_MID).pack(anchor="w", pady=(6, 0))
        coord_row = ttk.Frame(map_col)
        coord_row.pack(fill="x", pady=(2, 0))
        ttk.Label(coord_row, text="X:").pack(side="left")
        self.area_utm_x_var = tk.StringVar()
        ttk.Entry(coord_row, textvariable=self.area_utm_x_var, width=11).pack(
            side="left", padx=(2, 8))
        ttk.Label(coord_row, text="Y:").pack(side="left")
        self.area_utm_y_var = tk.StringVar()
        ttk.Entry(coord_row, textvariable=self.area_utm_y_var, width=11).pack(
            side="left", padx=(2, 0))
        ttk.Button(map_col, text="Go to coordinates",
                  command=self.on_area_coord_entry).pack(fill="x", pady=(4, 0))

        # --- config column ---
        form = ttk.Frame(config_col)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Area radius (m):").grid(row=0, column=0, columnspan=2, sticky="w")
        self.area_radius_var = tk.DoubleVar(value=area_sim.DEFAULT_RADIUS_M)
        ttk.Spinbox(form, from_=10, to=100, increment=5, textvariable=self.area_radius_var,
                    width=10).grid(row=1, column=0, columnspan=2, sticky="w")

        ttk.Label(form, text="Rain intensity (mm/h):").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.area_rain_var = tk.DoubleVar(value=30.0)
        ttk.Spinbox(form, from_=5, to=150, increment=5, textvariable=self.area_rain_var,
                    width=10).grid(row=3, column=0, columnspan=2, sticky="w")

        ttk.Label(form, text="2D grid duration (minutes):").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.area_duration_var = tk.DoubleVar(value=15.0)
        ttk.Spinbox(form, from_=1, to=180, increment=1, textvariable=self.area_duration_var,
                    width=10).grid(row=5, column=0, columnspan=2, sticky="w")

        ttk.Label(form, text="3D particle duration (seconds):").grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.area_particle_duration_var = tk.DoubleVar(value=area_sim.DEFAULT_PARTICLE_DURATION_S)
        ttk.Spinbox(form, from_=3, to=30, increment=1, textvariable=self.area_particle_duration_var,
                    width=10).grid(row=7, column=0, columnspan=2, sticky="w")

        note = ("Runs BOTH solvers on the exact same small area: the 2D grid for the "
                "full duration above, and the 3D SPH particle engine for a short, "
                "separately-capped window (SPH can't run a full storm at any usable "
                "frame rate). Toggle between the two afterward in one viewer. Keep "
                "the radius small - that's what makes the particle count actually "
                "read as water instead of scattered dust.")
        ttk.Label(config_col, text=note, wraplength=340, foreground=COLOR_MID).pack(
            fill="x", pady=(8, 6))

        btns = ttk.Frame(config_col)
        btns.pack(fill="x")
        self.area_run_btn = ttk.Button(btns, text="Run 3D area simulation",
                                       command=self.on_run_area, style="Accent.TButton")
        self.area_run_btn.pack(fill="x")
        self.area_cancel_btn = ttk.Button(btns, text="Cancel", command=self.on_cancel,
                                          state="disabled")
        self.area_cancel_btn.pack(fill="x", pady=(4, 0))

        out_btns = ttk.Frame(config_col)
        out_btns.pack(fill="x", pady=(6, 0))
        self.area_open_folder_btn = ttk.Button(
            out_btns, text="Open results folder",
            command=lambda: self.open_path(self.area_run_dir), state="disabled")
        self.area_open_folder_btn.pack(fill="x")
        self.area_view_btn = ttk.Button(
            out_btns, text="Open 2D / 3D toggle viewer",
            command=self.open_area_view, state="disabled")
        self.area_view_btn.pack(fill="x", pady=(4, 0))

        self.load_area_thumbnail()

    def load_area_thumbnail(self):
        ortho_path = os.path.join(TERRAIN_DIR, "ortho.png")
        transform_path = os.path.join(TERRAIN_DIR, "dem_transform.json")
        if not (os.path.exists(ortho_path) and os.path.exists(transform_path)):
            self.area_point_var.set("No terrain built yet - run a Corridor Flood "
                                    "simulation (Medium quality) first.")
            return
        with open(transform_path) as f:
            self.area_transform = json.load(f)
        img = Image.open(ortho_path)
        self.area_thumb_scale = MAP_CANVAS_W / img.width
        disp = img.resize((MAP_CANVAS_W, int(img.height * self.area_thumb_scale)), Image.LANCZOS)
        self.area_map_canvas.configure(height=disp.height)
        self.area_thumb_photo = ImageTk.PhotoImage(disp)
        self.area_map_canvas.delete("all")
        self.area_map_canvas.create_image(0, 0, anchor="nw", image=self.area_thumb_photo)

    def on_area_map_click(self, event):
        if not self.area_transform:
            return
        px, py = event.x / self.area_thumb_scale, event.y / self.area_thumb_scale
        x, y = pixel_to_utm(self.area_transform, px, py)
        self._set_area_point(float(x), float(y))

    def on_area_coord_entry(self):
        if not self.area_transform:
            messagebox.showinfo("No terrain built yet", "Run a Corridor Flood simulation "
                                "(Medium quality) first so there's a map to place this on.")
            return
        try:
            x = float(self.area_utm_x_var.get())
            y = float(self.area_utm_y_var.get())
        except ValueError:
            messagebox.showerror("Invalid coordinates", "UTM X and Y must be numbers.")
            return
        self._set_area_point(x, y)

    def _set_area_point(self, x, y):
        # single path for both "clicked the map" and "typed exact
        # coordinates" - keeps the marker, the label, and the two entry
        # boxes all in sync regardless of which one the user just used
        self.area_utm_x, self.area_utm_y = x, y
        self.area_utm_x_var.set(f"{x:.2f}")
        self.area_utm_y_var.set(f"{y:.2f}")
        self.area_point_var.set(f"Selected: UTM ({x:.1f}, {y:.1f})  EPSG:32636")
        self.area_map_canvas.delete("marker")
        col, row = utm_to_pixel(self.area_transform, x, y)
        cx, cy = col * self.area_thumb_scale, row * self.area_thumb_scale
        w, h = self.area_map_canvas.winfo_width(), self.area_map_canvas.winfo_height()
        if 0 <= cx <= w and 0 <= cy <= h:
            r = 5
            self.area_map_canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                             outline=COLOR_ACCENT, width=2, tags="marker")
        else:
            self.append_log(f"\n({x:.1f}, {y:.1f}) is outside the visible corridor map - "
                            "point is still set, just not shown on the thumbnail.")

    def on_run_area(self):
        if self.area_utm_x is None:
            messagebox.showerror("No point selected", "Click a point on the map first.")
            return
        try:
            radius_m = float(self.area_radius_var.get())
            rain = float(self.area_rain_var.get())
            duration_min = float(self.area_duration_var.get())
            particle_duration = float(self.area_particle_duration_var.get())
            if radius_m <= 0 or rain <= 0 or duration_min <= 0 or particle_duration <= 0:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid input", "All fields must be positive numbers.")
            return
        self.start_job(self.worker_area, (self.area_utm_x, self.area_utm_y, radius_m, rain,
                                          duration_min * 60, particle_duration))

    def worker_area(self, utm_x, utm_y, radius_m, rain_mmh, duration_s, particle_duration_s):
        try:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(OUTPUT_DIR, f"area3d_{stamp}")
            total_weight = AREA_TERRAIN_WEIGHT + AREA_2D_WEIGHT + AREA_3D_WEIGHT
            stage = {"n": "terrain"}

            # area_sim.py runs three phases in one process: stream the .las
            # to reconstruct local terrain (prints "scanning: NN.N%"), then
            # the 2D grid solver, then the capped 3D SPH solver (both print
            # "t=...s" lines, reused here via parse_sim_progress, re-based
            # against whichever stage's own duration is currently active).
            # All three run strictly sequentially, so a marker string in the
            # log line is enough to detect the handoff between stages.
            def parser(line):
                if "2D grid solver" in line:
                    stage["n"] = "2d"
                    self.log_q.put(("STATUS", f"2D grid solver ({duration_s:.0f}s)..."))
                    return AREA_TERRAIN_WEIGHT / total_weight
                if "3D SPH solver" in line:
                    stage["n"] = "3d"
                    self.log_q.put(("STATUS", f"3D SPH solver ({particle_duration_s:.0f}s, capped)..."))
                    return (AREA_TERRAIN_WEIGHT + AREA_2D_WEIGHT) / total_weight
                if stage["n"] == "terrain":
                    sub = parse_scan_progress(line)
                    return None if sub is None else (AREA_TERRAIN_WEIGHT * sub) / total_weight
                dur = duration_s if stage["n"] == "2d" else particle_duration_s
                sub = parse_sim_progress(line, dur)
                if sub is None:
                    return None
                base = AREA_TERRAIN_WEIGHT if stage["n"] == "2d" else AREA_TERRAIN_WEIGHT + AREA_2D_WEIGHT
                weight = AREA_2D_WEIGHT if stage["n"] == "2d" else AREA_3D_WEIGHT
                return (base + weight * sub) / total_weight

            self.log_q.put(("STATUS", "Building local terrain (streaming .las)..."))
            cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "area_sim.py"),
                  "--utm-x", str(utm_x), "--utm-y", str(utm_y), "--radius", str(radius_m),
                  "--rain", str(rain_mmh), "--duration", str(duration_s),
                  "--particle-duration", str(particle_duration_s), "--out", out_dir]
            rc = self.run_step(cmd, ROOT, base_weight=0, weight=1, total_weight=1, parser=parser)
            if self.fail_if_error(rc, "3D area simulation failed - see log above."):
                return

            self.log_q.put(("DONE_AREA", out_dir))
        except Exception as e:
            self.log_q.put(("ERROR", f"Unexpected error: {e}"))

    def open_area_view(self):
        if not self.area_run_dir or not os.path.exists(self.area_run_dir):
            messagebox.showinfo("No run yet", "Run a 3D area simulation first.")
            return
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "render_3d.py"), "area-view",
              "--dir", self.area_run_dir]
        self.launch_viewer(cmd, "area-view")

    # ------------------------------------------------------------- CAD tab --
    def build_cad_tab(self, parent):
        pad = {"padx": 4, "pady": 6}
        form = ttk.Frame(parent)
        form.pack(fill="x", **pad)
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="CAD model file:").grid(row=0, column=0, columnspan=2, sticky="w")
        self.cad_model_path = None
        self.cad_model_var = tk.StringVar(value="No file selected.")
        ttk.Label(form, textvariable=self.cad_model_var, foreground=COLOR_MID,
                 wraplength=700).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Button(form, text="Browse for CAD model...", command=self.browse_cad_model).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(2, 8))

        ttk.Label(form, text="Grid resolution (m/cell):").grid(row=3, column=0, columnspan=2, sticky="w")
        self.cad_res_var = tk.DoubleVar(value=0.4)
        ttk.Spinbox(form, from_=0.1, to=5.0, increment=0.1, textvariable=self.cad_res_var,
                    width=10).grid(row=4, column=0, columnspan=2, sticky="w")

        ttk.Label(form, text="Rain intensity (mm/h):").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.cad_rain_var = tk.DoubleVar(value=30.0)
        ttk.Spinbox(form, from_=5, to=150, increment=5, textvariable=self.cad_rain_var,
                    width=10).grid(row=6, column=0, columnspan=2, sticky="w")

        ttk.Label(form, text="2D grid duration (minutes):").grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.cad_duration_var = tk.DoubleVar(value=10.0)
        ttk.Spinbox(form, from_=1, to=180, increment=1, textvariable=self.cad_duration_var,
                    width=10).grid(row=8, column=0, columnspan=2, sticky="w")

        ttk.Label(form, text="3D particle duration (seconds):").grid(
            row=9, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.cad_particle_duration_var = tk.DoubleVar(value=cad_sim.DEFAULT_PARTICLE_DURATION_S)
        ttk.Spinbox(form, from_=3, to=30, increment=1, textvariable=self.cad_particle_duration_var,
                    width=10).grid(row=10, column=0, columnspan=2, sticky="w")

        note = ("Loads a Rhino .3dm CAD model, rasterizes it into a terrain the same way "
                "an area scan does (see load_cad_model.py), then runs both solvers on it. "
                "Only .3dm is supported here. .dwg technically converts (ODA File "
                "Converter is installed), but if its 3D content is ACIS/ASM solid bodies "
                "(common for AutoCAD-authored buildings) there's no vertex data to "
                "extract without a real ACIS kernel - export the solids to a mesh format "
                "(.obj, .3dm) from the CAD tool that made them instead. .skp needs "
                "Trimble's SketchUp SDK, which has no working Python binding. Not a "
                "real-world location - the model keeps its own local coordinates, same "
                "category as the Stairs tab's synthetic/real-stairs scenes.")
        ttk.Label(parent, text=note, wraplength=700, foreground=COLOR_MID).pack(
            fill="x", padx=8, pady=(0, 6))

        btns = ttk.Frame(parent)
        btns.pack(fill="x", **pad)
        self.cad_run_btn = ttk.Button(btns, text="Load model + run simulation",
                                      command=self.on_run_cad, style="Accent.TButton")
        self.cad_run_btn.pack(fill="x")
        self.cad_cancel_btn = ttk.Button(btns, text="Cancel", command=self.on_cancel,
                                         state="disabled")
        self.cad_cancel_btn.pack(fill="x", pady=(4, 0))

        out_btns = ttk.Frame(parent)
        out_btns.pack(fill="x", **pad)
        self.cad_open_folder_btn = ttk.Button(
            out_btns, text="Open results folder",
            command=lambda: self.open_path(self.cad_run_dir), state="disabled")
        self.cad_open_folder_btn.pack(fill="x")
        self.cad_view_btn = ttk.Button(
            out_btns, text="Open 2D / 3D toggle viewer",
            command=self.open_cad_view, state="disabled")
        self.cad_view_btn.pack(fill="x", pady=(4, 0))

    def browse_cad_model(self):
        path = filedialog.askopenfilename(
            title="Select a CAD model",
            filetypes=[("Rhino 3DM / AutoCAD DWG", "*.3dm;*.dwg"),
                      ("SketchUp SKP (not supported)", "*.skp"),
                      ("All files", "*.*")])
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext == ".skp":
            messagebox.showerror(
                "Format not supported",
                ".skp needs Trimble's SketchUp SDK, which has no working Python "
                "binding - this app can't read it. Export to .3dm, .dxf, or .obj "
                "instead, or pick a .3dm/.dwg file.")
            return
        if ext not in (".3dm", ".dwg"):
            messagebox.showerror("Format not supported",
                                 f"{ext} isn't one of the supported formats (.3dm, .dwg).")
            return
        if ext == ".dwg":
            self.append_log(
                "\nNote: .dwg support depends on the model's content - buildings modeled "
                "as ACIS solid bodies (common for AutoCAD) have no extractable vertex "
                "data and will fail with a clear coverage error; buildings modeled as "
                "meshes/polylines work fine. See load_dwg_model.py's docstring.")
        self.cad_model_path = path
        self.cad_model_var.set(path)

    def on_run_cad(self):
        if not self.cad_model_path:
            messagebox.showerror("No model selected", "Browse for a .3dm file first.")
            return
        try:
            res = float(self.cad_res_var.get())
            rain = float(self.cad_rain_var.get())
            duration_min = float(self.cad_duration_var.get())
            particle_duration = float(self.cad_particle_duration_var.get())
            if res <= 0 or rain <= 0 or duration_min <= 0 or particle_duration <= 0:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid input", "All fields must be positive numbers.")
            return
        self.start_job(self.worker_cad, (self.cad_model_path, res, rain,
                                         duration_min * 60, particle_duration))

    def worker_cad(self, model_path, res, rain_mmh, duration_s, particle_duration_s):
        try:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(OUTPUT_DIR, f"cad3d_{stamp}")
            total_weight = CAD_TESSELLATE_WEIGHT + CAD_2D_WEIGHT + CAD_3D_WEIGHT
            stage = {"n": "tessellate"}

            def parser(line):
                if "2D grid solver" in line:
                    stage["n"] = "2d"
                    self.log_q.put(("STATUS", f"2D grid solver ({duration_s:.0f}s)..."))
                    return CAD_TESSELLATE_WEIGHT / total_weight
                if "3D SPH solver" in line:
                    stage["n"] = "3d"
                    self.log_q.put(("STATUS", f"3D SPH solver ({particle_duration_s:.0f}s, capped)..."))
                    return (CAD_TESSELLATE_WEIGHT + CAD_2D_WEIGHT) / total_weight
                if stage["n"] == "tessellate":
                    sub = parse_tessellate_progress(line)
                    return None if sub is None else (CAD_TESSELLATE_WEIGHT * sub) / total_weight
                dur = duration_s if stage["n"] == "2d" else particle_duration_s
                sub = parse_sim_progress(line, dur)
                if sub is None:
                    return None
                base = CAD_TESSELLATE_WEIGHT if stage["n"] == "2d" else CAD_TESSELLATE_WEIGHT + CAD_2D_WEIGHT
                weight = CAD_2D_WEIGHT if stage["n"] == "2d" else CAD_3D_WEIGHT
                return (base + weight * sub) / total_weight

            self.log_q.put(("STATUS", "Loading CAD model (tessellating geometry)..."))
            cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "cad_sim.py"),
                  "--model", model_path, "--res", str(res), "--rain", str(rain_mmh),
                  "--duration", str(duration_s), "--particle-duration", str(particle_duration_s),
                  "--out", out_dir]
            rc = self.run_step(cmd, ROOT, base_weight=0, weight=1, total_weight=1, parser=parser)
            if self.fail_if_error(rc, "CAD model simulation failed - see log above."):
                return

            self.log_q.put(("DONE_CAD", out_dir))
        except Exception as e:
            self.log_q.put(("ERROR", f"Unexpected error: {e}"))

    def open_cad_view(self):
        if not self.cad_run_dir or not os.path.exists(self.cad_run_dir):
            messagebox.showinfo("No run yet", "Load a model and run a simulation first.")
            return
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "render_3d.py"), "area-view",
              "--dir", self.cad_run_dir]
        self.launch_viewer(cmd, "cad-view")

    def open_last_snapshot(self):
        if not self.stairs_run_dir:
            return
        shots = sorted(glob.glob(os.path.join(self.stairs_run_dir, "snapshot_*.png")))
        if shots:
            self.open_path(shots[-1])
        else:
            messagebox.showinfo("No snapshots", "This run didn't render snapshots "
                                "(the 'Render snapshots' checkbox was off).")

    def open_stairs_3d_view(self):
        if not self.stairs_run_dir or not os.path.exists(self.stairs_run_dir):
            messagebox.showinfo("No run yet", "Run a stairs simulation first.")
            return
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "particle_render.py"),
              "--run", self.stairs_run_dir, "--view"]
        self.launch_viewer(cmd, "stairs-view")

    def open_3d_view(self, with_water):
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            messagebox.showinfo("No terrain yet", "Run a simulation first to build the terrain.")
            return
        try:
            radius_km = float(self.reconstruct_radius_var.get())
            if radius_km <= 0:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid input", "Reconstruction radius must be a positive number.")
            return
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "render_3d.py"), "view",
              "--data-dir", self.cache_dir, "--reconstruct-radius", str(radius_km),
              "--reconstruct-rain", str(self.rain_var.get())]
        if with_water and self.run_dir:
            cmd += ["--run", self.run_dir]
        self.launch_viewer(cmd, "flood-view")

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def open_path(self, path):
        if path and os.path.exists(path):
            os.startfile(path)

    def launch_viewer(self, cmd, label):
        """Launch an interactive 3D view as its own OS window (true in-app
        embedding isn't available here - the pip-installed VTK build doesn't
        ship vtkRenderingTk, so vtkTkRenderWindowInteractor can't load; that
        would need a different VTK build or switching this whole GUI from
        Tkinter to Qt, both bigger changes than "add a view"). This is the
        practical middle ground: launched from the GUI, and unlike a bare
        fire-and-forget subprocess, its console output streams live into
        this window's own Log panel - tagged so it's distinguishable from
        the Run/build log - instead of vanishing into a window with no
        visible console.
        """
        self.append_log(f"\n> [{label}] " + " ".join(cmd))
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)

        def pump():
            for line in proc.stdout:
                self.log_q.put(("LOG", f"[{label}] {line.rstrip()}"))
            proc.wait()
            self.log_q.put(("LOG", f"[{label}] (window closed)"))

        threading.Thread(target=pump, daemon=True).start()

    def set_running(self, running):
        state = "disabled" if running else "normal"
        self.flood_run_btn.configure(state=state)
        self.stairs_run_btn.configure(state=state)
        self.area_run_btn.configure(state=state)
        self.cad_run_btn.configure(state=state)
        cancel_state = "normal" if running else "disabled"
        self.flood_cancel_btn.configure(state=cancel_state)
        self.stairs_cancel_btn.configure(state=cancel_state)
        self.area_cancel_btn.configure(state=cancel_state)
        self.cad_cancel_btn.configure(state=cancel_state)
        if running:
            for b in (self.open_folder_btn, self.open_heatmap_btn, self.open_video_btn,
                     self.view_water_btn, self.view_terrain_btn,
                     self.stairs_open_folder_btn, self.stairs_open_snapshot_btn,
                     self.stairs_view_btn, self.area_open_folder_btn, self.area_view_btn,
                     self.cad_open_folder_btn, self.cad_view_btn):
                b.configure(state="disabled")

    def start_job(self, target, args):
        # drain any stale messages left over from a previous job's tail end -
        # log_q is fed from a background thread, so without this a leftover
        # PROGRESS/LOG message queued just before this job started could get
        # processed just after, briefly showing the old run's state (e.g. a
        # trailing "100%" from the last job) instead of the fresh reset below
        try:
            while True:
                self.log_q.get_nowait()
        except queue.Empty:
            pass
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.cancelled = False
        self.set_running(True)
        self.status_var.set("Running...")
        self.progress["value"] = 0
        self.progress_label.configure(text="0%")
        threading.Thread(target=target, args=args, daemon=True).start()

    # ---------------------------------------------------------- flood run --
    def on_run_flood(self):
        try:
            rain = float(self.rain_var.get())
            duration_min = float(self.duration_var.get())
            if rain <= 0 or duration_min <= 0:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid input", "Rain and duration must be positive numbers.")
            return

        las_path = find_las_file(DATA_DIR)
        if not las_path:
            messagebox.showerror("Missing point cloud",
                                 f"Can't find any .las file in {DATA_DIR}.\n\n"
                                 "Drop one there - any .las works, plug and play.")
            return

        label = self.quality_var.get()
        res, cache_dir = next((r, d) for lbl, r, d in self.quality_options if lbl == label)
        make_video = self.video_var.get()
        self.start_job(self.worker_flood, (rain, duration_min, res, cache_dir, make_video, las_path))

    def on_cancel(self):
        # without this flag, terminating the subprocess makes run_step return
        # a non-zero code same as a real failure, so the worker's own
        # `if rc != 0: ... ERROR` check fired right after this "Cancelled."
        # status - two contradictory messages for one user action
        self.cancelled = True
        if self.current_proc is not None:
            self.current_proc.terminate()
        self.log_q.put(("STATUS", "Cancelled."))
        self.log_q.put(("ENABLE", None))

    def fail_if_error(self, rc, msg):
        """True if the worker should stop. Suppresses `msg` when this was a
        user-initiated Cancel - on_cancel already pushed its own "Cancelled."
        status, so terminating the subprocess makes run_step return non-zero
        same as a real failure would; without this check both a "Cancelled."
        and a contradictory "... failed" message showed up for one click."""
        if rc == 0:
            return False
        if not self.cancelled:
            self.log_q.put(("ERROR", msg))
        return True

    def run_step(self, cmd, cwd, base_weight=0, weight=0, total_weight=1, parser=None):
        self.log_q.put(("LOG", "\n> " + " ".join(cmd)))
        # PYTHONUNBUFFERED: without it, a child's stdout is fully block-buffered
        # (not line-buffered) whenever it's not a tty - which a pipe never is.
        # bufsize=1 below only affects how *we* read the pipe; it does nothing
        # about how the child itself writes to it. Unfixed, this is exactly
        # "progress bar sits at 0% then jumps to 100%": every print() the child
        # makes queues up in its own buffer and only actually flushes to us
        # when that buffer fills or the process exits, so none of the
        # incremental parser() progress ever fires until it's already done.
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        self.current_proc = proc

        def handle_line(line):
            line = line.rstrip()
            if not line:
                return
            self.log_q.put(("LOG", line))
            if parser is not None:
                frac = parser(line)
                if frac is not None:
                    pct = min(100.0, (base_weight + weight * frac) / total_weight * 100)
                    self.log_q.put(("PROGRESS", pct))

        # Not `for line in proc.stdout:` - that only splits on \n, so a
        # single-line, carriage-return-updated progress meter (e.g.
        # reconstruct_area.py's "\r  scanning: NN.N%", flushed every chunk)
        # never yields ANYTHING to this loop until a real \n eventually
        # shows up - the entire scan (~15-20s) reads as silence, then dumps
        # all its buffered updates at once. Reading char-by-char and
        # splitting on \r as well as \n turns every one of those updates
        # into its own line the moment it's written, same as the child's
        # own terminal output would look.
        buf = ""
        while True:
            ch = proc.stdout.read(1)
            if ch == "":
                break
            if ch in ("\r", "\n"):
                handle_line(buf)
                buf = ""
            else:
                buf += ch
        handle_line(buf)
        proc.wait()
        self.current_proc = None
        if proc.returncode == 0:
            pct = min(100.0, (base_weight + weight) / total_weight * 100)
            self.log_q.put(("PROGRESS", pct))
        return proc.returncode

    def worker_flood(self, rain, duration_min, res, cache_dir, make_video, las_path):
        try:
            duration_sec = duration_min * 60
            save_every = max(10, round(duration_sec / 120 / 5) * 5)
            quality_key = os.path.basename(cache_dir) if cache_dir != TERRAIN_DIR else "medium"
            run_name = f"run_rain{int(rain)}_dur{int(duration_min)}_{quality_key}"
            run_dir = os.path.join(SIM_DIR, run_name)

            dem_path = os.path.join(cache_dir, "dem.npy")
            need_build = not os.path.exists(dem_path)

            total_weight = (BUILD_WEIGHT if need_build else 0) + SIM_WEIGHT + HEATMAP_WEIGHT
            if make_video:
                total_weight += VIDEO_WEIGHT
            done = 0.0

            if need_build:
                self.log_q.put(("STATUS", f"Building terrain at {res} m resolution..."))
                rc = self.run_step([sys.executable, os.path.join(SCRIPTS_DIR, "build_dem.py"),
                                    las_path, "--res", str(res), "--out", cache_dir], ROOT,
                                   base_weight=done, weight=BUILD_WEIGHT, total_weight=total_weight,
                                   parser=parse_build_progress)
                if self.fail_if_error(rc, "Terrain build failed - see log above."):
                    return
                done += BUILD_WEIGHT
            else:
                self.log_q.put(("LOG", f"Reusing cached terrain at {res} m ({cache_dir})"))

            self.log_q.put(("STATUS", f"Simulating {rain:.0f} mm/h for {duration_min:.0f} min..."))
            rc = self.run_step([sys.executable, os.path.join(SCRIPTS_DIR, "flood_sim.py"),
                                "--dem", dem_path,
                                "--transform", os.path.join(cache_dir, "dem_transform.json"),
                                "--rain", str(rain), "--duration", str(duration_sec),
                                "--save-every", str(save_every), "--out", run_dir], ROOT,
                               base_weight=done, weight=SIM_WEIGHT, total_weight=total_weight,
                               parser=lambda line: parse_sim_progress(line, duration_sec))
            if self.fail_if_error(rc, "Simulation failed - see log above."):
                return
            done += SIM_WEIGHT

            self.log_q.put(("STATUS", "Rendering heatmap..."))
            heatmap_path = os.path.join(run_dir, "heatmap.png")
            rc = self.run_step([sys.executable, os.path.join(SCRIPTS_DIR, "render_3d.py"),
                                "heatmap", "--run", run_dir, "--data-dir", cache_dir,
                                "--out", heatmap_path], ROOT,
                               base_weight=done, weight=HEATMAP_WEIGHT, total_weight=total_weight)
            if self.fail_if_error(rc, "Heatmap render failed - see log above."):
                return
            done += HEATMAP_WEIGHT

            video_path = None
            if make_video:
                self.log_q.put(("STATUS", "Rendering 3D flyover video..."))
                video_path = os.path.join(run_dir, "clip.mp4")
                rc = self.run_step([sys.executable, os.path.join(SCRIPTS_DIR, "render_3d.py"),
                                    "video", "--run", run_dir, "--data-dir", cache_dir,
                                    "--out", video_path], ROOT,
                                   base_weight=done, weight=VIDEO_WEIGHT, total_weight=total_weight,
                                   parser=parse_video_progress)
                if self.fail_if_error(rc, "Video render failed - see log above."):
                    return
                done += VIDEO_WEIGHT

            meta_path = os.path.join(run_dir, "run_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                self.log_q.put(("LOG",
                    f"\nmass balance: rain {meta['vol_rain_m3']:.0f} m3, "
                    f"infiltrated {meta['vol_infiltrated_m3']:.0f}, "
                    f"drained {meta['vol_sunk_m3']:.0f}, "
                    f"stored at end {meta['vol_stored_end_m3']:.0f} m3"))

            self.log_q.put(("DONE_FLOOD", (run_dir, heatmap_path, video_path, cache_dir)))
        except Exception as e:
            self.log_q.put(("ERROR", f"Unexpected error: {e}"))

    # --------------------------------------------------------- stairs run --
    def on_run_stairs(self):
        try:
            duration_sec = float(self.stairs_duration_var.get())
            rate = float(self.stairs_rate_var.get())
            if duration_sec <= 0 or rate <= 0:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid input", "Duration and flow rate must be positive numbers.")
            return

        label = self.stairs_scene_var.get()
        scene = next(key for lbl, key in STAIRS_SCENE_OPTIONS if lbl == label)
        crop_size = None
        if scene == "map_point":
            if self.stairs_utm_x is None:
                messagebox.showerror("No point selected", "Click a point on the map first.")
                return
            # was outside the try/except above - a non-numeric crop size
            # crashed uncaught into Tkinter's callback handler instead of
            # showing a clean error like every other input here does
            try:
                crop_size = float(self.stairs_crop_size_var.get())
                if crop_size <= 0:
                    raise ValueError
            except (ValueError, tk.TclError):
                messagebox.showerror("Invalid input", "Crop size must be a positive number.")
                return
        render = self.stairs_render_var.get()
        self.start_job(self.worker_stairs,
                       (scene, duration_sec, rate, render, self.stairs_utm_x, self.stairs_utm_y, crop_size))

    def worker_stairs(self, scene, duration_sec, rate, render, utm_x=None, utm_y=None, crop_size=None):
        try:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(OUTPUT_DIR, f"particles_{scene}_{stamp}")
            total_weight = STAIRS_SIM_WEIGHT + (STAIRS_RENDER_WEIGHT if render else 0)

            # sim and render used to be one bundled `particle_sim.py --render`
            # call - the progress parser only ever matches "t=X.XXs" lines, so
            # once the sim loop ended the bar sat at 100% for however long the
            # (separate, un-tracked) snapshot rendering still took afterward.
            # Splitting into two run_step calls, same pattern worker_flood
            # already uses for build/sim/heatmap, gives each stage its own
            # slice of the bar instead.
            self.log_q.put(("STATUS", f"Simulating {scene} for {duration_sec:.0f}s "
                            f"(rate {rate:.2f} m3/s)..."))
            cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "particle_sim.py"),
                  "--scene", scene, "--duration", str(duration_sec),
                  "--rate", str(rate), "--out", run_dir]
            if scene == "map_point":
                cmd += ["--utm-x", str(utm_x), "--utm-y", str(utm_y), "--crop-size", str(crop_size)]

            rc = self.run_step(cmd, ROOT, base_weight=0, weight=STAIRS_SIM_WEIGHT,
                               total_weight=total_weight,
                               parser=lambda line: parse_sim_progress(line, duration_sec))
            if self.fail_if_error(rc, "Stairs simulation failed - see log above."):
                return

            if render:
                self.log_q.put(("STATUS", "Rendering snapshots..."))
                rc = self.run_step([sys.executable, os.path.join(SCRIPTS_DIR, "particle_render.py"),
                                    "--run", run_dir, "--n", str(STAIRS_RENDER_N)], ROOT,
                                   base_weight=STAIRS_SIM_WEIGHT, weight=STAIRS_RENDER_WEIGHT,
                                   total_weight=total_weight, parser=parse_snapshot_progress)
                if self.fail_if_error(rc, "Snapshot render failed - see log above."):
                    return

            self.log_q.put(("DONE_STAIRS", run_dir))
        except Exception as e:
            self.log_q.put(("ERROR", f"Unexpected error: {e}"))

    def poll_log(self):
        try:
            while True:
                kind, payload = self.log_q.get_nowait()
                if kind == "LOG":
                    self.append_log(payload)
                elif kind == "PROGRESS":
                    self.progress["value"] = payload
                    self.progress_label.configure(text=f"{payload:.0f}%")
                elif kind == "STATUS":
                    self.status_var.set(payload)
                    self.append_log(f"\n--- {payload} ---")
                elif kind == "ERROR":
                    self.append_log(f"\nERROR: {payload}")
                    self.status_var.set("Failed - see log.")
                    self.set_running(False)
                elif kind == "ENABLE":
                    self.set_running(False)
                elif kind == "DONE_FLOOD":
                    run_dir, heatmap_path, video_path, cache_dir = payload
                    self.run_dir = run_dir
                    self.heatmap_path = heatmap_path
                    self.video_path = video_path
                    self.cache_dir = cache_dir
                    self.progress["value"] = 100
                    self.progress_label.configure(text="100%")
                    self.status_var.set(f"Done -> {run_dir}")
                    self.append_log(f"\n--- Done -> {run_dir} ---")
                    self.set_running(False)
                    self.open_folder_btn.configure(state="normal")
                    self.open_heatmap_btn.configure(
                        state="normal" if heatmap_path and os.path.exists(heatmap_path) else "disabled")
                    self.open_video_btn.configure(
                        state="normal" if video_path and os.path.exists(video_path) else "disabled")
                    self.view_water_btn.configure(state="normal")
                    self.view_terrain_btn.configure(state="normal")
                    # terrain may have just been freshly built - refresh the
                    # "cached" vs "first build" time estimates, keeping the
                    # same quality selected (matched by cache_dir, not label,
                    # since the label text itself is what's changing)
                    self.quality_options = build_quality_options()
                    self.quality_combo.configure(values=[q[0] for q in self.quality_options])
                    match = next((lbl for lbl, r, cd in self.quality_options if cd == cache_dir), None)
                    if match:
                        self.quality_var.set(match)
                elif kind == "DONE_STAIRS":
                    run_dir = payload
                    self.stairs_run_dir = run_dir
                    self.progress["value"] = 100
                    self.progress_label.configure(text="100%")
                    self.status_var.set(f"Done -> {run_dir}")
                    self.append_log(f"\n--- Done -> {run_dir} ---")
                    self.set_running(False)
                    self.stairs_open_folder_btn.configure(state="normal")
                    has_snapshots = bool(glob.glob(os.path.join(run_dir, "snapshot_*.png")))
                    self.stairs_open_snapshot_btn.configure(
                        state="normal" if has_snapshots else "disabled")
                    has_frames = bool(glob.glob(os.path.join(run_dir, "particles_*.npz")))
                    self.stairs_view_btn.configure(state="normal" if has_frames else "disabled")
                elif kind == "DONE_AREA":
                    run_dir = payload
                    self.area_run_dir = run_dir
                    self.progress["value"] = 100
                    self.progress_label.configure(text="100%")
                    self.status_var.set(f"Done -> {run_dir}")
                    self.append_log(f"\n--- Done -> {run_dir} ---")
                    self.set_running(False)
                    self.area_open_folder_btn.configure(state="normal")
                    self.area_view_btn.configure(state="normal")
                elif kind == "DONE_CAD":
                    run_dir = payload
                    self.cad_run_dir = run_dir
                    self.progress["value"] = 100
                    self.progress_label.configure(text="100%")
                    self.status_var.set(f"Done -> {run_dir}")
                    self.append_log(f"\n--- Done -> {run_dir} ---")
                    self.set_running(False)
                    self.cad_open_folder_btn.configure(state="normal")
                    self.cad_view_btn.configure(state="normal")
                elif kind == "LAS_READY":
                    self.data_file_var.set(self._describe_current_las())
                    self.status_var.set("Ready.")
        except queue.Empty:
            pass
        self.root.after(100, self.poll_log)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
