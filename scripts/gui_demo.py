"""Plug-and-play GUI for the flood pipeline.

Pick a rain amount, a storm duration, and a terrain quality, hit Run -
it drives build_dem.py -> flood_sim.py -> render_3d.py for you and shows
live progress. No command line needed.

Usage:
  python scripts/gui_demo.py
"""

import glob
import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)
LAS_PATH = os.path.join(ROOT, "terrain", "corridor.las")
TERRAIN_DIR = os.path.join(ROOT, "terrain")
SIM_DIR = os.path.join(ROOT, "sim")

# label -> (resolution in meters, cache dir for that resolution's DEM)
# "Medium" reuses the terrain/ folder that's already built at 1.0 m.
# Runtime grows roughly with 1/res^3 (more cells AND a smaller stable
# timestep), so the high-detail option is capped well short of what
# the raw point cloud could support - a live demo shouldn't sit for hours.
QUALITY_OPTIONS = [
    ("Low  -  fast, ~2-5 min   (2.0 m grid)", 2.0, os.path.join(TERRAIN_DIR, "q_low")),
    ("Medium  -  ~15-20 min   (1.0 m grid)", 1.0, TERRAIN_DIR),
    ("High  -  slow, ~45-60 min   (0.75 m grid)", 0.75, os.path.join(TERRAIN_DIR, "q_high")),
]

# relative time weights used to turn per-stage progress into one overall bar
BUILD_WEIGHT = 3
SIM_WEIGHT = 10
HEATMAP_WEIGHT = 1
VIDEO_WEIGHT = 3


def parse_build_progress(line):
    m = re.search(r"pass (\d)/2:\s*([\d.]+)%", line)
    if not m:
        return None
    pass_num, pct = int(m.group(1)), float(m.group(2))
    return min(1.0, ((pass_num - 1) + pct / 100.0) / 2.0)


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


class App:
    def __init__(self, root):
        self.root = root
        root.title("Beirut Corridor Flood Simulator")
        root.geometry("900x600")

        self.log_q = queue.Queue()
        self.current_proc = None
        self.run_dir = None
        self.cache_dir = None
        self.heatmap_path = None
        self.video_path = None

        left = ttk.Frame(root, width=300)
        left.pack(side="left", fill="y", padx=8, pady=8)
        left.pack_propagate(False)

        log_frame = ttk.Frame(root)
        log_frame.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=8)
        ttk.Label(log_frame, text="Log:").pack(anchor="w")
        self.log = scrolledtext.ScrolledText(log_frame, font=("Consolas", 9))
        self.log.pack(fill="both", expand=True)
        self.log.configure(state="disabled")

        pad = {"padx": 4, "pady": 6}

        form = ttk.Frame(left)
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
        self.quality_var = tk.StringVar(value=QUALITY_OPTIONS[1][0])
        combo = ttk.Combobox(form, textvariable=self.quality_var, state="readonly",
                              values=[q[0] for q in QUALITY_OPTIONS])
        combo.grid(row=5, column=0, columnspan=2, sticky="we")

        self.video_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="Also render 3D flyover video",
                        variable=self.video_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))

        btns = ttk.Frame(left)
        btns.pack(fill="x", **pad)
        self.run_btn = ttk.Button(btns, text="Run simulation", command=self.on_run)
        self.run_btn.pack(fill="x")
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self.on_cancel, state="disabled")
        self.cancel_btn.pack(fill="x", pady=(4, 0))

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.status_var, wraplength=280).pack(fill="x", **pad)

        prog_row = ttk.Frame(left)
        prog_row.pack(fill="x", **pad)
        self.progress = ttk.Progressbar(prog_row, mode="determinate", maximum=100)
        self.progress.pack(fill="x")
        self.progress_label = ttk.Label(prog_row, text="0%")
        self.progress_label.pack(anchor="e")

        out_btns = ttk.Frame(left)
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
        self.view_water_btn = ttk.Button(
            out_btns, text="Open 3D View (terrain + water)",
            command=lambda: self.open_3d_view(with_water=True), state="disabled")
        self.view_water_btn.pack(fill="x", pady=(4, 0))
        self.view_terrain_btn = ttk.Button(
            out_btns, text="Open 3D View (terrain only)",
            command=lambda: self.open_3d_view(with_water=False), state="disabled")
        self.view_terrain_btn.pack(fill="x", pady=(4, 0))

        self.root.after(100, self.poll_log)

    def open_3d_view(self, with_water):
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            messagebox.showinfo("No terrain yet", "Run a simulation first to build the terrain.")
            return
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "render_3d.py"), "view",
              "--data-dir", self.cache_dir]
        if with_water and self.run_dir:
            cmd += ["--run", self.run_dir]
        subprocess.Popen(cmd, cwd=ROOT)

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def open_path(self, path):
        if path and os.path.exists(path):
            os.startfile(path)

    def set_running(self, running):
        state = "disabled" if running else "normal"
        self.run_btn.configure(state=state)
        self.cancel_btn.configure(state=("normal" if running else "disabled"))
        for b in (self.open_folder_btn, self.open_heatmap_btn, self.open_video_btn,
                 self.view_water_btn, self.view_terrain_btn):
            if running:
                b.configure(state="disabled")

    def on_run(self):
        try:
            rain = float(self.rain_var.get())
            duration_min = float(self.duration_var.get())
            if rain <= 0 or duration_min <= 0:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid input", "Rain and duration must be positive numbers.")
            return

        if not os.path.exists(LAS_PATH):
            messagebox.showerror("Missing file", f"Can't find {LAS_PATH}")
            return

        label = self.quality_var.get()
        res, cache_dir = next((r, d) for lbl, r, d in QUALITY_OPTIONS if lbl == label)
        make_video = self.video_var.get()

        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.set_running(True)
        self.status_var.set("Running...")
        self.progress["value"] = 0
        self.progress_label.configure(text="0%")

        threading.Thread(target=self.worker, args=(rain, duration_min, res, cache_dir, make_video),
                         daemon=True).start()

    def on_cancel(self):
        if self.current_proc is not None:
            self.current_proc.terminate()
        self.log_q.put(("STATUS", "Cancelled."))
        self.log_q.put(("ENABLE", None))

    def run_step(self, cmd, cwd, base_weight=0, weight=0, total_weight=1, parser=None):
        self.log_q.put(("LOG", "\n> " + " ".join(cmd)))
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.current_proc = proc
        for line in proc.stdout:
            self.log_q.put(("LOG", line.rstrip()))
            if parser is not None:
                frac = parser(line)
                if frac is not None:
                    pct = min(100.0, (base_weight + weight * frac) / total_weight * 100)
                    self.log_q.put(("PROGRESS", pct))
        proc.wait()
        self.current_proc = None
        if proc.returncode == 0:
            pct = min(100.0, (base_weight + weight) / total_weight * 100)
            self.log_q.put(("PROGRESS", pct))
        return proc.returncode

    def worker(self, rain, duration_min, res, cache_dir, make_video):
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
                                    LAS_PATH, "--res", str(res), "--out", cache_dir], ROOT,
                                   base_weight=done, weight=BUILD_WEIGHT, total_weight=total_weight,
                                   parser=parse_build_progress)
                if rc != 0:
                    self.log_q.put(("ERROR", "Terrain build failed - see log above."))
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
            if rc != 0:
                self.log_q.put(("ERROR", "Simulation failed - see log above."))
                return
            done += SIM_WEIGHT

            self.log_q.put(("STATUS", "Rendering heatmap..."))
            heatmap_path = os.path.join(run_dir, "heatmap.png")
            rc = self.run_step([sys.executable, os.path.join(SCRIPTS_DIR, "render_3d.py"),
                                "heatmap", "--run", run_dir, "--data-dir", cache_dir,
                                "--out", heatmap_path], ROOT,
                               base_weight=done, weight=HEATMAP_WEIGHT, total_weight=total_weight)
            if rc != 0:
                self.log_q.put(("ERROR", "Heatmap render failed - see log above."))
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
                if rc != 0:
                    self.log_q.put(("ERROR", "Video render failed - see log above."))
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

            self.log_q.put(("DONE", (run_dir, heatmap_path, video_path, cache_dir)))
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
                elif kind == "DONE":
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
        except queue.Empty:
            pass
        self.root.after(100, self.poll_log)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
