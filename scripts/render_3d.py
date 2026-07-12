"""Render flood simulation results in 3D with PyVista (offscreen).

  # interactive on-screen viewer (rotate/zoom with the mouse)
  python scripts/render_3d.py view --data-dir terrain
  # + water, with a time slider to scrub the storm and space to play/pause
  python scripts/render_3d.py view --data-dir terrain --run sim/run_baseline

  # animated MP4 of one run
  python scripts/render_3d.py video --run output/run_baseline --out output/baseline.mp4

  # static max-depth comparison (baseline vs scenario)
  python scripts/render_3d.py compare --runs output/run_baseline output/run_channel \
      --out output/compare.png

  # 2D max-depth heatmap over the ortho (fast fallback)
  python scripts/render_3d.py heatmap --run output/run_baseline --out output/heat.png
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def load_common(data_dir):
    from PIL import Image
    dem = np.load(os.path.join(data_dir, "dem.npy")).astype(np.float32)
    ortho = np.asarray(Image.open(os.path.join(data_dir, "ortho.png")))
    with open(os.path.join(data_dir, "dem_transform.json")) as f:
        t = json.load(f)
    return dem, ortho, t


def make_terrain(dem, res, z_exagg=1.0):
    import pyvista as pv
    h, w = dem.shape
    x = np.arange(w) * res
    y = np.arange(h) * res
    xx, yy = np.meshgrid(x, y)
    grid = pv.StructuredGrid(xx, yy, dem * z_exagg)
    # texture coordinates for draping the ortho
    grid.active_texture_coordinates = np.column_stack([
        (xx.ravel(order="F") / x.max()),
        1.0 - (yy.ravel(order="F") / y.max()),
    ]).astype(np.float32)
    return grid


def water_mesh(dem, depth, res, z_exagg=1.0, min_depth=0.02):
    import pyvista as pv
    h, w = dem.shape
    x = np.arange(w) * res
    y = np.arange(h) * res
    xx, yy = np.meshgrid(x, y)
    surf = np.where(depth > min_depth, dem + depth, np.nan)
    grid = pv.StructuredGrid(xx, yy, surf * z_exagg)
    grid["depth"] = depth.ravel(order="F")
    return grid.threshold(min_depth, scalars="depth")


def setup_plotter(dem, ortho, res, z_exagg, window=(1600, 1000)):
    import pyvista as pv
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=list(window))
    terrain = make_terrain(dem, res, z_exagg)
    tex = pv.numpy_to_texture(np.ascontiguousarray(ortho[::1]))
    pl.add_mesh(terrain, texture=tex, name="terrain")
    pl.set_background("black")
    return pl


def add_water(pl, dem, depth, res, z_exagg, clim=(0.0, 1.0), render=True):
    wm = water_mesh(dem, depth, res, z_exagg)
    if wm.n_points > 0:
        pl.add_mesh(wm, scalars="depth", cmap="Blues", clim=clim,
                    opacity=0.75, name="water", show_scalar_bar=True,
                    scalar_bar_args={"title": "depth (m)", "color": "white"},
                    render=render)
    return wm


def cmd_view(args):
    import pyvista as pv
    dem, ortho, t = load_common(args.data_dir)
    res = t["res"]

    pl = pv.Plotter()
    terrain = make_terrain(dem, res, args.z_exagg)
    tex = pv.numpy_to_texture(np.ascontiguousarray(ortho))
    pl.add_mesh(terrain, texture=tex, name="terrain")
    pl.set_background("black")
    pl.camera_position = "xy"
    pl.camera.elevation = -55

    if not args.run:
        pl.show()
        return

    frame_paths = sorted(glob.glob(os.path.join(args.run, "depth_*.npy")))
    if not frame_paths:
        sys.exit(f"no depth_*.npy in {args.run}")
    times = [int(os.path.basename(fp).split("_")[1].split(".")[0]) for fp in frame_paths]
    depths = [np.load(fp).astype(np.float32) for fp in frame_paths]

    max_depth = np.load(os.path.join(args.run, "max_depth.npy")).astype(np.float32)
    clim = (0.0, max(0.5, float(max_depth.max()) * 0.8))

    state = {"idx": 0, "playing": False, "speed": 1.0, "progress": 0.0}

    def show_frame(idx):
        # remove_actor() and add_mesh() each render by default - doing both
        # per frame flashed an empty-water frame in between (the "restarts
        # from scratch" flicker). Suppress those and render once at the end.
        idx = max(0, min(len(depths) - 1, idx))
        state["idx"] = idx
        try:
            pl.remove_actor("water", render=False)
        except Exception:
            pass
        add_water(pl, dem, depths[idx], res, args.z_exagg, clim, render=False)
        mm, ss = divmod(times[idx], 60)
        pl.add_text(f"t = {mm:02d}:{ss:02d}", name="clock", color="white",
                    font_size=14, position="upper_right", render=False)
        pl.render()

    def on_time_slider(value):
        idx = min(range(len(times)), key=lambda i: abs(times[i] - value))
        show_frame(idx)

    def on_speed_slider(value):
        state["speed"] = value

    def set_playstate(playing):
        state["playing"] = playing
        pl.add_text("PLAYING" if playing else "PAUSED",
                    name="playstate", color="yellow" if playing else "white",
                    font_size=10, position="lower_right")

    # both sliders live in the header strip (top of the window) so they
    # don't collide with the depth color-scale bar PyVista draws along
    # the bottom for the water mesh's scalar_bar.
    time_slider = pl.add_slider_widget(on_time_slider, rng=[times[0], times[-1]], value=times[0],
                                       title="Time (s)", pointa=(0.25, 0.9), pointb=(0.8, 0.9),
                                       style="modern", interaction_event="always")
    pl.add_slider_widget(on_speed_slider, rng=[0.25, 4.0], value=1.0, title="Speed",
                        pointa=(0.03, 0.9), pointb=(0.2, 0.9), style="modern",
                        interaction_event="always")
    pl.add_checkbox_button_widget(set_playstate, value=False, position=(10, 10),
                                  size=40, color_on="yellow", color_off="grey")
    pl.add_text("Play/Pause", position=(58, 20), font_size=10, color="white")
    pl.add_text("Click button: play/pause    drag sliders: scrub time / speed",
               position="lower_left", font_size=10, color="white")
    set_playstate(False)
    show_frame(0)

    # VTK's own timer/animation machinery (add_timer_event, repeating
    # CreateRepeatingTimer via TimerEvent observers) proved unreliable here -
    # either never firing or badly delayed depending on the interactor.
    # Driving the animation from a plain Python loop around interactive_update
    # sidesteps that: pl.update() both redraws and processes mouse/button
    # input each iteration, and frame advancement is paced by real elapsed
    # wall-clock time, so it self-corrects if a render happens to be slow
    # instead of silently falling behind.
    FRAMES_PER_SEC = 12.0
    pl.show(auto_close=False, interactive_update=True)
    last_t = time.time()
    while not pl._closed:
        now = time.time()
        dt = now - last_t
        last_t = now
        if state["playing"]:
            state["progress"] += dt * state["speed"] * FRAMES_PER_SEC
            if state["progress"] >= 1.0:
                advance = int(state["progress"])
                state["progress"] -= advance
                nxt = state["idx"] + advance
                if nxt >= len(depths):
                    nxt = 0  # loop back to the start of the storm
                show_frame(nxt)
                time_slider.GetRepresentation().SetValue(times[nxt])
        pl.update()
        time.sleep(0.02)


def cmd_video(args):
    import imageio.v2 as imageio
    dem, ortho, t = load_common(args.data_dir)
    if args.dem_override and os.path.exists(args.dem_override):
        dem = np.load(args.dem_override).astype(np.float32)
    res = t["res"]
    frames = sorted(glob.glob(os.path.join(args.run, "depth_*.npy")))
    if not frames:
        sys.exit(f"no depth_*.npy in {args.run}")
    print(f"{len(frames)} frames")

    clim = (0.0, max(0.5, float(np.load(
        os.path.join(args.run, "max_depth.npy")).max()) * 0.8))

    pl = setup_plotter(dem, ortho, res, args.z_exagg)
    pl.camera_position = "xy"
    pl.camera.elevation = -55  # oblique view
    pl.camera.zoom(args.zoom)

    writer = imageio.get_writer(args.out, fps=args.fps, quality=8)
    for i, fp in enumerate(frames):
        depth = np.load(fp).astype(np.float32)
        try:
            pl.remove_actor("water")
        except Exception:
            pass
        add_water(pl, dem, depth, res, args.z_exagg, clim)
        tsec = int(os.path.basename(fp).split("_")[1].split(".")[0])
        pl.add_text(f"t = {tsec // 60:02d}:{tsec % 60:02d}", name="clock",
                    color="white", font_size=14)
        img = pl.screenshot(return_img=True)
        writer.append_data(img)
        print(f"\r  frame {i + 1}/{len(frames)}", end="", flush=True)
    writer.close()
    pl.close()
    print(f"\nwrote {args.out}")


def cmd_compare(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dem, ortho, t = load_common(args.data_dir)
    n = len(args.runs)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 8))
    axes = np.atleast_1d(axes)
    vmax = max(float(np.load(os.path.join(r, "max_depth.npy")).max())
               for r in args.runs)
    vmax = min(vmax, 2.0)
    for ax, r in zip(axes, args.runs):
        md = np.load(os.path.join(r, "max_depth.npy"))
        ax.imshow(ortho)
        im = ax.imshow(np.where(md > 0.05, md, np.nan), cmap="turbo",
                       vmin=0, vmax=vmax, alpha=0.8)
        with open(os.path.join(r, "run_meta.json")) as f:
            meta = json.load(f)
        ax.set_title(f"{os.path.basename(r)}  (rain {meta['rain_mmh']:.0f} mm/h)\n"
                     f"max depth {md.max():.2f} m")
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), label="max water depth (m)", shrink=0.7)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


def cmd_heatmap(args):
    args.runs = [args.run]
    cmd_compare(args)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    vw = sub.add_parser("view")
    vw.add_argument("--data-dir", default="output")
    vw.add_argument("--run", help="optional: overlay max_depth water surface from a sim run")
    vw.add_argument("--z-exagg", type=float, default=1.0)
    vw.set_defaults(func=cmd_view)

    v = sub.add_parser("video")
    v.add_argument("--run", required=True)
    v.add_argument("--data-dir", default="output")
    v.add_argument("--dem-override", help="scenario dem_mod.npy for correct terrain")
    v.add_argument("--out", required=True)
    v.add_argument("--fps", type=int, default=12)
    v.add_argument("--zoom", type=float, default=1.3)
    v.add_argument("--z-exagg", type=float, default=1.0)
    v.set_defaults(func=cmd_video)

    c = sub.add_parser("compare")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--data-dir", default="output")
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_compare)

    hm = sub.add_parser("heatmap")
    hm.add_argument("--run", required=True)
    hm.add_argument("--data-dir", default="output")
    hm.add_argument("--out", required=True)
    hm.set_defaults(func=cmd_heatmap)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
