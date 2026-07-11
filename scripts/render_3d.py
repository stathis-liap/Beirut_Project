#!/usr/bin/env python3
"""Render flood simulation results in 3D with PyVista (offscreen).

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


def add_water(pl, dem, depth, res, z_exagg, clim=(0.0, 1.0)):
    wm = water_mesh(dem, depth, res, z_exagg)
    if wm.n_points > 0:
        pl.add_mesh(wm, scalars="depth", cmap="Blues", clim=clim,
                    opacity=0.75, name="water", show_scalar_bar=True,
                    scalar_bar_args={"title": "depth (m)", "color": "white"})
    return wm


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
