#!/usr/bin/env python3
"""Lightweight, on-demand 3D reconstruction of a small area around a point,
streamed directly from a .las point cloud - the "click a point in the flood
view, get a focused close-up" feature (see render_3d.py's Ctrl+Left-click
handler in cmd_view, which launches this).

Deliberately NOT the full build_dem.py pipeline: one streaming pass (no
second refinement pass), no flow-accumulation/hillshade QA outputs, and the
crop itself is bounded to a small area around the point - the point is to
pop up fast for a "what does this specific spot actually look like" look,
not to match the main corridor DEM's fidelity or run a fresh flood sim
there.

"Radius" is approximated as a square crop (2*radius across) rather than an
exact circle - visually near-identical for this purpose, and much simpler
than maintaining a persistent circular mask through hole-filling.

Usage:
  python scripts/reconstruct_area.py --las data/corridor.las \
      --utm-x 733200 --utm-y 3753400 --radius 1500
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
from las_common import LasHeader, iter_chunks, save_transform, resolve_las_path
from build_dem import fill_holes, despeckle

DEFAULT_RADIUS_M = 1500.0
TARGET_GRID_DIM = 1200   # auto-picked resolution keeps the grid around this
                          # many cells per axis regardless of radius, so a
                          # bigger crop doesn't mean a much heavier one


def auto_res(radius):
    return round(max(0.5, (2 * radius) / TARGET_GRID_DIM), 2)


def grid_local(header, cx, cy, radius, res):
    minx, maxx = cx - radius, cx + radius
    miny, maxy = cy - radius, cy + radius
    w = int(np.ceil((maxx - minx) / res))
    h = int(np.ceil((maxy - miny) / res))

    zmin = np.full(h * w, np.inf, dtype=np.float32)
    count = np.zeros(h * w, dtype=np.int64)
    rgb_sum = np.zeros((h * w, 3), dtype=np.float64)

    n = header.n_points_in_file
    n_kept = 0
    t0 = time.time()
    for i, pts in iter_chunks(header, chunk_points=10_000_000):
        x, y = header.scale_xy(pts)
        m = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
        if m.any():
            idx_m = np.flatnonzero(m)
            xf, yf = x[idx_m], y[idx_m]
            ptsf = pts[idx_m]
            z = (ptsf["Z"] * header.sz + header.oz).astype(np.float32)
            col = np.clip(((xf - minx) / res).astype(np.int64), 0, w - 1)
            row = np.clip(((maxy - yf) / res).astype(np.int64), 0, h - 1)
            idx = row * w + col
            np.minimum.at(zmin, idx, z)
            np.add.at(count, idx, 1)
            for ci, ch in enumerate(("red", "green", "blue")):
                np.add.at(rgb_sum[:, ci], idx, ptsf[ch].astype(np.float64))
            n_kept += len(ptsf)
        print(f"\r  scanning: {100 * (i + len(pts)) / n:5.1f}%  ({n_kept:,} pts kept)",
              end="", flush=True)
    print(f"\n  {n_kept:,} points in the crop, {time.time() - t0:.0f}s")

    covered = count > 0
    dem = np.where(covered, zmin, np.nan).reshape(h, w)
    rgb = np.zeros((h * w, 3))
    rgb[covered] = rgb_sum[covered] / count[covered, None]
    if rgb.size and rgb.max() > 255:
        rgb /= 256.0
    ortho = np.clip(rgb, 0, 255).astype(np.uint8).reshape(h, w, 3)
    return dem, ortho, covered.reshape(h, w), minx, maxy


def build(las_path, utm_x, utm_y, radius, res, out_dir):
    header = LasHeader(las_path)
    print(header.describe())
    if not (header.minx <= utm_x <= header.maxx and header.miny <= utm_y <= header.maxy):
        print(f"WARNING: ({utm_x:.1f}, {utm_y:.1f}) looks outside this LAS's extent "
              f"(X {header.minx:.1f}-{header.maxx:.1f}, Y {header.miny:.1f}-{header.maxy:.1f}) "
              "- it may come back empty.")

    dem, ortho, covered, minx, maxy = grid_local(header, utm_x, utm_y, radius, res)
    if not covered.any():
        sys.exit("no points found in this area - try a different point or a bigger radius")

    print("filling holes / despeckling...")
    dem = fill_holes(dem)
    dem = despeckle(dem)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "dem.npy"), dem.astype(np.float32))
    Image.fromarray(ortho).save(os.path.join(out_dir, "ortho.png"))
    h, w = dem.shape
    save_transform(os.path.join(out_dir, "dem_transform.json"), minx, maxy - h * res, res, w, h,
                   extra={"source_las": os.path.abspath(las_path),
                          "center_utm": [utm_x, utm_y], "radius_m": radius})
    print(f"wrote {out_dir}/dem.npy, ortho.png, dem_transform.json "
          f"({w}x{h} @ {res}m, {2 * radius:.0f}m across)")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--las", help="path to a .las file - if omitted, auto-discovers "
                    "the one in data/ (plug and play, any file works)")
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(SCRIPTS_DIR), "data"),
                    help="fallback folder to search for a .las if --las isn't given "
                         "or doesn't exist anymore")
    ap.add_argument("--utm-x", type=float, required=True)
    ap.add_argument("--utm-y", type=float, required=True)
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M, help="meters")
    ap.add_argument("--res", type=float, help="m/cell (default: auto, scaled to the radius)")
    ap.add_argument("--out", help="default: output/reconstruct_<x>_<y>")
    ap.add_argument("--no-view", action="store_true", help="build only, don't open the viewer")
    args = ap.parse_args()

    las_path = resolve_las_path(args.las, args.data_dir)
    if not las_path:
        sys.exit(f"no .las file found (looked for {args.las!r} and in {args.data_dir})")

    res = args.res or auto_res(args.radius)
    out_dir = args.out or os.path.join(
        os.path.dirname(SCRIPTS_DIR), "output",
        f"reconstruct_{int(args.utm_x)}_{int(args.utm_y)}_r{int(args.radius)}")

    build(las_path, args.utm_x, args.utm_y, args.radius, res, out_dir)

    if not args.no_view:
        import render_3d

        class ViewArgs:
            data_dir = out_dir
            run = None
            z_exagg = 1.0
        render_3d.cmd_view(ViewArgs())


if __name__ == "__main__":
    main()
