#!/usr/bin/env python3
"""Crop a huge LAS file to a polygon marked by clicking on the preview image.

Interactive:
  python scripts/crop_cloud.py data/Beirut_drone.las --out data/corridor.las
    left-click  add vertex
    right-click remove last vertex
    Enter       finish polygon and start the crop

Non-interactive (reuse a saved polygon):
  python scripts/crop_cloud.py data/Beirut_drone.las --out data/corridor.las \
      --polygon output/crop_polygon.json

The polygon JSON holds UTM (EPSG:32636) coordinates, so it stays valid for
any preview resolution and for the full file once the download completes.
Requires make_preview.py to have been run first (for the map + transform).
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from matplotlib.path import Path as MplPath

sys.path.insert(0, os.path.dirname(__file__))
from las_common import (LasHeader, iter_chunks, load_transform,
                        utm_to_pixel, pixel_to_utm)


def pick_polygon(preview_dir):
    """Show the preview and let the user click a polygon. Returns UTM coords."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from PIL import Image

    t = load_transform(os.path.join(preview_dir, "preview_transform.json"))
    img = np.asarray(Image.open(os.path.join(preview_dir, "preview_rgb.png")))

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img)
    ax.set_title("Click polygon vertices (right-click = undo, Enter = done)")
    xs, ys = [], []
    line, = ax.plot([], [], "-o", color="yellow", lw=1.5, ms=4)

    def redraw():
        if xs:
            line.set_data(xs + [xs[0]], ys + [ys[0]])
        else:
            line.set_data([], [])
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes != ax:
            return
        if ev.button == 1:
            xs.append(ev.xdata)
            ys.append(ev.ydata)
        elif ev.button == 3 and xs:
            xs.pop()
            ys.pop()
        redraw()

    def on_key(ev):
        if ev.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if len(xs) < 3:
        sys.exit("need at least 3 vertices, got %d" % len(xs))
    ux, uy = pixel_to_utm(t, np.array(xs), np.array(ys))
    return np.column_stack([ux, uy])


def filter_range(args):
    """Return (n_kept, bytes_of_kept_records, zmin, zmax) for one point range."""
    path, poly, start, stop = args
    header = LasHeader(path)
    mpath = MplPath(poly)
    bx0, by0 = poly.min(axis=0)
    bx1, by1 = poly.max(axis=0)
    kept = []
    zmin, zmax = np.inf, -np.inf
    for _, pts in iter_chunks(header, start=start, stop=stop):
        x, y = header.scale_xy(pts)
        m = (x >= bx0) & (x <= bx1) & (y >= by0) & (y <= by1)
        if not m.any():
            continue
        xi, yi = x[m], y[m]
        inside = mpath.contains_points(np.column_stack([xi, yi]))
        if not inside.any():
            continue
        sel = pts[m][inside]
        kept.append(sel.tobytes())
        z = sel["Z"] * header.sz + header.oz
        zmin = min(zmin, z.min())
        zmax = max(zmax, z.max())
    return len(kept) and sum(len(b) // header.record_length for b in kept), b"".join(kept), zmin, zmax


def check_coverage(preview_dir, poly):
    """Warn if part of the polygon has no points (partial download)."""
    count = np.load(os.path.join(preview_dir, "preview_count.npy"))
    t = load_transform(os.path.join(preview_dir, "preview_transform.json"))
    h, w = count.shape
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    ux, uy = pixel_to_utm(t, cols.ravel() + 0.5, rows.ravel() + 0.5)
    inside = MplPath(poly).contains_points(np.column_stack([ux, uy]))
    n_in = inside.sum()
    if n_in == 0:
        print("WARNING: polygon is outside the preview grid")
        return
    covered = (count.ravel()[inside] > 0).sum()
    pct = 100.0 * covered / n_in
    print(f"coverage inside polygon: {pct:.1f}% of cells have points")
    if pct < 95:
        print("WARNING: polygon has data gaps - the download may still be "
              "incomplete there. Re-run the crop when the file is complete.")


def write_las(out_path, header, point_bytes, n_kept, poly, zmin, zmax):
    """Write kept records under a copy of the source header with fixed counts/extents."""
    import struct
    raw = bytearray(header.read_raw_prefix())
    struct.pack_into("<I", raw, 107, n_kept)          # point count
    struct.pack_into("<5I", raw, 111, n_kept, 0, 0, 0, 0)  # returns histogram
    bx0, by0 = poly.min(axis=0)
    bx1, by1 = poly.max(axis=0)
    struct.pack_into("<6d", raw, 179, bx1, bx0, by1, by0, zmax, zmin)
    with open(out_path, "wb") as f:
        f.write(raw)
        for b in point_bytes:
            f.write(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("las")
    ap.add_argument("--out", required=True, help="output cropped .las")
    ap.add_argument("--polygon", help="reuse saved polygon JSON instead of clicking")
    ap.add_argument("--preview-dir", default="output")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = ap.parse_args()

    header = LasHeader(args.las)
    print(header.describe())

    if args.polygon:
        with open(args.polygon) as f:
            poly = np.array(json.load(f)["utm_polygon"])
    else:
        poly = pick_polygon(args.preview_dir)
        pj = os.path.join(args.preview_dir, "crop_polygon.json")
        with open(pj, "w") as f:
            json.dump({"crs": "EPSG:32636", "utm_polygon": poly.tolist()}, f, indent=2)
        print(f"polygon saved to {pj}")
    print("polygon vertices (UTM):")
    for x, y in poly:
        print(f"  {x:.1f}, {y:.1f}")

    if os.path.exists(os.path.join(args.preview_dir, "preview_count.npy")):
        check_coverage(args.preview_dir, poly)

    n = header.n_points_in_file
    n_ranges = args.workers * 4
    bounds = np.linspace(0, n, n_ranges + 1, dtype=np.int64)
    jobs = [(args.las, poly, int(bounds[i]), int(bounds[i + 1]))
            for i in range(n_ranges) if bounds[i] < bounds[i + 1]]

    t0 = time.time()
    results = []
    with mp.Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap(filter_range, jobs)):
            results.append(res)
            print(f"\r  {100 * (k + 1) / len(jobs):5.1f}%  ({time.time() - t0:.0f}s)",
                  end="", flush=True)
    print()

    n_kept = sum(r[0] for r in results)
    if n_kept == 0:
        sys.exit("no points inside polygon - nothing written")
    zmin = min(r[2] for r in results)
    zmax = max(r[3] for r in results)
    write_las(args.out, header, [r[1] for r in results], n_kept, poly, zmin, zmax)

    print(f"kept {n_kept:,} of {n:,} points "
          f"({100 * n_kept / n:.2f}%) -> {args.out} "
          f"({os.path.getsize(args.out) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
