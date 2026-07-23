#!/usr/bin/env python3
"""Define the cut study domain around the Al-Masar Al-Akhdar corridor.

The cut has to contain the corridor *and* everything that drains into it,
otherwise a smaller-domain run starves the corridor of upslope runoff and
under-predicts ponding. So the domain is

    (D8 upslope watershed of the corridor strip)  u  (buffer around it)

taken on the sink-filled 1 m DEM, clipped to the survey extent and reduced
to its axis-aligned bounding box (crop_cloud.py wants a simple polygon and
a rectangle keeps the exported ASCII grids rectangular for LISFLOOD-FP).

Usage:
  python scripts/cut_domain.py --terrain output/terrain_1.0 \
      --polygon output/masar_row.json --buffer 150 --out output/cut_polygon.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from las_common import load_transform, utm_to_pixel
from build_dem import fill_sinks


def receivers(dem):
    """Steepest-descent neighbour index for every cell (-1 = pit/outlet)."""
    h, w = dem.shape
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dist = np.array([np.hypot(dr, dc) for dr, dc in dirs])
    best_drop = np.zeros((h, w), dtype=np.float32)
    best_dir = np.full((h, w), -1, dtype=np.int8)
    for k, (dr, dc) in enumerate(dirs):
        shifted = np.full((h, w), np.inf, dtype=np.float32)
        rs = slice(max(0, -dr), h - max(0, dr))
        cs = slice(max(0, -dc), w - max(0, dc))
        rs2 = slice(max(0, dr), h - max(0, -dr))
        cs2 = slice(max(0, dc), w - max(0, -dc))
        shifted[rs, cs] = dem[rs2, cs2]
        drop = (dem - shifted) / dist[k]
        better = drop > best_drop
        best_drop[better] = drop[better]
        best_dir[better] = k
    return best_dir, dirs


def upslope_of(dem, seed):
    """Cells whose D8 flow path reaches `seed` (boolean mask)."""
    best_dir, dirs = receivers(dem)
    mask = seed.copy()
    h, w = dem.shape
    order = np.argsort(dem, axis=None)          # low to high: receiver first
    rows, cols = np.unravel_index(order, dem.shape)
    for r, c in zip(rows, cols):
        if mask[r, c]:
            continue
        k = best_dir[r, c]
        if k < 0:
            continue
        nr, nc = r + dirs[k][0], c + dirs[k][1]
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            mask[r, c] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="output/terrain_1.0")
    ap.add_argument("--polygon", default="output/masar_row.json")
    ap.add_argument("--buffer", type=float, default=150.0,
                    help="metres of padding around corridor + watershed")
    ap.add_argument("--out", default="output/cut_polygon.json")
    args = ap.parse_args()

    t = load_transform(os.path.join(args.terrain, "dem_transform.json"))
    dem = np.load(os.path.join(args.terrain, "dem.npy")).astype(np.float32)
    res = t["res"]

    from matplotlib.path import Path as MplPath
    poly = np.asarray(json.load(open(args.polygon))["polygon"], dtype=float)
    px, py = utm_to_pixel(t, poly[:, 0], poly[:, 1])
    h, w = dem.shape
    yy, xx = np.mgrid[0:h, 0:w]
    strip = MplPath(np.column_stack([px, py])).contains_points(
        np.column_stack([xx.ravel(), yy.ravel()])).reshape(h, w)
    print(f"corridor strip: {strip.sum()} cells")

    print("filling sinks...")
    filled = fill_sinks(np.nan_to_num(dem, nan=np.nanmax(dem)))
    print("tracing upslope watershed (D8)...")
    ws = upslope_of(filled, strip)
    print(f"watershed: {ws.sum()} cells = {ws.sum() * res * res / 1e4:.1f} ha")

    rows, cols = np.nonzero(ws | strip)
    pad = args.buffer / res
    r0 = max(0, int(rows.min() - pad)); r1 = min(h - 1, int(rows.max() + pad))
    c0 = max(0, int(cols.min() - pad)); c1 = min(w - 1, int(cols.max() + pad))
    x0 = t["minx"] + c0 * res; x1 = t["minx"] + c1 * res
    y1 = t["maxy"] - r0 * res; y0 = t["maxy"] - r1 * res
    print(f"cut bbox: X {x0:.0f}-{x1:.0f} ({x1-x0:.0f} m), "
          f"Y {y0:.0f}-{y1:.0f} ({y1-y0:.0f} m)")
    for r in (1.0, 0.5, 0.25):
        print(f"  at {r} m: {int((x1-x0)/r)} x {int((y1-y0)/r)} = "
              f"{(x1-x0)*(y1-y0)/r/r/1e6:.2f} M cells")

    out = {"crs": t.get("crs", "EPSG:32636"),
           "utm_polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
           "note": f"cut domain: corridor + D8 upslope watershed + "
                   f"{args.buffer:.0f} m buffer, from {args.polygon}"}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    ortho = np.asarray(Image.open(os.path.join(args.terrain, "ortho.png")))
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.imshow(ortho)
    ax.imshow(np.ma.masked_where(~ws, ws), cmap="autumn", alpha=0.28)
    ax.plot(np.append(px, px[0]), np.append(py, py[0]), color="lime", lw=2)
    ax.plot([c0, c1, c1, c0, c0], [r0, r0, r1, r1, r0], color="cyan", lw=2.2)
    ax.set_title("Cut domain (cyan) = corridor (green) + upslope watershed "
                 f"(orange) + {args.buffer:.0f} m")
    ax.axis("off")
    qa = os.path.join(args.terrain, "cut_domain_qa.png")
    fig.savefig(qa, dpi=130, bbox_inches="tight")
    print(f"wrote {qa}")


if __name__ == "__main__":
    main()
