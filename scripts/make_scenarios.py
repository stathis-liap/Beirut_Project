#!/usr/bin/env python3
"""Auto-generate what-if scenario JSONs from the corridor DEM.

Traces the main flow channel (max D8 accumulation) and builds three
scenarios along it:
  greened_upslope   infiltration strip (permeable soil) on the UPPER half
  escape_channel    0.4 m carved drainage channel along the LOWER half
  combined          both edits

Writes scenarios/*.json plus scenarios_map.png (edit polygons over the
ortho) for a visual check.

Usage:
  python scripts/make_scenarios.py [--data-dir output] [--scen-dir scenarios]
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from las_common import load_transform, utm_to_pixel
from build_dem import fill_sinks, d8_flow_accumulation

DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def d8_best_dir(dem):
    """Steepest-descent direction index per cell (-1 = pit)."""
    h, w = dem.shape
    dist = np.array([np.hypot(dr, dc) for dr, dc in DIRS])
    best_drop = np.zeros((h, w), dtype=np.float32)
    best_dir = np.full((h, w), -1, dtype=np.int8)
    for k, (dr, dc) in enumerate(DIRS):
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
    return best_dir


def mask_to_utm_polygon(mask, t, step_m=20.0):
    """Outline of a boolean mask as a simple UTM polygon (subsampled)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cs = ax.contour(mask.astype(float), levels=[0.5])
    segs = max((s for lvl in cs.allsegs for s in lvl), key=len)
    plt.close(fig)
    keep = max(1, int(step_m / t["res"]))
    seg = segs[::keep]
    ux = t["minx"] + (seg[:, 0] + 0.5) * t["res"]
    uy = t["maxy"] - (seg[:, 1] + 0.5) * t["res"]
    return np.column_stack([ux, uy])


def coarsen_min(dem, factor):
    """Block-minimum downsample: streets stay at street level, photogram
    noise (cars, trees, balconies) drops out -> clean D8 channels."""
    h, w = dem.shape
    hc, wc = h // factor, w // factor
    d = dem[:hc * factor, :wc * factor].reshape(hc, factor, wc, factor)
    return np.nanmin(np.nanmin(d, axis=3), axis=1)


def trace_main_channel(dem, res, factor=4):
    """Main drainage stem on a coarsened DEM (D8 at 0.5 m is too noisy).

    Downhill tracing from an uphill seed is fragile (sink filling creates
    edge lakes with huge accumulation), so instead: take the max-accumulation
    INTERIOR cell as the outlet and climb upstream along the largest
    tributary. Returns path cells in FINE grid coordinates, uphill first."""
    coarse = coarsen_min(dem, factor)
    finite = np.isfinite(coarse)
    coarse = np.where(finite, coarse, np.nanmax(coarse) + 50.0)
    filled = fill_sinks(coarse)
    acc = d8_flow_accumulation(filled)
    bd = d8_best_dir(filled)

    interior = ndimage.distance_transform_edt(finite) > 4  # > ~8 m inside
    interior[:3, :] = interior[-3:, :] = False
    interior[:, :3] = interior[:, -3:] = False
    r, c = np.unravel_index(np.argmax(np.where(interior, acc, 0)), acc.shape)
    stop_acc = max(100.0, 0.005 * acc[r, c])

    h, w = coarse.shape
    path = [(r, c)]
    while True:
        best, best_acc = None, stop_acc
        for k, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not finite[nr, nc]:
                continue
            # neighbor drains into (r, c) if its best_dir is the reverse k
            if bd[nr, nc] == 7 - k and acc[nr, nc] > best_acc:
                best, best_acc = (nr, nc), acc[nr, nc]
        if best is None:
            break
        r, c = best
        path.append((r, c))
    path.reverse()
    path = np.array(path) * factor + factor // 2  # fine-grid coordinates
    fine_ok = np.isfinite(dem[path[:, 0], path[:, 1]])
    return path[fine_ok]


def buffer_polygon(path_cells, shape, t, width_m):
    mask = np.zeros(shape, dtype=bool)
    mask[path_cells[:, 0], path_cells[:, 1]] = True
    dist = ndimage.distance_transform_edt(~mask) * t["res"]
    return mask_to_utm_polygon(dist <= width_m / 2.0, t, step_m=5.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="output")
    ap.add_argument("--scen-dir", default="scenarios")
    ap.add_argument("--infil-width", type=float, default=16.0)
    ap.add_argument("--infil-mmh", type=float, default=60.0)
    ap.add_argument("--channel-width", type=float, default=4.0)
    ap.add_argument("--channel-depth", type=float, default=0.4)
    args = ap.parse_args()

    t = load_transform(os.path.join(args.data_dir, "dem_transform.json"))
    dem = np.load(os.path.join(args.data_dir, "dem.npy"))
    print(f"dem {dem.shape[1]} x {dem.shape[0]} at {t['res']} m")

    path = trace_main_channel(dem, t["res"])
    n = len(path)
    print(f"main channel: {n} cells, elev "
          f"{dem[tuple(path[0])]:.1f} -> {dem[tuple(path[-1])]:.1f} m")

    upper = path[: n // 2]
    lower = path[n // 2:]
    poly_infil = buffer_polygon(upper, dem.shape, t, args.infil_width)
    poly_chan = buffer_polygon(lower, dem.shape, t, args.channel_width)

    os.makedirs(args.scen_dir, exist_ok=True)
    edits_infil = [{"op": "infiltrate", "mmh": args.infil_mmh,
                    "polygon": poly_infil.tolist()}]
    edits_chan = [{"op": "lower", "meters": args.channel_depth,
                   "polygon": poly_chan.tolist()}]
    scenarios = [
        ("greened_upslope", edits_infil),
        ("escape_channel", edits_chan),
        ("combined", edits_infil + edits_chan),
    ]
    for name, edits in scenarios:
        p = os.path.join(args.scen_dir, f"{name}.json")
        with open(p, "w") as f:
            json.dump({"name": name, "edits": edits}, f, indent=2)
        print(f"wrote {p}")

    # visual check over the ortho
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ortho = np.asarray(Image.open(os.path.join(args.data_dir, "ortho.png")))
    fig, ax = plt.subplots(figsize=(10, 16))
    ax.imshow(ortho)
    ax.plot(path[:, 1], path[:, 0], "--", color="deepskyblue", lw=1.2,
            label="main flow channel")
    for poly, col, lab in [
            (poly_infil, "lime", f"infiltration strip ({args.infil_mmh:.0f} mm/h)"),
            (poly_chan, "orange", f"escape channel (-{args.channel_depth} m)")]:
        px, py = utm_to_pixel(t, poly[:, 0], poly[:, 1])
        ax.plot(np.append(px, px[0]), np.append(py, py[0]), color=col,
                lw=2.5, label=lab)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_title("Scenario edits on the corridor")
    ax.axis("off")
    out = os.path.join(args.data_dir, "scenarios_map.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
