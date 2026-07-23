#!/usr/bin/env python3
"""Greedy, accumulation-targeted storm-drain placement.

The default drains.py sprays inlets across the whole street network. This
instead finds the FEWEST inlets that clear the MOST standing water, by
placing them only where water actually accumulates.

Method: rank candidate street cells by the volume of standing water they
command (residual/max ponding depth x local contributing area), then place
inlets greedily, forbidding a new inlet within `--spacing` metres of an
existing one, until either the drain budget is hit or the next-best site
commands less than `--min-depth` of ponding. This yields inlets sitting on
the ponding hotspots - the same low points the green-corridor design marks
for bioretention.
"""
import argparse, json, os, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
from las_common import load_transform

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="output/terrain_cut_0.5")
    ap.add_argument("--ponding", required=True,
                    help=".npy of baseline residual (or max) depth, m")
    ap.add_argument("--spacing", type=float, default=25.0, help="min inlet spacing, m")
    ap.add_argument("--max-drains", type=int, default=120)
    ap.add_argument("--min-depth", type=float, default=0.10,
                    help="stop once the best remaining site ponds less than this")
    ap.add_argument("--cap", type=float, default=0.03, help="inlet capacity m3/s")
    ap.add_argument("--out", default="output/terrain_cut_corridor/drains_opt.npz")
    args = ap.parse_args()

    t = load_transform(os.path.join(args.terrain, "dem_transform.json"))
    res = t["res"]
    masks = np.load(os.path.join(args.terrain, "masks.npz"))
    valid, building = masks["valid"], masks["building"]
    courtyard = masks["courtyard"] if "courtyard" in masks else np.zeros_like(valid)
    water = masks["water"]
    street = valid & ~building & ~courtyard & ~water
    pond = np.load(args.ponding)
    pond = np.where(street, np.nan_to_num(pond), 0.0)

    # commanded volume ~ depth x local contributing area: smooth the ponding
    # so a site is credited with the pooled water around it, not one pixel
    sigma = max(1.0, 3.0 / res)
    commanded = ndimage.gaussian_filter(pond, sigma) * street

    order = np.argsort(commanded, axis=None)[::-1]
    rows_all, cols_all = np.unravel_index(order, pond.shape)
    spacing_px = args.spacing / res
    chosen_r, chosen_c = [], []
    for r, c in zip(rows_all, cols_all):
        # ranked by `commanded` (descending), so once it drops below the
        # threshold every later site is worse too -> stop. (Thresholding the
        # RAW pond here instead was the bug that stopped after 2 inlets.)
        if commanded[r, c] < args.min_depth:
            break
        if len(chosen_r) >= args.max_drains:
            break
        if chosen_r:
            dr = np.hypot(np.array(chosen_r) - r, np.array(chosen_c) - c)
            if dr.min() < spacing_px:
                continue
        chosen_r.append(int(r)); chosen_c.append(int(c))

    rows = np.array(chosen_r); cols = np.array(chosen_c)
    cap = np.full(len(rows), args.cap, np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, rows=rows, cols=cols, cap=cap)
    # ponded volume captured at these sites (depth over cell) as a diagnostic
    vol = float(pond[rows, cols].sum()) * res * res if len(rows) else 0.0
    print(f"placed {len(rows)} inlets (cap {cap.sum():.2f} m3/s), "
          f"spacing >= {args.spacing:.0f} m, on the ponding hotspots")
    print(f"wrote {args.out}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    ortho = np.asarray(Image.open(os.path.join(args.terrain, "ortho.png")))
    fig, ax = plt.subplots(figsize=(9, 12))
    ax.imshow(ortho)
    pm = np.ma.masked_where(pond < 0.05, pond)
    ax.imshow(pm, cmap="Blues", vmin=0, vmax=0.6, alpha=0.7)
    if len(rows):
        ax.scatter(cols, rows, s=26, c="red", edgecolors="k", linewidths=0.4,
                   label=f"{len(rows)} optimized inlets")
        ax.legend(loc="upper right", fontsize=11)
    ax.set_title(f"Optimized drain placement: {len(rows)} inlets on the "
                 f"ponding hotspots\n(vs 600 in the blanket layout)",
                 fontsize=12, loc="left", fontweight="bold")
    ax.axis("off")
    fig.savefig(args.out.replace(".npz", "_qa.png"), dpi=130, bbox_inches="tight")
    print(f"wrote {args.out.replace('.npz','_qa.png')}")

if __name__ == "__main__":
    main()
