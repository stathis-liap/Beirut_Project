#!/usr/bin/env python3
"""Cross-check our GPU solver against an external engine (PLAN.md D3).

Both engines must consume IDENTICAL inputs, so this tool:
  run-ours   runs flood_gpu in "plain mode" on the exported dem.asc/n.asc
             (uniform rain everywhere, no roof rerouting, no drains, no
             water mask, free edges) - i.e. exactly what LISFLOOD-FP sees
  compare    compares two max-depth rasters (ours .npy vs the engine's
             ESRI-ASCII .max) - agreement metrics + side-by-side figure

Typical LISFLOOD-FP session:
  python scripts/export_ascii.py --terrain output/terrain_test \
      --storm storms/v1_nov2025.json --out output/export_test
  <lisflood build>/lisflood -v output/export_test/v1_nov2025.par
  python scripts/crosscheck.py run-ours --export output/export_test \
      --storm storms/v1_nov2025.json --out output/crosscheck/ours_test
  python scripts/crosscheck.py compare \
      --ours output/crosscheck/ours_test/max_depth.npy \
      --theirs output/export_test/results/res.max \
      --export output/export_test --out output/crosscheck/test_vs_lfp.png
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from flood_gpu import simulate

NODATA = -9999.0


def read_asc(path):
    hdr = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            hdr[k.lower()] = float(v)
        arr = np.loadtxt(f, dtype=np.float64)
    nod = hdr.get("nodata_value", NODATA)
    arr[arr == nod] = np.nan
    return arr, hdr


def cmd_run_ours(args):
    dem, hdr = read_asc(os.path.join(args.export, "dem.asc"))
    n, _ = read_asc(os.path.join(args.export, "n.asc"))
    res = hdr["cellsize"]
    with open(args.storm) as f:
        storm = json.load(f)
    valid = np.isfinite(dem)
    dem32 = dem.astype(np.float32)
    n32 = np.nan_to_num(n, nan=0.03).astype(np.float32)
    rain_w = valid.astype(np.float32)                # uniform, incl. roofs
    pad = 0
    if args.closed:
        # LISFLOOD's borders are closed walls; wall the domain so our
        # zeroed outer ring is unreachable, then crop the results back
        pad = 3
        wall = float(np.nanmax(dem)) + 100.0
        # Interior NODATA has to be walled too, not left as a sink. simulate()
        # turns invalid cells into low ground that swallows whatever reaches
        # them, while LISFLOOD substitutes nodata_elevation (1e7) and blocks
        # them - so leaving them as sinks would compare an engine that drains
        # to the sea against one that does not. Matching behaviour matters
        # more than realism here; the physical outflow mask belongs to the
        # full pipeline runs, not to this plain-mode numerical comparison.
        dem32 = np.where(valid, dem32, wall)
        rain_w = np.where(valid, rain_w, 0.0)
        valid = np.ones_like(valid)
        dem32 = np.pad(dem32, pad, constant_values=wall)
        n32 = np.pad(n32, pad, constant_values=0.03)
        valid = np.pad(valid, pad, constant_values=True)
        rain_w = np.pad(rain_w, pad, constant_values=0.0)
    print(f"plain-mode run ({'closed' if args.closed else 'open'} borders): "
          f"{dem.shape[1]} x {dem.shape[0]} at {res} m, storm {storm['name']}")
    simulate(dem32, res, storm["steps"], storm["duration"], args.out,
             manning=n32, valid=valid, rain_weight=rain_w,
             save_every=args.save_every, device=args.device,
             save_frames=False)
    if pad:
        for f_ in ("max_depth", "max_vel", "max_hazard", "final_depth"):
            p = os.path.join(args.out, f_ + ".npy")
            np.save(p, np.load(p)[pad:-pad, pad:-pad])


def cmd_compare(args):
    ours = np.load(args.ours).astype(np.float64)
    theirs, hdr = read_asc(args.theirs)
    theirs = np.nan_to_num(theirs, nan=0.0)
    if ours.shape != theirs.shape:
        sys.exit(f"shape mismatch: ours {ours.shape} vs theirs {theirs.shape}")
    dem, _ = read_asc(os.path.join(args.export, "dem.asc"))
    valid = np.isfinite(dem)
    # interior only: the engines treat the outer boundary differently
    from scipy import ndimage
    interior = ndimage.binary_erosion(valid, iterations=args.edge_buffer)

    a, b = ours[interior], theirs[interior]
    wet_a, wet_b = a > 0.10, b > 0.10
    both, either = (wet_a & wet_b).sum(), (wet_a | wet_b).sum()
    hit = both / max(either, 1)                      # IoU of flooded extent
    wet = wet_a | wet_b
    corr = float(np.corrcoef(a[wet], b[wet])[0, 1]) if wet.sum() > 10 else 0.0
    rmse = float(np.sqrt(np.mean((a[wet] - b[wet]) ** 2))) if wet.any() else 0.0
    bias = float(np.mean(a[wet] - b[wet])) if wet.any() else 0.0
    stats = {
        "cells_interior": int(interior.sum()),
        "wet_area_ours_m2": float(wet_a.sum()),
        "wet_area_theirs_m2": float(wet_b.sum()),
        "extent_iou_gt10cm": round(hit, 3),
        "depth_corr_wet": round(corr, 3),
        "rmse_wet_m": round(rmse, 3),
        "bias_ours_minus_theirs_m": round(bias, 3),
        "p99_ours": round(float(np.percentile(a[wet_a], 99)) if wet_a.any() else 0, 3),
        "p99_theirs": round(float(np.percentile(b[wet_b], 99)) if wet_b.any() else 0, 3),
    }
    print(json.dumps(stats, indent=2))
    # AGREE: near-identical fields. CONSISTENT: within normal inter-model
    # tolerance for 2D urban codes (EA/Neelz-Pender benchmark reports show
    # +/-10 cm scatter between industry models on urban cases) - expected
    # here since the engines use different stabilizations (our positivity
    # scaling vs LISFLOOD's theta diffusion).
    if hit > 0.6 and corr > 0.85 and abs(bias) < 0.05 and rmse < 0.05:
        verdict = "AGREE"
    elif hit > 0.7 and rmse < 0.10 and abs(bias) < 0.05:
        verdict = "CONSISTENT (within inter-model tolerance)"
    else:
        verdict = "DIVERGE - investigate"
    print(f"\ncross-check verdict: {verdict}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    vmax = max(np.percentile(a[wet_a], 99) if wet_a.any() else 0.3, 0.3)
    for ax, fld, ttl in [(axes[0], ours, "our solver (torch/CUDA)"),
                         (axes[1], theirs, "external engine")]:
        im = ax.imshow(np.where(fld > 0.02, fld, np.nan), cmap="turbo",
                       vmin=0, vmax=vmax)
        ax.set_title(f"{ttl} - max depth")
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.7)
    axes[2].plot([0, vmax], [0, vmax], "k--", lw=1)
    sel = wet & (np.random.default_rng(0).random(wet.shape[0] if wet.ndim == 1
                                                 else wet.size).reshape(wet.shape)
                 < min(1.0, 20000 / max(wet.sum(), 1)))
    axes[2].plot(b[sel], a[sel], ".", ms=2, alpha=0.4, color="#2a78d6")
    axes[2].set_xlabel("external engine depth (m)")
    axes[2].set_ylabel("our solver depth (m)")
    axes[2].set_title(f"wet-cell scatter  (corr {corr:.3f}, "
                      f"RMSE {rmse * 100:.1f} cm, IoU {hit:.2f})")
    fig.suptitle(f"Cross-check: {verdict}", fontsize=15,
                 color="red" if verdict.startswith("DIVERGE") else "green")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump({**stats, "verdict": verdict}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run-ours")
    r.add_argument("--export", required=True, help="dir with dem.asc/n.asc")
    r.add_argument("--storm", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--save-every", type=float, default=300.0)
    r.add_argument("--device", default="auto")
    r.add_argument("--closed", action="store_true",
                   help="wall the borders (match LISFLOOD closed edges)")
    r.set_defaults(func=cmd_run_ours)
    c = sub.add_parser("compare")
    c.add_argument("--ours", required=True, help="max_depth.npy")
    c.add_argument("--theirs", required=True, help="engine .max ESRI ascii")
    c.add_argument("--export", required=True)
    c.add_argument("--edge-buffer", type=int, default=10, help="cells")
    c.add_argument("--out", required=True, help="comparison figure png")
    c.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
