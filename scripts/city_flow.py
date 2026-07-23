#!/usr/bin/env python3
"""City-wide quick-look: where does water concentrate?

One parallel min-Z pass over the full LAS at coarse resolution, then
sink-filled D8 flow accumulation. Output: flow channels drawn over the
preview image — the "rivers" of Beirut. Meant to locate flood corridors
before cropping, not for the dynamic solver.

Usage:
  python scripts/city_flow.py ~/Work/Beirut_drone.las [--res 2.0]
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from las_common import LasHeader, iter_chunks, save_transform
from build_dem import fill_holes, despeckle, fill_sinks, d8_flow_accumulation

_worker_header = None


def _init_worker(path):
    global _worker_header
    _worker_header = LasHeader(path)


def grid_shape(header, res):
    w = int(np.ceil((header.maxx - header.minx) / res))
    h = int(np.ceil((header.maxy - header.miny) / res))
    return h, w


def min_batch(args):
    """Sparse per-cell minimum Z for one point batch."""
    res, start, stop = args
    header = _worker_header
    h, w = grid_shape(header, res)
    for _, pts in iter_chunks(header, chunk_points=stop - start,
                              start=start, stop=stop):
        z = (pts["Z"] * header.sz + header.oz).astype(np.float32)
        x, y = header.scale_xy(pts)
        col = np.clip(((x - header.minx) / res).astype(np.int64), 0, w - 1)
        row = np.clip(((header.maxy - y) / res).astype(np.int64), 0, h - 1)
        idx = row * w + col
        order = np.argsort(idx, kind="stable")
        idx_sorted = idx[order]
        cells, starts = np.unique(idx_sorted, return_index=True)
        zmin = np.minimum.reduceat(z[order], starts)
        return cells, zmin
    return np.empty(0, np.int64), np.empty(0, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("las")
    ap.add_argument("--res", type=float, default=2.0)
    ap.add_argument("--out", default="output")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--batch-points", type=int, default=4_000_000)
    args = ap.parse_args()

    header = LasHeader(args.las)
    h, w = grid_shape(header, args.res)
    n = header.n_points_in_file
    print(f"grid {w} x {h} at {args.res} m, {n:,} points")

    bounds = np.arange(0, n + args.batch_points, args.batch_points)
    bounds[-1] = min(bounds[-1], n)
    jobs = [(args.res, int(bounds[i]), int(bounds[i + 1]))
            for i in range(len(bounds) - 1) if bounds[i] < bounds[i + 1]]

    t0 = time.time()
    zmin = np.full(h * w, np.inf, dtype=np.float32)
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.las,)) as pool:
        for k, (cells, zm) in enumerate(pool.imap_unordered(min_batch, jobs)):
            zmin[cells] = np.minimum(zmin[cells], zm)
            print(f"\r  gridding {100 * (k + 1) / len(jobs):5.1f}%  "
                  f"({time.time() - t0:.0f}s)", end="", flush=True)
    print()

    dem = np.where(np.isfinite(zmin), zmin, np.nan).reshape(h, w)
    covered = np.isfinite(dem)
    print(f"coverage {100 * covered.mean():.1f}%; filling/despeckling...")
    dem = fill_holes(dem, max_iters=30)
    dem = despeckle(dem)
    # outside the survey nothing was filled -> set high so flow stays inside
    dem = np.where(np.isnan(dem), np.nanmax(dem) + 100.0, dem)

    print("sink filling...")
    t1 = time.time()
    filled = fill_sinks(dem)
    print(f"  {time.time() - t1:.0f}s")
    print("D8 accumulation...")
    t1 = time.time()
    acc = d8_flow_accumulation(filled)
    print(f"  {time.time() - t1:.0f}s")

    np.save(os.path.join(args.out, "city_dem.npy"), dem.astype(np.float32))
    np.save(os.path.join(args.out, "city_flow_accum.npy"), acc.astype(np.float32))
    save_transform(os.path.join(args.out, "city_flow_transform.json"),
                   header.minx, header.miny, args.res, w, h)

    # overlay channels on the (downsampled) preview
    prev = np.asarray(Image.open(
        os.path.join(args.out, "preview_rgb.png")).resize((w, h)))
    la = np.log10(acc)
    la[~covered] = 0.0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.imshow(prev)
    ax.imshow(np.where(la > 2.5, la, np.nan), cmap="cool",
              alpha=0.9, vmin=2.5, vmax=np.nanmax(la))
    ax.set_title(f"City-wide D8 flow accumulation at {args.res} m "
                 "(bright = more upstream area)")
    ax.axis("off")
    fig.savefig(os.path.join(args.out, "city_flow.png"),
                dpi=150, bbox_inches="tight")
    print(f"wrote city_flow.png / city_dem.npy / city_flow_accum.npy "
          f"in {args.out}/")


if __name__ == "__main__":
    main()
