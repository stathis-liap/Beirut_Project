#!/usr/bin/env python3
"""Full-extent multi-layer raster stack from the huge LAS (streaming, parallel).

One pass over the file. Per cell: lowest Z (flow-surface candidate),
highest Z (DSM-ish, for height-above-ground), point count, mean RGB.
Sparse per-batch aggregation (sort + reduceat) in workers, dense merge in
the parent — same pattern as city_flow.py, extended to multiple statistics.
Feeds build_terrain.py.

Usage:
  python scripts/build_stack.py ~/Work/Beirut_drone.las --res 1.0 \
      --out output/stack_1.0
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from las_common import LasHeader, iter_chunks, save_transform

_worker_header = None


def _init_worker(path):
    global _worker_header
    _worker_header = LasHeader(path)


def grid_shape(header, res):
    w = int(np.ceil((header.maxx - header.minx) / res))
    h = int(np.ceil((header.maxy - header.miny) / res))
    return h, w


def stats_batch(args):
    """Sparse per-cell min/max Z, count, RGB sums for one point batch."""
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
        idx = idx[order]
        z = z[order]
        cells, starts = np.unique(idx, return_index=True)
        seg_end = np.append(starts[1:], len(idx))
        rgb = np.empty((3, len(cells)), dtype=np.float64)
        for i, chn in enumerate(("red", "green", "blue")):
            rgb[i] = np.add.reduceat(pts[chn].astype(np.float64)[order], starts)
        return (cells,
                np.minimum.reduceat(z, starts),
                np.maximum.reduceat(z, starts),
                (seg_end - starts).astype(np.int64),
                rgb)
    return (np.empty(0, np.int64), np.empty(0, np.float32),
            np.empty(0, np.float32), np.empty(0, np.int64),
            np.empty((3, 0), np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("las")
    ap.add_argument("--res", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="default output/stack_<res>")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--batch-points", type=int, default=4_000_000)
    args = ap.parse_args()
    out = args.out or f"output/stack_{args.res:g}"

    header = LasHeader(args.las)
    print(header.describe())
    h, w = grid_shape(header, args.res)
    n = header.n_points_in_file
    print(f"grid {w} x {h} at {args.res} m ({h * w / 1e6:.1f} M cells), "
          f"{n:,} points")

    bounds = np.arange(0, n + args.batch_points, args.batch_points)
    bounds[-1] = min(bounds[-1], n)
    jobs = [(args.res, int(bounds[i]), int(bounds[i + 1]))
            for i in range(len(bounds) - 1) if bounds[i] < bounds[i + 1]]

    zmin = np.full(h * w, np.inf, dtype=np.float32)
    zmax = np.full(h * w, -np.inf, dtype=np.float32)
    count = np.zeros(h * w, dtype=np.int64)
    rgb_sum = np.zeros((3, h * w), dtype=np.float64)

    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.las,)) as pool:
        for k, (cells, zmn, zmx, cnt, rgb) in enumerate(
                pool.imap_unordered(stats_batch, jobs)):
            np.minimum.at(zmin, cells, zmn)
            np.maximum.at(zmax, cells, zmx)
            count[cells] += cnt
            rgb_sum[:, cells] += rgb
            print(f"\r  gridding {100 * (k + 1) / len(jobs):5.1f}%  "
                  f"({time.time() - t0:.0f}s)", end="", flush=True)
    print()

    covered = count > 0
    print(f"coverage {100 * covered.mean():.1f}%")

    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "zlow.npy"),
            np.where(covered, zmin, np.nan).reshape(h, w))
    np.save(os.path.join(out, "zhigh.npy"),
            np.where(covered, zmax, np.nan).reshape(h, w))
    np.save(os.path.join(out, "count.npy"),
            count.reshape(h, w).astype(np.int32))

    rgb = np.zeros((h * w, 3), dtype=np.float64)
    rgb[covered] = (rgb_sum[:, covered] / count[covered]).T
    if rgb.max() > 255:          # 16-bit color scaled to 8-bit
        rgb /= 256.0
    np.save(os.path.join(out, "rgb.npy"),
            np.clip(rgb, 0, 255).astype(np.uint8).reshape(h, w, 3))

    save_transform(os.path.join(out, "transform.json"),
                   header.minx, header.miny, args.res, w, h,
                   extra={"source_las": os.path.abspath(args.las)})
    print(f"wrote zlow/zhigh/count/rgb + transform in {out}/ "
          f"({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
