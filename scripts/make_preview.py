#!/usr/bin/env python3
"""Render a top-down preview of a huge LAS file (streamed, parallel).

Memory-bounded design: workers each process a small batch of points
(--batch-points) and return a SPARSE result (only the grid cells that
batch touched); the single dense grid lives only in the main process.
Counts are integers and RGB sums are float64, so the merged result is
exact -- identical to accumulating everything in one array.

Produces in --out (default output/):
  preview_rgb.png   mean-color orthoimage (gray where no data)
  preview_z.npy     max-Z per cell (NaN where no data)
  preview_count.npy points per cell (coverage mask)
  preview_transform.json  pixel<->UTM mapping

Usage:
  python scripts/make_preview.py data/Beirut_drone.las [--res 0.5] [--workers 6]
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


def grid_shape(header, res):
    width = int(np.ceil((header.maxx - header.minx) / res))
    height = int(np.ceil((header.maxy - header.miny) / res))
    return height, width


_worker_header = None  # parsed once per worker process, not per batch


def _init_worker(path):
    global _worker_header
    _worker_header = LasHeader(path)


def process_batch(args):
    """Aggregate one point batch -> sparse (cell_ids, count, rgb_sum, maxz).

    Sorting by cell id lets np.*.reduceat do exact per-cell reductions
    without allocating a full-grid array in the worker.
    """
    res, start, stop = args
    header = _worker_header
    h, w = grid_shape(header, res)

    for _, pts in iter_chunks(header, chunk_points=stop - start,
                              start=start, stop=stop):
        x, y = header.scale_xy(pts)
        z = (pts["Z"] * header.sz + header.oz).astype(np.float32)
        col = np.clip(((x - header.minx) / res).astype(np.int64), 0, w - 1)
        row = np.clip(((header.maxy - y) / res).astype(np.int64), 0, h - 1)
        idx = row * w + col
        del x, y, col, row

        order = np.argsort(idx, kind="stable")
        idx_sorted = idx[order]
        cells, starts = np.unique(idx_sorted, return_index=True)
        cnt = np.diff(np.append(starts, idx_sorted.size)).astype(np.int64)
        zmax = np.maximum.reduceat(z[order], starts)
        rgb = np.empty((cells.size, 3), dtype=np.float64)
        for ci, ch in enumerate(("red", "green", "blue")):
            rgb[:, ci] = np.add.reduceat(
                pts[ch][order].astype(np.float64), starts)
        return cells, cnt, rgb, zmax
    return (np.empty(0, np.int64), np.empty(0, np.int64),
            np.empty((0, 3), np.float64), np.empty(0, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("las", help="input LAS file")
    ap.add_argument("--res", type=float, default=0.5, help="meters per pixel")
    ap.add_argument("--out", default="output")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--batch-points", type=int, default=4_000_000,
                    help="points per worker job; lower = less RAM")
    args = ap.parse_args()

    header = LasHeader(args.las)
    print(header.describe())
    h, w = grid_shape(header, args.res)
    print(f"grid: {w} x {h} px at {args.res} m/px")

    n = header.n_points_in_file
    bounds = np.arange(0, n + args.batch_points, args.batch_points)
    bounds[-1] = min(bounds[-1], n)
    jobs = [(args.res, int(bounds[i]), int(bounds[i + 1]))
            for i in range(len(bounds) - 1) if bounds[i] < bounds[i + 1]]
    print(f"{len(jobs)} batches of <= {args.batch_points:,} points, "
          f"{args.workers} workers")

    t0 = time.time()
    count = np.zeros(h * w, dtype=np.int64)
    rgb_sum = np.zeros((h * w, 3), dtype=np.float64)
    maxz = np.full(h * w, -np.inf, dtype=np.float32)
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.las,)) as pool:
        for k, (cells, cnt, rgb, zm) in enumerate(
                pool.imap_unordered(process_batch, jobs)):
            count[cells] += cnt
            rgb_sum[cells] += rgb
            maxz[cells] = np.maximum(maxz[cells], zm)
            done = (k + 1) / len(jobs)
            eta = (time.time() - t0) / done * (1 - done)
            print(f"\r  {done * 100:5.1f}%  ({time.time() - t0:.0f}s, "
                  f"eta {eta:.0f}s)", end="", flush=True)
    print()

    os.makedirs(args.out, exist_ok=True)
    covered = count > 0

    rgb = np.full((h * w, 3), 60, dtype=np.uint8)  # dark gray = no data
    vals = rgb_sum[covered] / count[covered, None]
    if vals.size and vals.max() > 255:  # 16-bit color
        vals /= 256.0
    rgb[covered] = np.clip(vals, 0, 255).astype(np.uint8)
    Image.fromarray(rgb.reshape(h, w, 3)).save(os.path.join(args.out, "preview_rgb.png"))

    z_out = np.where(covered, maxz, np.nan).reshape(h, w).astype(np.float32)
    np.save(os.path.join(args.out, "preview_z.npy"), z_out)
    np.save(os.path.join(args.out, "preview_count.npy"),
            count.reshape(h, w).astype(np.uint32))
    save_transform(os.path.join(args.out, "preview_transform.json"),
                   header.minx, header.miny, args.res, w, h,
                   extra={"source_las": os.path.abspath(args.las),
                          "points_used": int(n),
                          "partial": header.is_partial})

    pct = 100.0 * covered.sum() / covered.size
    print(f"coverage: {pct:.1f}% of bbox cells have points")
    print(f"wrote preview_rgb.png / preview_z.npy / preview_count.npy in {args.out}/")


if __name__ == "__main__":
    main()
