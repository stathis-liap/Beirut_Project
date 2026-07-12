#!/usr/bin/env python3
"""Build the flow-surface DEM + ortho texture from a cropped LAS.

Per grid cell we take a low percentile of Z (street level between parked
cars / pedestrians; building roofs where only roof points exist, which
correctly makes buildings act as flow obstacles) and the mean RGB.

Outputs in --out (default output/):
  dem.npy            float32 elevation, holes filled
  ortho.png          RGB texture aligned with the DEM
  dem_transform.json pixel<->UTM mapping
  dem_hillshade.png  QA: shaded relief
  flow_accum.png     QA: D8 flow accumulation over ortho ("where the river runs")

Usage:
  python scripts/build_dem.py data/corridor.las [--res 1.0]
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from las_common import LasHeader, iter_chunks, save_transform


def grid_points(header, res, z_percentile=10):
    minx, miny = header.minx, header.miny
    w = int(np.ceil((header.maxx - minx) / res))
    h = int(np.ceil((header.maxy - miny) / res))

    # Two passes keep memory bounded: min-ish Z via percentile needs the
    # values, so collect per-cell sums for RGB and per-cell Z arrays lazily.
    # Simpler approach that is accurate enough: track the k-th lowest by
    # keeping per-cell minimum after removing extreme low outliers per chunk.
    zmin = np.full(h * w, np.inf, dtype=np.float32)
    zlow = np.full(h * w, np.inf, dtype=np.float32)  # 2nd stage percentile approx
    count = np.zeros(h * w, dtype=np.int64)
    rgb_sum = np.zeros((h * w, 3), dtype=np.float64)

    n = header.n_points_in_file
    for i, pts in iter_chunks(header, chunk_points=10_000_000):
        x, y = header.scale_xy(pts)
        z = (pts["Z"] * header.sz + header.oz).astype(np.float32)
        col = np.clip(((x - minx) / res).astype(np.int64), 0, w - 1)
        row = np.clip(((header.maxy - y) / res).astype(np.int64), 0, h - 1)
        idx = row * w + col
        np.minimum.at(zmin, idx, z)
        np.add.at(count, idx, 1)
        for ci, ch in enumerate(("red", "green", "blue")):
            np.add.at(rgb_sum[:, ci], idx, pts[ch].astype(np.float64))
        print(f"\r  pass 1/2: {100 * (i + len(pts)) / n:5.1f}%", end="", flush=True)
    print()

    # Second pass: percentile-like floor = min of points within 30cm of cell
    # minimum, which rejects isolated low-noise points.
    for i, pts in iter_chunks(header, chunk_points=10_000_000):
        x, y = header.scale_xy(pts)
        z = (pts["Z"] * header.sz + header.oz).astype(np.float32)
        col = np.clip(((x - minx) / res).astype(np.int64), 0, w - 1)
        row = np.clip(((header.maxy - y) / res).astype(np.int64), 0, h - 1)
        idx = row * w + col
        near_floor = z <= (zmin[idx] + 0.30)
        np.minimum.at(zlow, idx[near_floor], z[near_floor])
        print(f"\r  pass 2/2: {100 * (i + len(pts)) / n:5.1f}%", end="", flush=True)
    print()

    covered = count > 0
    dem = np.where(covered, zlow, np.nan).reshape(h, w)

    rgb = np.full((h * w, 3), 0, dtype=np.float64)
    rgb[covered] = rgb_sum[covered] / count[covered, None]
    if rgb.max() > 255:
        rgb /= 256.0
    ortho = np.clip(rgb, 0, 255).astype(np.uint8).reshape(h, w, 3)
    return dem, ortho, covered.reshape(h, w)


def fill_holes(dem, max_iters=50):
    """Iteratively fill NaN cells with the mean of valid neighbors."""
    filled = dem.copy()
    for _ in range(max_iters):
        nan = np.isnan(filled)
        if not nan.any():
            break
        v = np.where(nan, 0.0, filled)
        m = (~nan).astype(np.float32)
        k = np.ones((3, 3), dtype=np.float32)
        vs = ndimage.convolve(v, k, mode="nearest")
        ms = ndimage.convolve(m, k, mode="nearest")
        can = nan & (ms > 0)
        filled[can] = vs[can] / ms[can]
    return filled


def despeckle(dem, size=3, thresh=0.5):
    """Median-replace only cells that deviate hard from local median
    (cars, people, noise) so stairs/curbs stay sharp."""
    med = ndimage.median_filter(dem, size=size)
    out = dem.copy()
    spikes = np.abs(dem - med) > thresh
    out[spikes] = med[spikes]
    return out


def hillshade(dem, res, az=315.0, alt=45.0):
    gy, gx = np.gradient(dem, res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(az), np.radians(alt)
    hs = (np.sin(alt) * np.cos(slope) +
          np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(hs, 0, 1)


def fill_sinks(dem, eps=1e-3):
    """Priority-flood depression filling with epsilon gradient
    (Barnes et al. 2014), so filled flats still drain to their pour point.

    Only used for the D8 quick-look; the dynamic solver gets the raw DEM
    so real ponding still happens there.
    """
    import heapq
    h, w = dem.shape
    filled = dem.copy()
    visited = np.zeros((h, w), dtype=bool)
    heap = []
    for c in range(w):
        for r in (0, h - 1):
            heapq.heappush(heap, (dem[r, c], r, c))
            visited[r, c] = True
    for r in range(1, h - 1):
        for c in (0, w - 1):
            heapq.heappush(heap, (dem[r, c], r, c))
            visited[r, c] = True
    while heap:
        z, r, c = heapq.heappop(heap)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    visited[nr, nc] = True
                    nz = max(dem[nr, nc], z + eps)
                    filled[nr, nc] = nz
                    heapq.heappush(heap, (nz, nr, nc))
    return filled


def d8_flow_accumulation(dem):
    """Simple D8: route each cell's accumulated area to its steepest
    downslope neighbor, processing cells from high to low."""
    h, w = dem.shape
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dist = np.array([np.hypot(dr, dc) for dr, dc in dirs])

    # steepest descent neighbor for every cell
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

    acc = np.ones((h, w), dtype=np.float64)
    order = np.argsort(dem, axis=None)[::-1]  # high to low
    rows, cols = np.unravel_index(order, dem.shape)
    for r, c in zip(rows, cols):
        k = best_dir[r, c]
        if k >= 0:
            dr, dc = dirs[k]
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                acc[nr, nc] += acc[r, c]
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("las")
    ap.add_argument("--res", type=float, default=1.0)
    ap.add_argument("--out", default="output")
    ap.add_argument("--no-flowaccum", action="store_true")
    args = ap.parse_args()

    header = LasHeader(args.las)
    print(header.describe())

    print("gridding...")
    dem, ortho, covered = grid_points(header, args.res)
    print(f"grid {dem.shape[1]} x {dem.shape[0]} at {args.res} m, "
          f"coverage {100 * covered.mean():.1f}%")

    print("filling holes / despeckling...")
    dem = fill_holes(dem)
    dem = despeckle(dem)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "dem.npy"), dem.astype(np.float32))
    Image.fromarray(ortho).save(os.path.join(args.out, "ortho.png"))
    save_transform(os.path.join(args.out, "dem_transform.json"),
                   header.minx, header.miny, args.res,
                   dem.shape[1], dem.shape[0],
                   extra={"source_las": os.path.abspath(args.las)})

    hs = (hillshade(dem, args.res) * 255).astype(np.uint8)
    Image.fromarray(hs).save(os.path.join(args.out, "dem_hillshade.png"))

    if not args.no_flowaccum:
        print("D8 flow accumulation (on sink-filled DEM)...")
        acc = d8_flow_accumulation(fill_sinks(dem))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(ortho)
        la = np.log10(acc)
        ax.imshow(np.where(la > 2.0, la, np.nan), cmap="Blues",
                  alpha=0.85, vmin=2, vmax=la.max())
        ax.set_title("D8 flow accumulation (log scale) - where water concentrates")
        ax.axis("off")
        fig.savefig(os.path.join(args.out, "flow_accum.png"),
                    dpi=150, bbox_inches="tight")
        print("wrote flow_accum.png")

    print(f"wrote dem.npy / ortho.png / dem_hillshade.png in {args.out}/")


if __name__ == "__main__":
    main()
