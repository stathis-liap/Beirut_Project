#!/usr/bin/env python3
"""Locate real stone staircases in the Beirut corridor point cloud.

The corridor-wide DEM (terrain/dem.npy) is gridded at 1 m, too coarse to
resolve individual stair treads/risers (~30 cm / ~15 cm). But the source
point cloud is dense (~475 pts/m^2, ~4.6 cm spacing), dense enough to
resolve real steps *if* we regrid a small area at high resolution.

Two-stage search:
  1. Coarse pass on the existing 1 m DEM: flag narrow, steep, elongated,
     street-level (non-rooftop) passages as candidate ROIs. Cheap, but
     can't tell a real staircase from a steep ramp or a cleared lot.
  2. Fine pass: stream corridor.las ONCE, bucket points into whichever
     candidate ROI they fall in (via a coarse id-lookup grid, so it's a
     single O(N) pass regardless of candidate count), then regrid each
     ROI at ~8 cm and look for a genuinely periodic sawtooth elevation
     profile along the ROI's long axis - the real signature of stair
     treads/risers that a smooth ramp does not have.

Usage:
  python scripts/find_real_stairs.py --top 8
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from las_common import LasHeader, iter_chunks, load_transform, pixel_to_utm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- stage 1: coarse candidate search on the 1 m DEM ---------------------
ROOF_WINDOW_M = 25       # local-min filter footprint: bigger than a building
ROOF_MARGIN_M = 3.0      # a cell is "street-level" if within this of local min
SLOPE_LO, SLOPE_HI = 0.25, 1.3
MIN_SIZE_CELLS = 15
MIN_ELONGATION = 1.8
MAX_SHORT_SIDE_M = 8
MIN_RUN_M = 6
MAX_RUN_M = 150
MIN_RISE_M, MAX_RISE_M = 0.8, 25.0
ROI_MARGIN_M = 3.0       # padding around each candidate bbox

# --- stage 2: fine regrid + periodicity scoring ---------------------------
FINE_RES_MIN, FINE_RES_MAX = 0.05, 0.20
TARGET_PTS_PER_CELL = 6
MIN_PTS_FOR_FINE = 400
TREAD_LAG_LO, TREAD_LAG_HI = 0.20, 0.55   # plausible stair tread depth, m
STEP_RISE_LO, STEP_RISE_HI = 0.06, 0.28   # plausible single-step rise, m


def coarse_candidates(dem, transform, res):
    nan_mask = np.isnan(dem)
    if nan_mask.any():
        idx = ndimage.distance_transform_edt(nan_mask, return_distances=False,
                                              return_indices=True)
        dem_f = dem[tuple(idx)]
    else:
        dem_f = dem

    win = max(3, int(round(ROOF_WINDOW_M / res)))
    local_min = ndimage.minimum_filter(dem_f, size=win, mode="nearest")
    street_like = dem_f < (local_min + ROOF_MARGIN_M)

    gy, gx = np.gradient(dem_f, res)
    slope = np.hypot(gx, gy)
    steep = (slope > SLOPE_LO) & (slope < SLOPE_HI)

    cand_mask = steep & street_like & ~nan_mask
    lbl, n = ndimage.label(cand_mask, structure=np.ones((3, 3)))
    print(f"coarse pass: {cand_mask.sum()} steep/street-level cells, "
          f"{n} connected components")

    results = []
    for comp_id in range(1, n + 1):
        ys, xs = np.where(lbl == comp_id)
        size = len(ys)
        if size < MIN_SIZE_CELLS:
            continue
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        h_extent, w_extent = (r1 - r0 + 1) * res, (c1 - c0 + 1) * res
        long_side, short_side = max(h_extent, w_extent), min(h_extent, w_extent)
        elong = long_side / max(short_side, 1e-6)
        if elong < MIN_ELONGATION or short_side > MAX_SHORT_SIDE_M:
            continue
        if not (MIN_RUN_M <= long_side <= MAX_RUN_M):
            continue
        rise = float(dem_f[ys, xs].max() - dem_f[ys, xs].min())
        if not (MIN_RISE_M <= rise <= MAX_RISE_M):
            continue

        pad = int(round(ROI_MARGIN_M / res))
        pr0, pr1 = max(0, r0 - pad), min(dem.shape[0] - 1, r1 + pad)
        pc0, pc1 = max(0, c0 - pad), min(dem.shape[1] - 1, c1 + pad)

        # principal axis via PCA on the component's cell coordinates
        pts = np.stack([xs * res, (dem.shape[0] - ys) * res], axis=1).astype(np.float64)
        pts -= pts.mean(axis=0)
        cov = pts.T @ pts / max(len(pts) - 1, 1)
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, np.argmax(evals)]  # (dx, dy) in local x/(-row) meters

        ux0, uy0 = pixel_to_utm(transform, pc0, pr0)
        ux1, uy1 = pixel_to_utm(transform, pc1, pr1)
        results.append(dict(
            comp_id=comp_id, size=size, rows=(int(r0), int(r1)), cols=(int(c0), int(c1)),
            padded_rows=(int(pr0), int(pr1)), padded_cols=(int(pc0), int(pc1)),
            run_m=float(long_side), width_m=float(short_side), rise_m=rise,
            mean_slope=float(slope[ys, xs].mean()), axis=(float(axis[0]), float(axis[1])),
            utm_bbox=(float(min(ux0, ux1)), float(min(uy0, uy1)),
                      float(max(ux0, ux1)), float(max(uy0, uy1))),
        ))
    print(f"coarse pass: {len(results)} candidates after shape/rise filtering")
    return results


def build_id_grid(dem_shape, candidates):
    id_grid = np.full(dem_shape, -1, dtype=np.int32)
    for i, c in enumerate(candidates):
        r0, r1 = c["padded_rows"]
        c0, c1 = c["padded_cols"]
        id_grid[r0:r1 + 1, c0:c1 + 1] = i
    return id_grid


def collect_points_per_candidate(las_path, transform, id_grid, n_candidates):
    header = LasHeader(las_path)
    print(header.describe())
    buckets = [[] for _ in range(n_candidates)]
    h, w = id_grid.shape
    n_total = header.n_points_in_file
    for i, pts in iter_chunks(header, chunk_points=20_000_000):
        x, y = header.scale_xy(pts)
        z = (pts["Z"] * header.sz + header.oz).astype(np.float32)
        col = np.round((x - transform["minx"]) / transform["res"]).astype(np.int64)
        row = np.round((transform["maxy"] - y) / transform["res"]).astype(np.int64)
        valid = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        idx = np.full(len(pts), -1, dtype=np.int32)
        idx[valid] = id_grid[row[valid], col[valid]]
        hit = idx >= 0
        if hit.any():
            hx, hy, hz, hid = x[hit], y[hit], z[hit], idx[hit]
            order = np.argsort(hid)
            hx, hy, hz, hid = hx[order], hy[order], hz[order], hid[order]
            splits = np.searchsorted(hid, np.arange(n_candidates + 1))
            for k in range(n_candidates):
                s0, s1 = splits[k], splits[k + 1]
                if s1 > s0:
                    buckets[k].append(np.stack([hx[s0:s1], hy[s0:s1], hz[s0:s1]], axis=1))
        print(f"\r  scanning point cloud: {100 * (i + len(pts)) / n_total:5.1f}%",
              end="", flush=True)
    print()
    return [np.concatenate(b, axis=0) if b else np.empty((0, 3), np.float64)
            for b in buckets]


def score_periodicity(pts_xyz, axis):
    """Project raw points onto the ROI's long axis, detrend the overall
    ramp, and run a Lomb-Scargle periodogram (works directly on irregular,
    sparse point spacing - no binning-resolution artifacts) to look for a
    genuine spectral peak in the spatial-frequency band that corresponds to
    real stair tread depth, well above the noise floor at other
    frequencies. A smooth ramp or generic rough pavement has no such
    narrowband peak; real stair treads/risers do.
    """
    from scipy.signal import lombscargle

    n = len(pts_xyz)
    if n < MIN_PTS_FOR_FINE:
        return None

    xy = pts_xyz[:, :2]
    z = pts_xyz[:, 2].astype(np.float64)
    ax = np.array(axis, dtype=np.float64)
    ax /= np.linalg.norm(ax) + 1e-12
    perp = np.array([-ax[1], ax[0]])

    centroid = xy.mean(axis=0)
    s = (xy - centroid) @ ax          # along-axis coordinate, m
    t = (xy - centroid) @ perp        # across-axis (width) coordinate, m
    run_m = float(s.max() - s.min())
    if run_m < MIN_RUN_M:
        return None

    # linear detrend (remove the overall ramp slope) on the raw scatter
    A = np.vstack([s, np.ones_like(s)]).T
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    resid = z - A @ coef

    stair_freqs = np.linspace(1.0 / TREAD_LAG_HI, 1.0 / TREAD_LAG_LO, 150)
    noise_freqs = np.concatenate([np.linspace(0.3, 1.5, 40), np.linspace(6.0, 15.0, 40)])

    stair_power = lombscargle(s, resid, 2 * np.pi * stair_freqs, normalize=True)
    noise_power = lombscargle(s, resid, 2 * np.pi * noise_freqs, normalize=True)
    baseline = float(np.median(noise_power)) + 1e-9

    peak_i = int(np.argmax(stair_power))
    peak_power = float(stair_power[peak_i])
    is_local_max = bool(0 < peak_i < len(stair_power) - 1 and
        stair_power[peak_i] > stair_power[peak_i - 1] and
        stair_power[peak_i] > stair_power[peak_i + 1])
    snr = float(peak_power / baseline)
    tread_m = float(1.0 / stair_freqs[peak_i])

    # amplitude of the best-fit sinusoid at the peak frequency
    w = 2 * np.pi * stair_freqs[peak_i]
    Xb = np.vstack([np.cos(w * s), np.sin(w * s), np.ones_like(s)]).T
    coef2, *_ = np.linalg.lstsq(Xb, resid, rcond=None)
    amp = float(np.hypot(coef2[0], coef2[1]))
    step_rise_est = 2 * amp  # peak-to-trough of the sinusoid ~ one riser
    amp_ok = bool(STEP_RISE_LO <= step_rise_est <= STEP_RISE_HI)

    score = min(snr / 8.0, 1.0) * (1.0 if is_local_max else 0.4) * (1.0 if amp_ok else 0.3)
    n_steps_est = int(round(run_m / tread_m)) if tread_m > 0 else 0

    # profile for plotting: binned mean at ~tread/4 resolution, raw scatter
    # is too dense to ship in JSON
    bin_res = max(tread_m / 4.0, 0.05)
    nbins = max(int(round(run_m / bin_res)), 3)
    s0 = s.min()
    bidx = np.clip(((s - s0) / max(run_m, 1e-9) * nbins).astype(np.int64), 0, nbins - 1)
    counts = np.bincount(bidx, minlength=nbins)
    sums = np.bincount(bidx, weights=z, minlength=nbins)
    prof_idx = np.arange(nbins)
    has = counts > 0
    profile_z = np.full(nbins, np.nan)
    profile_z[has] = sums[has] / counts[has]
    if has.sum() >= 2:
        profile_z = np.interp(prof_idx, prof_idx[has], profile_z[has])
    resid_sums = np.bincount(bidx, weights=resid, minlength=nbins)
    profile_resid = np.full(nbins, np.nan)
    profile_resid[has] = resid_sums[has] / counts[has]
    if has.sum() >= 2:
        profile_resid = np.interp(prof_idx, prof_idx[has], profile_resid[has])

    return dict(
        n_points=n, tread_m=float(tread_m), snr=float(snr), peak_power=peak_power,
        is_local_max=is_local_max, step_rise_est_m=step_rise_est, amp_plausible=amp_ok,
        score=float(score), n_steps_est=n_steps_est, run_m=run_m,
        centroid_utm=(float(centroid[0]), float(centroid[1])),
        profile_s=(prof_idx * bin_res).tolist(), profile_z=profile_z.tolist(),
        profile_resid=profile_resid.tolist(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", default=os.path.join(ROOT, "terrain", "dem.npy"))
    ap.add_argument("--transform", default=os.path.join(ROOT, "terrain", "dem_transform.json"))
    ap.add_argument("--las", default=os.path.join(ROOT, "terrain", "corridor.las"))
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "stair_candidates"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dem = np.load(args.dem).astype(np.float64)
    transform = load_transform(args.transform)

    candidates = coarse_candidates(dem, transform, transform["res"])
    if not candidates:
        print("No coarse candidates found - loosen thresholds.")
        return

    id_grid = build_id_grid(dem.shape, candidates)
    buckets = collect_points_per_candidate(args.las, transform, id_grid, len(candidates))

    scored = []
    for c, pts in zip(candidates, buckets):
        s = score_periodicity(pts, c["axis"])
        if s is not None:
            scored.append({**c, **s})

    scored.sort(key=lambda d: d["score"], reverse=True)
    print(f"\n{len(scored)} candidates scored (of {len(candidates)} coarse hits)\n")

    header = f"{'rank':>4} {'score':>6} {'snr':>7} {'tread_m':>7} {'rise_m':>7} {'ok':>3} {'lmax':>4} {'run_m':>6} {'n_steps':>7}  utm centroid"
    print(header)
    for rank, d in enumerate(scored[:args.top], 1):
        print(f"{rank:>4} {d['score']:6.2f} {d['snr']:7.2f} {d['tread_m']:7.2f} "
              f"{d['step_rise_est_m']:7.2f} {'Y' if d['amp_plausible'] else 'n':>3} "
              f"{'Y' if d['is_local_max'] else 'n':>4} "
              f"{d['run_m']:6.1f} {d['n_steps_est']:7d}  "
              f"({d['centroid_utm'][0]:.1f}, {d['centroid_utm'][1]:.1f})")

    with open(os.path.join(args.out, "candidates.json"), "w") as f:
        json.dump(scored[:max(args.top, 20)], f, indent=2)
    print(f"\nwrote {args.out}/candidates.json (top {max(args.top, 20)})")


if __name__ == "__main__":
    main()
