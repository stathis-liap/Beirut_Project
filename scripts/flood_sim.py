#!/usr/bin/env python3
"""2D rain-on-grid flood simulation on the DEM.

Inertial shallow-water scheme of Bates, Horritt & Fewtrell (2010), the
LISFLOOD-FP formulation: explicit, staggered grid, semi-implicit friction,
adaptive CFL timestep. The per-step stencil (x/y momentum + continuity) is
fused into one Numba-jitted, multi-threaded kernel instead of ~20 separate
NumPy passes - NumPy's per-op temporary-array allocation and dispatch
overhead was the actual bottleneck at these grid sizes (hundreds of
thousands of cells, tens of thousands of adaptive timesteps), not raw FLOPs.

Usage:
  python scripts/flood_sim.py --dem output/dem.npy \
      --rain 30 --duration 3600 --save-every 30 --out output/run_baseline

Optional what-if inputs (produced by scenario.py):
  --dem-override <npy>      modified DEM (walls, channels, fill)
  --infil <npy>             per-cell infiltration rate map, mm/h
  --sink <npy>              bool map: cells that drain out (culvert/storm drain)
"""

import argparse
import json
import os

import numba as nb
import numpy as np


@nb.njit(parallel=True, fastmath=True, cache=True)
def _step_kernel(dem, depth, qx, qy, max_depth, dt, res, g, n_manning, h_min,
                  rain_ms, raining, infil_ms, sink_mask, cell_area):
    """One explicit LISFLOOD-FP timestep, fused into a single compiled pass.

    Three sequential sweeps (x-momentum, y-momentum, continuity) - each
    parallelized over rows - replace the original's ~20 whole-array NumPy
    ops per step. Semantics (order of operations, flux limiter, mass
    accounting) match the original vectorized implementation exactly.
    """
    h, w = depth.shape
    new_qx = np.empty_like(qx)
    new_qy = np.empty_like(qy)
    new_depth = np.empty_like(depth)

    # --- x-direction momentum (face j: between column j and j+1) ---
    for i in nb.prange(h):
        for j in range(w - 1):
            zl = dem[i, j] + depth[i, j]
            zr = dem[i, j + 1] + depth[i, j + 1]
            hflow = max(zl, zr) - max(dem[i, j], dem[i, j + 1])
            if hflow < 0.0:
                hflow = 0.0
            if hflow <= h_min:
                new_qx[i, j] = 0.0
                continue
            slope = (zl - zr) / res
            q_old = qx[i, j]
            denom = 1.0 + g * dt * n_manning ** 2 * abs(q_old) / hflow ** (10.0 / 3.0)
            q_new = (q_old + g * hflow * dt * slope) / denom
            lo = -depth[i, j + 1] * res / dt / 4.0
            hi = depth[i, j] * res / dt / 4.0
            if q_new < lo:
                q_new = lo
            elif q_new > hi:
                q_new = hi
            new_qx[i, j] = q_new

    # --- y-direction momentum (face i: between row i and i+1; +y = south) ---
    for i in nb.prange(h - 1):
        for j in range(w):
            zt = dem[i, j] + depth[i, j]
            zb = dem[i + 1, j] + depth[i + 1, j]
            hflow = max(zt, zb) - max(dem[i, j], dem[i + 1, j])
            if hflow < 0.0:
                hflow = 0.0
            if hflow <= h_min:
                new_qy[i, j] = 0.0
                continue
            slope = (zt - zb) / res
            q_old = qy[i, j]
            denom = 1.0 + g * dt * n_manning ** 2 * abs(q_old) / hflow ** (10.0 / 3.0)
            q_new = (q_old + g * hflow * dt * slope) / denom
            lo = -depth[i + 1, j] * res / dt / 4.0
            hi = depth[i, j] * res / dt / 4.0
            if q_new < lo:
                q_new = lo
            elif q_new > hi:
                q_new = hi
            new_qy[i, j] = q_new

    # --- continuity + rain + infiltration + sink + open boundary ---
    vol_infil = 0.0
    vol_sink = 0.0
    for i in nb.prange(h):
        row_infil = 0.0
        row_sink = 0.0
        for j in range(w):
            d = depth[i, j]
            if j >= 1:
                d += new_qx[i, j - 1] * dt / res
            if j < w - 1:
                d -= new_qx[i, j] * dt / res
            if i >= 1:
                d += new_qy[i - 1, j] * dt / res
            if i < h - 1:
                d -= new_qy[i, j] * dt / res

            if raining:
                d += rain_ms * dt

            im = infil_ms[i, j]
            if im > 0.0:
                cap = im * dt
                di = d if d < cap else cap
                d -= di
                row_infil += di

            if sink_mask[i, j]:
                row_sink += d
                d = 0.0

            if i == 0 or i == h - 1 or j == 0 or j == w - 1:
                d = 0.0
            if d < 0.0:
                d = 0.0

            new_depth[i, j] = d
            if d > max_depth[i, j]:
                max_depth[i, j] = d
        vol_infil += row_infil
        vol_sink += row_sink

    return new_depth, new_qx, new_qy, vol_infil * cell_area, vol_sink * cell_area


def simulate(dem, rain_mmh, duration, save_every, out_dir,
             n_manning=0.03, infil_mmh=None, sink=None,
             rain_stop=None, res=1.0, g=9.81, alpha=0.7,
             h_min=1e-4):
    h, w = dem.shape
    depth = np.zeros((h, w), dtype=np.float64)
    # discharges on cell faces (staggered): qx between (i,j) and (i,j+1)
    qx = np.zeros((h, w - 1), dtype=np.float64)
    qy = np.zeros((h - 1, w), dtype=np.float64)

    rain_ms = rain_mmh / 1000.0 / 3600.0            # m/s of water column
    infil_ms = (infil_mmh / 1000.0 / 3600.0 if infil_mmh is not None
                else np.zeros((h, w), dtype=np.float64))
    sink_mask = sink if sink is not None else np.zeros((h, w), dtype=np.bool_)
    rain_stop = duration if rain_stop is None else rain_stop

    os.makedirs(out_dir, exist_ok=True)
    max_depth = np.zeros_like(depth)
    max_vel = np.zeros_like(depth)
    saved = []
    t, next_save, it = 0.0, 0.0, 0
    vol_in = vol_infil = vol_sink = 0.0
    cell_area = res * res

    while t < duration:
        hmax = depth.max()
        dt = min(alpha * res / np.sqrt(g * max(hmax, 0.01)), 5.0)
        dt = min(dt, duration - t + 1e-9)
        raining = t < rain_stop

        depth, qx, qy, d_infil, d_sink = _step_kernel(
            dem, depth, qx, qy, max_depth, dt, res, g, n_manning, h_min,
            rain_ms, raining, infil_ms, sink_mask, cell_area)
        if raining:
            vol_in += rain_ms * dt * depth.size * cell_area
        vol_infil += d_infil
        vol_sink += d_sink

        # velocity magnitude at cell centers (for hazard maps)
        if it % 10 == 0:
            vx = np.zeros_like(depth)
            vy = np.zeros_like(depth)
            vx[:, :-1] += qx / 2
            vx[:, 1:] += qx / 2
            vy[:-1, :] += qy / 2
            vy[1:, :] += qy / 2
            vel = np.hypot(vx, vy) / np.maximum(depth, h_min)
            vel[depth < 0.01] = 0.0
            np.maximum(max_vel, vel, out=max_vel)

        if t >= next_save:
            fn = os.path.join(out_dir, f"depth_{int(round(t)):06d}.npy")
            np.save(fn, depth.astype(np.float32))
            saved.append(os.path.basename(fn))
            storage = depth.sum() * cell_area
            print(f"  t={t:7.1f}s dt={dt:5.2f}s max_h={depth.max():5.2f} m "
                  f"wet={100 * (depth > 0.02).mean():4.1f}% "
                  f"storage={storage:9.0f} m3")
            next_save += save_every
        t += dt
        it += 1

    np.save(os.path.join(out_dir, "max_depth.npy"), max_depth.astype(np.float32))
    np.save(os.path.join(out_dir, "max_vel.npy"), max_vel.astype(np.float32))
    meta = {
        "rain_mmh": rain_mmh, "duration": duration, "n_manning": n_manning,
        "res": res, "frames": saved,
        "vol_rain_m3": vol_in, "vol_infiltrated_m3": vol_infil,
        "vol_sunk_m3": vol_sink,
        "vol_stored_end_m3": float(depth.sum() * cell_area),
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done: {len(saved)} frames -> {out_dir}/")
    print(f"mass balance: rain {vol_in:.0f} m3, infiltrated {vol_infil:.0f}, "
          f"drained(sink) {vol_sink:.0f}, stored at end {meta['vol_stored_end_m3']:.0f} "
          f"(rest left through the open boundary)")
    return max_depth, max_vel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", default="output/dem.npy")
    ap.add_argument("--dem-override")
    ap.add_argument("--transform", default="output/dem_transform.json")
    ap.add_argument("--rain", type=float, default=30.0, help="mm/h")
    ap.add_argument("--rain-stop", type=float, default=None,
                    help="stop rain at this second (default: whole run)")
    ap.add_argument("--duration", type=float, default=3600.0)
    ap.add_argument("--save-every", type=float, default=30.0)
    ap.add_argument("--manning", type=float, default=0.03)
    ap.add_argument("--infil", help="npy infiltration map mm/h")
    ap.add_argument("--sink", help="npy bool sink map")
    ap.add_argument("--out", default="output/run")
    args = ap.parse_args()

    dem = np.load(args.dem_override or args.dem).astype(np.float64)
    with open(args.transform) as f:
        res = json.load(f)["res"]
    infil = np.load(args.infil).astype(np.float64) if args.infil else None
    sink = np.load(args.sink).astype(bool) if args.sink else None

    print(f"DEM {dem.shape[1]} x {dem.shape[0]} at {res} m, "
          f"rain {args.rain} mm/h for {args.rain_stop or args.duration:.0f}s, "
          f"sim {args.duration:.0f}s")
    simulate(dem, args.rain, args.duration, args.save_every, args.out,
             n_manning=args.manning, infil_mmh=infil, sink=sink,
             rain_stop=args.rain_stop, res=res)


if __name__ == "__main__":
    main()
