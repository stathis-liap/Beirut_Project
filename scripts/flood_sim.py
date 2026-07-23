#!/usr/bin/env python3
"""2D rain-on-grid flood simulation on the DEM.

Inertial shallow-water scheme of Bates, Horritt & Fewtrell (2010), the
LISFLOOD-FP formulation: explicit, staggered grid, semi-implicit friction,
adaptive CFL timestep. Vectorized NumPy; a corridor-scale grid
(~2000 x 2000 at 1 m) runs a 1-hour storm in minutes.

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

import numpy as np


def simulate(dem, rain_mmh, duration, save_every, out_dir,
             n_manning=0.03, infil_mmh=None, sink=None,
             rain_stop=None, res=1.0, g=9.81, alpha=0.7,
             h_min=1e-4, valid=None):
    """`valid` marks surveyed cells: rain falls only there and water is
    removed outside them (open boundary along the crop polygon edge, not
    just the grid edge). NaN cells in `dem` must already be replaced with
    LOW terrain by the caller so water can exit toward them."""
    h, w = dem.shape
    depth = np.zeros((h, w), dtype=np.float64)
    # discharges on cell faces (staggered): qx between (i,j) and (i,j+1)
    qx = np.zeros((h, w - 1), dtype=np.float64)
    qy = np.zeros((h - 1, w), dtype=np.float64)

    rain_ms = rain_mmh / 1000.0 / 3600.0            # m/s of water column
    infil_ms = None
    if infil_mmh is not None:
        infil_ms = infil_mmh / 1000.0 / 3600.0
    rain_stop = duration if rain_stop is None else rain_stop
    if valid is None:
        valid = np.ones((h, w), dtype=bool)
    rain_cells = valid.astype(np.float64)
    n_valid = int(valid.sum())

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
        dt = max(np.floor(dt * 1000.0) / 1000.0, 1e-3)  # match flood_gpu quantization
        dt = min(dt, duration - t + 1e-9)

        # --- momentum: x faces ---
        zl, zr = dem[:, :-1] + depth[:, :-1], dem[:, 1:] + depth[:, 1:]
        hflow = np.maximum(zl, zr) - np.maximum(dem[:, :-1], dem[:, 1:])
        hflow = np.maximum(hflow, 0.0)
        slope = (zl - zr) / res
        active = hflow > h_min
        qn = np.zeros_like(qx)
        hf = np.where(active, hflow, 1.0)
        # denominator exponent is 7/3: g*dt*Sf*|q|/q with Sf = n^2 q^2 / h^(10/3)
        # and numerator g*h*dt*S -> g*dt*n^2*|q|/h^(7/3)  (Bates et al. 2010 eq. 11;
        # fixed 2026-07-13, was 10/3 which overestimated friction by 1/h)
        qn = (qx + g * hf * dt * slope) / (
            1.0 + g * dt * n_manning ** 2 * np.abs(qx) / hf ** (7.0 / 3.0))
        qx = np.where(active, qn, 0.0)

        # --- momentum: y faces (row 0 is north; +y flow = toward row 0) ---
        zt, zb = dem[:-1, :] + depth[:-1, :], dem[1:, :] + depth[1:, :]
        hflow = np.maximum(zt, zb) - np.maximum(dem[:-1, :], dem[1:, :])
        hflow = np.maximum(hflow, 0.0)
        slope = (zt - zb) / res
        active = hflow > h_min
        hf = np.where(active, hflow, 1.0)
        qn = (qy + g * hf * dt * slope) / (
            1.0 + g * dt * n_manning ** 2 * np.abs(qy) / hf ** (7.0 / 3.0))
        qy = np.where(active, qn, 0.0)

        # --- flux limiter: never drain more than the donor cell holds ---
        # (keeps the scheme stable on stairs/steep steps)
        out_x = np.clip(qx, -depth[:, 1:] * res / dt / 4, depth[:, :-1] * res / dt / 4)
        out_y = np.clip(qy, -depth[1:, :] * res / dt / 4, depth[:-1, :] * res / dt / 4)
        qx, qy = out_x, out_y

        # --- continuity ---
        dv = np.zeros_like(depth)
        dv[:, :-1] -= qx * dt / res
        dv[:, 1:] += qx * dt / res
        dv[:-1, :] -= qy * dt / res
        dv[1:, :] += qy * dt / res
        depth += dv

        if t < rain_stop:
            depth += rain_ms * dt * rain_cells
            vol_in += rain_ms * dt * n_valid * cell_area
        if infil_ms is not None:
            di = np.minimum(depth, infil_ms * dt)
            depth -= di
            vol_infil += di.sum() * cell_area
        if sink is not None:
            vol_sink += depth[sink].sum() * cell_area
            depth[sink] = 0.0

        # open boundary: water leaves at grid edge and outside the survey
        depth[0, :] = depth[-1, :] = depth[:, 0] = depth[:, -1] = 0.0
        depth[~valid] = 0.0
        np.maximum(depth, 0.0, out=depth)

        np.maximum(max_depth, depth, out=max_depth)
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
    valid = np.isfinite(dem)
    if not valid.all():
        # outside the crop polygon: low terrain + no rain + depth reset
        # in simulate() = open outflow boundary along the coverage edge
        dem = np.where(valid, dem, np.nanmin(dem) - 5.0)
        print(f"open boundary on {100 * (~valid).mean():.1f}% no-data cells")
    with open(args.transform) as f:
        res = json.load(f)["res"]
    infil = np.load(args.infil).astype(np.float64) if args.infil else None
    sink = np.load(args.sink).astype(bool) if args.sink else None

    print(f"DEM {dem.shape[1]} x {dem.shape[0]} at {res} m, "
          f"rain {args.rain} mm/h for {args.rain_stop or args.duration:.0f}s, "
          f"sim {args.duration:.0f}s")
    simulate(dem, args.rain, args.duration, args.save_every, args.out,
             n_manning=args.manning, infil_mmh=infil, sink=sink,
             rain_stop=args.rain_stop, res=res, valid=valid)


if __name__ == "__main__":
    main()
