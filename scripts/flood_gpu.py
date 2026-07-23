#!/usr/bin/env python3
"""GPU rain-on-grid flood solver (PLAN.md Phase B).

Same numerics as flood_sim.py — the Bates, Horritt & Fewtrell (2010)
inertial shallow-water scheme with semi-implicit Manning friction and the
donor-cell flux limiter — ported to torch (CUDA or CPU), fp32 state, with:

  - hyetograph rain (storms/<name>.json step functions)
  - rain-weight raster (roof/courtyard downspout rerouting, Phase A5)
  - spatially varying Manning n and infiltration rate
  - storm-drain inlets with per-inlet capacity caps (m3/s)
  - open boundary at grid edge / outside survey / water cells, with the
    outflow VOLUME accounted (mass balance closes to ~fp32 accuracy)
  - gauges: depth/velocity time series at probe points
  - hazard accumulators: max depth, max |v|, max h*(|v|+0.5)

Usage:
  python scripts/flood_gpu.py --terrain output/terrain_1.0 \
      --storm storms/v1_nov2025.json --out output/runs/S0__v1_nov2025
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from las_common import load_transform, utm_to_pixel

G = 9.81


def simulate(dem, res, steps, duration, out_dir, *, manning=0.03,
             infil_mmh=None, valid=None, water=None, rain_weight=None,
             drains=None, gauges=None, sources=None, save_every=60.0, device="auto",
             alpha=0.7, h_min=1e-4, save_frames=True, init_depth=None,
             progress=True, dtype="float32", limiter="scale"):
    """steps: list of (t0_s, t1_s, mm/h). drains: (rows, cols, cap_m3s)
    arrays. gauges: list of {'name','row','col'}. sources: list of
    {'row','col','series': [[t_s, q_m3s], ...]} point inflows, linearly
    interpolated in time (used by the EA Test 8A benchmark). Returns meta
    dict.

    limiter: 'scale' (default) rescales each cell's OUTflux so it never
    exports more volume than it holds - mass-conserving positivity fix
    that does not cap physical velocities. 'clip4' reproduces the legacy
    flood_sim.py clip (q <= h*res/4dt) exactly, for reference comparison
    only - it suppresses velocities whenever v > sqrt(g*hmax)/(4*alpha)."""
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    f32 = dict(dtype=getattr(torch, dtype), device=device)
    h, w = dem.shape
    area = res * res

    if valid is None:
        valid = np.isfinite(dem)
    dem_np = np.where(valid, dem, np.nanmin(dem[valid]) - 5.0).astype(np.float32)
    dem_t = torch.tensor(dem_np, **f32)

    n_np = np.broadcast_to(np.asarray(manning, dtype=np.float32), (h, w))
    n2x = torch.tensor(np.maximum(n_np[:, :-1], n_np[:, 1:]) ** 2, **f32)
    n2y = torch.tensor(np.maximum(n_np[:-1, :], n_np[1:, :]) ** 2, **f32)

    if rain_weight is None:
        rain_weight = valid.astype(np.float32)
    rw = torch.tensor(rain_weight.astype(np.float32), **f32)
    rw_sum = float(rain_weight.sum())

    infil_t = None
    if infil_mmh is not None and np.any(infil_mmh > 0):
        infil_t = torch.tensor(
            (np.asarray(infil_mmh, np.float32) / 3.6e6), **f32)  # m/s

    out_np = ~valid
    out_np[0, :] = out_np[-1, :] = out_np[:, 0] = out_np[:, -1] = True
    if water is not None:
        out_np = out_np | water
    keep = torch.tensor((~out_np).astype(np.float32), **f32)

    drain_idx = drain_cap = None
    if drains is not None and len(drains[0]) > 0:
        rr, cc, cap = drains
        drain_idx = (torch.tensor(rr.astype(np.int64), device=device),
                     torch.tensor(cc.astype(np.int64), device=device))
        drain_cap = torch.tensor(cap.astype(np.float32), **f32)

    src_list = []
    for s in (sources or []):
        ser = np.asarray(s["series"], dtype=np.float64)
        src_list.append((int(s["row"]), int(s["col"]), ser[:, 0], ser[:, 1]))

    depth = torch.zeros((h, w), **f32)
    if init_depth is not None:
        depth = torch.tensor(init_depth.astype(np.float32), **f32)
    qx = torch.zeros((h, w - 1), **f32)
    qy = torch.zeros((h - 1, w), **f32)
    max_depth = torch.zeros((h, w), **f32)
    max_vel = torch.zeros((h, w), **f32)
    max_haz = torch.zeros((h, w), **f32)

    os.makedirs(out_dir, exist_ok=True)
    vol_in = vol_infil = vol_drain = vol_out = 0.0
    saved, gauge_rows, outflow_series, storage_series = [], [], [], []
    last_out, last_out_t = 0.0, 0.0
    t, next_save, it, si = 0.0, 0.0, 0, 0
    t0_wall = time.time()

    def rain_now(tt):
        for t0s, t1s, mmh in steps:
            if t0s <= tt < t1s:
                return mmh / 3.6e6  # m/s
        return 0.0

    while t < duration:
        hmax = float(depth.max().item())
        dt = min(alpha * res / math.sqrt(G * max(hmax, 0.01)), 5.0)
        # quantize so fp32/fp64 runs take identical step sequences
        dt = max(math.floor(dt * 1000.0) / 1000.0, 1e-3)
        dt = min(dt, duration - t + 1e-9)

        # momentum x
        zl = dem_t[:, :-1] + depth[:, :-1]
        zr = dem_t[:, 1:] + depth[:, 1:]
        hflow = torch.clamp(torch.maximum(zl, zr) -
                            torch.maximum(dem_t[:, :-1], dem_t[:, 1:]), min=0.0)
        active = hflow > h_min
        hfx = torch.where(active, hflow, torch.ones_like(hflow))
        qn = (qx + G * hfx * dt * (zl - zr) / res) / (
            1.0 + G * dt * n2x * torch.abs(qx) / hfx ** (7.0 / 3.0))
        qx = torch.where(active, qn, torch.zeros_like(qn))
        act_x = active

        # momentum y
        zt = dem_t[:-1, :] + depth[:-1, :]
        zb = dem_t[1:, :] + depth[1:, :]
        hflow = torch.clamp(torch.maximum(zt, zb) -
                            torch.maximum(dem_t[:-1, :], dem_t[1:, :]), min=0.0)
        active = hflow > h_min
        hfy = torch.where(active, hflow, torch.ones_like(hflow))
        qn = (qy + G * hfy * dt * (zt - zb) / res) / (
            1.0 + G * dt * n2y * torch.abs(qy) / hfy ** (7.0 / 3.0))
        qy = torch.where(active, qn, torch.zeros_like(qn))
        act_y = active

        if limiter == "clip4":      # legacy flood_sim.py behavior
            qx = torch.clamp(qx, min=-depth[:, 1:] * res / dt / 4,
                             max=depth[:, :-1] * res / dt / 4)
            qy = torch.clamp(qy, min=-depth[1:, :] * res / dt / 4,
                             max=depth[:-1, :] * res / dt / 4)
        else:                       # outflux scaling: positivity, mass exact
            pos_x = torch.clamp(qx, min=0)
            neg_x = torch.clamp(-qx, min=0)
            pos_y = torch.clamp(qy, min=0)
            neg_y = torch.clamp(-qy, min=0)
            outr = torch.zeros_like(depth)
            outr[:, :-1] += pos_x
            outr[:, 1:] += neg_x
            outr[:-1, :] += pos_y
            outr[1:, :] += neg_y
            s = torch.clamp(depth * res / (outr * dt + 1e-12), max=1.0)
            qx = pos_x * s[:, :-1] - neg_x * s[:, 1:]
            qy = pos_y * s[:-1, :] - neg_y * s[1:, :]

        # continuity
        dv = torch.zeros_like(depth)
        dv[:, :-1] -= qx * dt / res
        dv[:, 1:] += qx * dt / res
        dv[:-1, :] -= qy * dt / res
        dv[1:, :] += qy * dt / res
        depth = depth + dv

        i_ms = rain_now(t)
        if i_ms > 0:
            depth = depth + i_ms * dt * rw
            vol_in += i_ms * dt * rw_sum * area
        for r, c, ts, qs in src_list:
            q = float(np.interp(t, ts, qs))
            if q != 0.0:
                depth[r, c] += q * dt / area
                vol_in += q * dt
        if infil_t is not None:
            di = torch.minimum(depth, infil_t * dt)
            depth = depth - di
            vol_infil += float(di.sum().item()) * area
        if drain_idx is not None:
            d = depth[drain_idx]
            take = torch.minimum(d, drain_cap * dt / area)
            depth[drain_idx] = d - take
            vol_drain += float(take.sum().item()) * area

        # open boundary: count then remove
        esc = depth * (1.0 - keep)
        vol_out += float(esc.sum().item()) * area
        depth = torch.clamp(depth * keep, min=0.0)

        torch.maximum(max_depth, depth, out=max_depth)
        if it % 5 == 0:
            # face velocity = q / face flow depth (what momentum actually
            # used) - dividing by the cell's post-update depth explodes in
            # freshly drained cells and poisons the hazard maps
            vfx = torch.where(act_x, qx / hfx, torch.zeros_like(qx))
            vfy = torch.where(act_y, qy / hfy, torch.zeros_like(qy))
            vx = torch.zeros_like(depth)
            vy = torch.zeros_like(depth)
            vx[:, :-1] += vfx / 2
            vx[:, 1:] += vfx / 2
            vy[:-1, :] += vfy / 2
            vy[1:, :] += vfy / 2
            vel = torch.hypot(vx, vy)
            vel = torch.where(depth > 0.01, vel, torch.zeros_like(vel))
            torch.maximum(max_vel, vel, out=max_vel)
            torch.maximum(max_haz, depth * (vel + 0.5), out=max_haz)

        if t >= next_save:
            if save_frames:
                fn = f"depth_{int(round(t)):06d}.npy"
                np.save(os.path.join(out_dir, fn),
                        depth.cpu().numpy().astype(np.float16))
                saved.append(fn)
            storage = float(depth.sum().item()) * area
            storage_series.append([round(t, 1), round(storage, 1)])
            if t > 0:
                rate = (vol_out - last_out) / max(t - last_out_t, 1e-9)
                outflow_series.append([round(t, 1), round(rate, 3)])
            last_out, last_out_t = vol_out, t
            if gauges:
                dc = depth.cpu().numpy()
                vc = max_vel.cpu().numpy()
                gauge_rows.append(
                    [round(t, 1)] +
                    [round(float(dc[g["row"], g["col"]]), 4) for g in gauges] +
                    [round(float(vc[g["row"], g["col"]]), 3) for g in gauges])
            if progress:
                print(f"  t={t:7.1f}s dt={dt:5.2f}s rain={i_ms * 3.6e6:5.1f}mm/h "
                      f"max_h={hmax:5.2f} storage={storage:9.0f}m3 "
                      f"out={vol_out:8.0f}m3 [{time.time() - t0_wall:5.0f}s wall]",
                      flush=True)
            next_save += save_every
            si += 1
        t += dt
        it += 1

    # NB: name must not match the depth_*.npy frame glob
    np.save(os.path.join(out_dir, "final_depth.npy"),
            depth.cpu().numpy().astype(np.float32))
    np.save(os.path.join(out_dir, "max_depth.npy"),
            max_depth.cpu().numpy().astype(np.float32))
    np.save(os.path.join(out_dir, "max_vel.npy"),
            max_vel.cpu().numpy().astype(np.float32))
    np.save(os.path.join(out_dir, "max_hazard.npy"),
            max_haz.cpu().numpy().astype(np.float32))

    stored = float(depth.sum().item()) * area
    closure = vol_in - (vol_infil + vol_drain + vol_out + stored)
    meta = {
        "steps_mmh": steps, "duration": duration, "res": res,
        "device": device, "n_iterations": it,
        "wall_s": round(time.time() - t0_wall, 1),
        "frames": saved,
        "vol_rain_m3": round(vol_in, 2),
        "vol_infiltrated_m3": round(vol_infil, 2),
        "vol_drained_m3": round(vol_drain, 2),
        "vol_outflow_m3": round(vol_out, 2),
        "vol_stored_end_m3": round(stored, 2),
        "closure_m3": round(closure, 2),
        "closure_rel": round(closure / max(vol_in, 1e-9), 6),
        "outflow_series_m3s": outflow_series,
        "storage_series_m3": storage_series,
    }
    if gauges:
        meta["gauges"] = [{"name": g["name"], "row": g["row"], "col": g["col"]}
                          for g in gauges]
        import csv
        with open(os.path.join(out_dir, "gauges.csv"), "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["t_s"] + [f"h_{g['name']}" for g in gauges] +
                          [f"vmax_{g['name']}" for g in gauges])
            wcsv.writerows(gauge_rows)
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if progress:
        print(f"done in {meta['wall_s']:.0f}s wall, {it} steps. mass balance: "
              f"rain {vol_in:.0f} = infil {vol_infil:.0f} + drains {vol_drain:.0f} "
              f"+ outflow {vol_out:.0f} + stored {stored:.0f} "
              f"(closure {meta['closure_rel'] * 100:.3f}%)")
    return meta


def load_terrain(terrain):
    dem = np.load(os.path.join(terrain, "dem.npy")).astype(np.float32)
    t = load_transform(os.path.join(terrain, "dem_transform.json"))
    masks = np.load(os.path.join(terrain, "masks.npz"))
    manning = np.load(os.path.join(terrain, "manning.npy"))
    infil = np.load(os.path.join(terrain, "infil_mmh.npy"))
    rain_w = np.load(os.path.join(terrain, "rain_weight.npy"))
    gauges = []
    gpath = os.path.join(terrain, "gauges.json")
    if os.path.exists(gpath):
        with open(gpath) as f:
            for g in json.load(f):
                c, r = utm_to_pixel(t, g["x"], g["y"])
                gauges.append({"name": g["name"], "row": int(r), "col": int(c)})
    return dem, t, masks, manning, infil, rain_w, gauges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", required=True)
    ap.add_argument("--storm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drains", help="drains.npz (rows, cols, cap)")
    ap.add_argument("--duration", type=float, help="override storm sim duration")
    ap.add_argument("--save-every", type=float, default=60.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-frames", action="store_true")
    args = ap.parse_args()

    dem, t, masks, manning, infil, rain_w, gauges = load_terrain(args.terrain)
    with open(args.storm) as f:
        storm = json.load(f)
    drains = None
    if args.drains:
        dz = np.load(args.drains)
        drains = (dz["rows"], dz["cols"], dz["cap"])
        print(f"{len(dz['rows'])} drain inlets, "
              f"total capacity {dz['cap'].sum():.2f} m3/s")

    print(f"DEM {dem.shape[1]} x {dem.shape[0]} at {t['res']} m | "
          f"storm {storm['name']} ({storm['total_mm']} mm) | "
          f"sim {args.duration or storm['duration']:.0f}s")
    simulate(dem, t["res"], storm["steps"],
             args.duration or storm["duration"], args.out,
             manning=manning, infil_mmh=infil, valid=masks["valid"],
             water=masks["water"], rain_weight=rain_w, drains=drains,
             gauges=gauges, save_every=args.save_every, device=args.device,
             save_frames=not args.no_frames)


if __name__ == "__main__":
    main()
