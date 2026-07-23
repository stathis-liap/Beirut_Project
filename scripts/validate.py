#!/usr/bin/env python3
"""Solver validation suite (PLAN.md Phase D1/D4).

  1. closed box     rain into a walled basin: volume conservation
  2. planar runoff  steady rain on a 2% plane: outflow == i*A (kinematic),
                    mid-slope depth vs Manning normal depth
  3. lake at rest   a bowl with still water: no spurious currents
  4. equivalence    synthetic urban DEM: numpy reference (flood_sim) vs
                    torch CPU vs torch CUDA max-depth agreement

Usage: python scripts/validate.py [--device auto]
Exit code 0 = all pass.
"""

import argparse
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from flood_gpu import simulate

FAILURES = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def tmpdir():
    return tempfile.mkdtemp(prefix="validate_", dir="output")


def test_closed_box(device):
    print("== closed box: conservation ==")
    h, w = 120, 120
    dem = np.zeros((h, w), dtype=np.float32)
    dem[:8, :] = dem[-8:, :] = dem[:, :8] = dem[:, -8:] = 30.0  # walls
    rain_w = np.zeros((h, w), dtype=np.float32)
    rain_w[8:-8, 8:-8] = 1.0
    out = tmpdir()
    meta = simulate(dem, 1.0, [(0, 600, 20.0)], 900, out, manning=0.03,
                    rain_weight=rain_w, save_every=300, device=device,
                    save_frames=False, progress=False)
    err = abs(meta["vol_rain_m3"] -
              (meta["vol_stored_end_m3"] + meta["vol_outflow_m3"]))
    rel = err / meta["vol_rain_m3"]
    check("volume closure", rel < 1e-3, f"rel err {rel:.2e}")
    check("no leakage over walls", meta["vol_outflow_m3"] < 0.01 * meta["vol_rain_m3"],
          f"outflow {meta['vol_outflow_m3']:.2f} m3 of {meta['vol_rain_m3']:.0f}")
    shutil.rmtree(out)


def test_planar_runoff(device):
    print("== planar runoff: steady state vs analytic ==")
    h, w = 400, 60
    S, n, i_mmh, res = 0.02, 0.03, 50.0, 1.0
    i_ms = i_mmh / 3.6e6
    rows = np.arange(h)[:, None].astype(np.float32)
    dem = ((h - 1 - rows) * S * res) * np.ones((1, w), dtype=np.float32)
    dem[:, :5] += 30.0
    dem[:, -5:] += 30.0            # side walls; open outflow at bottom edge
    dem[0, :] += 30.0              # top wall
    rain_w = np.zeros((h, w), dtype=np.float32)
    rain_w[1:-1, 5:-5] = 1.0
    out = tmpdir()
    meta = simulate(dem, res, [(0, 7200, i_mmh)], 7200, out, manning=n,
                    rain_weight=rain_w, save_every=600, device=device,
                    save_frames=False, progress=False)
    A = rain_w.sum() * res * res
    q_expect = i_ms * A                       # m3/s at equilibrium
    rate = meta["outflow_series_m3s"][-1][1]
    check("steady outflow == i*A", abs(rate - q_expect) / q_expect < 0.05,
          f"{rate:.4f} vs {q_expect:.4f} m3/s")
    # Manning normal depth at row r: upstream length L = (r-0.5)*res
    # (rain starts at row 1), q = i*L, h = (n q / sqrt(S))^(3/5).
    # Compare the STEADY (final) depth - max depth includes the filling
    # transient.
    row = 300
    L = (row - 0.5) * res
    h_expect = (n * i_ms * L / math.sqrt(S)) ** 0.6
    fd = np.load(os.path.join(out, "final_depth.npy"))
    h_sim = float(np.median(fd[row, 10:-10]))
    check("downslope steady depth vs Manning",
          abs(h_sim - h_expect) / h_expect < 0.15,
          f"{h_sim * 1000:.1f} vs {h_expect * 1000:.1f} mm at L={L:.0f} m")
    shutil.rmtree(out)


def test_lake_at_rest(device):
    print("== lake at rest: no spurious currents ==")
    h, w = 150, 150
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (h / 2) ** 2
    dem = (5.0 * r2).astype(np.float32)        # parabolic bowl
    wl = 2.0
    init = np.clip(wl - dem, 0, None).astype(np.float32)
    out = tmpdir()
    simulate(dem, 1.0, [], 600, out, manning=0.03, init_depth=init,
             rain_weight=np.zeros_like(dem), save_every=600, device=device,
             save_frames=True, progress=False)
    mv = np.load(os.path.join(out, "max_vel.npy"))
    frames = sorted(f for f in os.listdir(out) if f.startswith("depth_"))
    end = np.load(os.path.join(out, frames[-1])).astype(np.float32)
    surf = dem + end
    wet = end > 0.05
    rng = float(surf[wet].max() - surf[wet].min())
    check("still water stays still", float(mv[wet].max()) < 5e-3,
          f"max |v| {float(mv[wet].max()):.2e} m/s")
    check("flat surface preserved", rng < 0.005, f"surface range {rng * 1000:.2f} mm")
    shutil.rmtree(out)


def test_equivalence(device):
    """Port gate: torch fp64 with the legacy clip4 limiter must reproduce
    the validated numpy solver almost exactly (same numerics, same
    arithmetic). Then fp32 + production 'scale' limiter is compared to its
    own fp64 twin to MEASURE drift under the production configuration."""
    print("== backend equivalence on synthetic urban DEM ==")
    from test_synthetic import synthetic_dem
    from flood_sim import simulate as simulate_np
    dem, street = synthetic_dem()
    out_np, out_64, out_s64, out_s32 = tmpdir(), tmpdir(), tmpdir(), tmpdir()
    md_ref, _ = simulate_np(dem, rain_mmh=30, duration=900, save_every=900,
                            out_dir=out_np, rain_stop=600, res=1.0)
    common = dict(manning=0.03, save_every=900, save_frames=False,
                  progress=False)
    simulate(dem, 1.0, [(0, 600, 30.0)], 900, out_64, device="cpu",
             dtype="float64", limiter="clip4", **common)
    md64 = np.load(os.path.join(out_64, "max_depth.npy"))
    wet = md_ref > 0.05
    mad = float(np.abs(md_ref[wet] - md64[wet]).mean())
    mx = float(np.abs(md_ref[wet] - md64[wet]).max())
    check("port correctness (fp64+clip4 vs numpy)", mx < 0.002,
          f"mean|d| {mad * 1e6:.1f} um, max|d| {mx * 1000:.3f} mm")

    dev = "cuda" if device != "cpu" else "cpu"
    simulate(dem, 1.0, [(0, 600, 30.0)], 900, out_s64, device="cpu",
             dtype="float64", limiter="scale", **common)
    simulate(dem, 1.0, [(0, 600, 30.0)], 900, out_s32, device=dev,
             dtype="float32", limiter="scale", **common)
    # fp32 rounding seeds chaotic divergence at stair-pond edges, so a
    # cellwise gate is meaningless; require STATISTICAL equivalence.
    a = np.load(os.path.join(out_s64, "max_depth.npy"))
    b = np.load(os.path.join(out_s32, "max_depth.npy"))
    wa, wb = a[a > 0.05], b[b > 0.05]
    dvol = abs(a.sum() - b.sum()) / max(a.sum(), 1e-9)
    darea = abs(wa.size - wb.size) / max(wa.size, 1)
    dp = [abs(np.percentile(wa, p) - np.percentile(wb, p)) /
          max(np.percentile(wa, p), 1e-9) for p in (50, 90, 99)]
    corr = float(np.corrcoef(a[a > 0.05], b[a > 0.05])[0, 1])
    check(f"fp32 statistical equivalence ({dev})",
          dvol < 0.02 and darea < 0.05 and max(dp) < 0.10,
          f"vol d {dvol * 100:.2f}%, wet-area d {darea * 100:.1f}%, "
          f"p50/90/99 d {[f'{x * 100:.1f}%' for x in dp]} "
          f"(cellwise corr {corr:.3f} - chaotic, info only)")
    for d in (out_np, out_64, out_s64, out_s32):
        shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    os.makedirs("output", exist_ok=True)

    test_closed_box(device)
    test_planar_runoff(device)
    test_lake_at_rest(device)
    test_equivalence(device)

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("\nALL VALIDATION TESTS PASSED")


if __name__ == "__main__":
    main()
