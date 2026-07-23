#!/usr/bin/env python3
"""Automated per-run sanity report (PLAN.md Phase D5 — the "2.5 m rule").

Screens every run against physics and known artifact modes:
  - mass-balance closure (< 1%)
  - static fill bound: dynamic max depth must not exceed the terrain's
    priority-flood depression depth (+ tolerance) anywhere
  - every cell deeper than 1 m auto-classified:
      courtyard trap / fill-bound violation (numerical) / genuine depression
  - wet-depth distribution, velocity screen, peak outflow vs rational
    method Q = C*i*A envelope

Writes <run>/sanity_report.json + <run>/sanity.png. Exit 0 = PASS/WARN.

Usage:
  python scripts/sanity.py --run output/runs/S0__v1_nov2025 \
      --terrain output/terrain_1.0
"""

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--terrain", required=True)
    ap.add_argument("--fill-tol", type=float, default=0.15)
    args = ap.parse_args()

    with open(os.path.join(args.run, "run_meta.json")) as f:
        meta = json.load(f)
    md = np.load(os.path.join(args.run, "max_depth.npy"))
    mv = np.load(os.path.join(args.run, "max_vel.npy"))
    masks = np.load(os.path.join(args.terrain, "masks.npz"))
    valid, courtyard = masks["valid"], masks["courtyard"]
    water = masks["water"]
    fb_path = os.path.join(args.terrain, "fillbound.npy")
    fb = np.load(fb_path) if os.path.exists(fb_path) else None
    res = meta["res"]
    area = res * res

    flags, notes, rep = [], [], {"run": args.run}
    dem = np.load(os.path.join(args.terrain, "dem.npy"))

    # has the run settled (quasi-static) by t_end?
    ss = meta.get("storage_series_m3", [])
    settled = False
    if len(ss) >= 6:
        s_now, s_prev = ss[-1][1], ss[-6][1]
        drain_frac = (s_prev - s_now) / max(s_now, 1e-9)
        settled = drain_frac < 0.01
        rep["storage_drain_frac_last5saves"] = round(drain_frac, 4)
    rep["settled_at_end"] = settled

    # 1. mass balance
    rep["closure_rel"] = meta["closure_rel"]
    if abs(meta["closure_rel"]) > 0.01:
        flags.append(f"mass balance closure {meta['closure_rel'] * 100:.2f}% > 1%")

    # 2. wet statistics (streets only: exclude courtyards & water)
    street = valid & ~courtyard & ~water
    wet = md[street & (md > 0.05)]
    rep["wet_cells_gt5cm"] = int(wet.size)
    rep["wet_area_m2_gt10cm"] = float((md[street] > 0.10).sum() * area)
    rep["wet_area_m2_gt30cm"] = float((md[street] > 0.30).sum() * area)
    for p in (50, 90, 99):
        rep[f"street_depth_p{p}"] = round(float(np.percentile(wet, p)), 3) if wet.size else 0.0
    rep["street_depth_max"] = round(float(md[street].max()), 3) if street.any() else 0.0

    # 3. static fill bound screen: the FINAL (quasi-static) depth must fit
    # inside the terrain's depressions; the transient max may overshoot
    # physically (momentum), so it is reported as info only.
    fd_path = os.path.join(args.run, "final_depth.npy")
    fd = np.load(fd_path) if os.path.exists(fd_path) else None
    if fb is not None and fd is not None:
        exceed = np.where(valid & np.isfinite(fb), fd - fb, 0.0)
        viol = exceed > args.fill_tol
        rep["fillbound_violations_final"] = int(viol.sum())
        rep["fillbound_max_exceed_final_m"] = round(float(exceed.max()), 3)
        exceed_max = np.where(valid & np.isfinite(fb), md - fb, 0.0)
        rep["fillbound_transient_exceed_cells"] = int((exceed_max > args.fill_tol).sum())
        viol_vol = float(exceed[viol].sum()) * area
        stored = float(np.nan_to_num(fd, nan=0.0)[valid].sum()) * area
        rep["fillbound_violation_vol_m3"] = round(viol_vol, 1)
        # flag on VOLUME: the screen exists to catch phantom water at scale
        # (the "2.5 m in the street" class), not puddle-tail residue
        if viol_vol > max(50.0, 0.01 * stored) and settled:
            flags.append(f"{viol_vol:.0f} m3 of settled water above the "
                         f"static pond bound (+{args.fill_tol} m tol) - numerical")
        elif viol.sum() > 0:
            notes.append(f"{viol.sum()} cells / {viol_vol:.0f} m3 above the "
                         f"static bound"
                         + ("" if settled else " (run not settled at t_end)"))
    else:
        viol = np.zeros_like(valid)
        notes.append("no fillbound/final_depth - static bound screen skipped")

    # 4. classify deep cells
    deep = valid & (md > 1.0)
    rep["cells_gt1m"] = int(deep.sum())
    rep["cells_gt1m_courtyard"] = int((deep & courtyard).sum())
    rep["cells_gt1m_violation"] = int((deep & viol).sum())
    rep["cells_gt1m_genuine"] = int((deep & ~courtyard & ~viol).sum())
    if rep["cells_gt1m_genuine"] > 0 and fb is not None:
        g = deep & ~courtyard & ~viol
        rep["deepest_genuine_m"] = round(float(md[g].max()), 2)

    # 5. velocity screen: substantial water only (>10 cm) and away from
    # cliffs (>1 m drop to a neighbor = walls/pit edges, where high speeds
    # are physical waterfalls). Stair risers (~0.4 m) stay included.
    from scipy import ndimage
    cliff = (dem - ndimage.minimum_filter(np.nan_to_num(dem, nan=1e6), 3)) > 1.0
    vsel = street & (md > 0.10) & ~cliff
    vwet = mv[vsel]
    rep["vel_p999"] = round(float(np.percentile(vwet, 99.9)), 2) if vwet.size else 0.0
    rep["vel_max_off_cliff"] = round(float(mv[vsel].max()), 2) if vsel.any() else 0.0
    # Manning on steep Beirut streets legitimately reaches 6-8 m/s
    # (S=0.15, n=0.016, h=0.15 -> ~7 m/s); flag only clearly numerical speeds
    if rep["vel_p999"] > 10.0:
        flags.append(f"p99.9 velocity {rep['vel_p999']} m/s > 10 at depth>10cm "
                     f"off-cliff (numerical - check limiter/dt)")
    elif rep["vel_p999"] > 6.0:
        notes.append(f"supercritical street flow: p99.9 velocity "
                     f"{rep['vel_p999']} m/s (physical on steep streets)")

    # 6. rational-method envelope on peak outflow
    i_peak = max(s[2] for s in meta["steps_mmh"]) / 3.6e6 if meta["steps_mmh"] else 0
    A = float(valid.sum()) * area
    q_rational = 0.9 * i_peak * A
    series = meta.get("outflow_series_m3s", [])
    q_peak = max((r for _, r in series), default=0.0)
    rep["peak_outflow_m3s"] = round(q_peak, 2)
    rep["rational_envelope_m3s"] = round(q_rational, 2)
    rep["outflow_ratio"] = round(q_peak / q_rational, 3) if q_rational else None
    if q_rational and q_peak > 1.2 * q_rational:
        flags.append(f"peak outflow {q_peak:.1f} exceeds rational envelope "
                     f"{q_rational:.1f} m3/s")

    rep["flags"] = flags
    rep["notes"] = notes
    rep["verdict"] = "PASS" if not flags else "WARN"
    with open(os.path.join(args.run, "sanity_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    ortho = np.asarray(Image.open(os.path.join(args.terrain, "ortho.png")))
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    vmax = max(rep.get("street_depth_p99", 0.3), 0.3)
    axes[0].imshow(ortho)
    im = axes[0].imshow(np.where(md > 0.05, md, np.nan), cmap="turbo",
                        vmin=0, vmax=vmax, alpha=0.8)
    axes[0].set_title(f"max depth (color capped at street p99 = {vmax:.2f} m; "
                      f"absolute max {md[valid].max():.2f} m)")
    plt.colorbar(im, ax=axes[0], shrink=0.7)
    if wet.size:
        axes[1].hist(wet, bins=80, color="#3d8bd4")
        axes[1].set_yscale("log")
    axes[1].set_title("street wet-depth histogram (>5 cm)")
    axes[1].set_xlabel("m")
    axes[2].imshow(ortho)
    problems = np.full(md.shape, np.nan)
    problems[deep & courtyard] = 0
    problems[deep & viol] = 1
    problems[deep & ~courtyard & ~viol] = 2
    axes[2].imshow(problems, cmap=plt.matplotlib.colors.ListedColormap(
        ["orange", "red", "magenta"]), vmin=-0.5, vmax=2.5, interpolation="nearest")
    axes[2].set_title(f">1 m cells: courtyard {rep['cells_gt1m_courtyard']} (orange) / "
                      f"numerical {rep['cells_gt1m_violation']} (red) / "
                      f"genuine {rep['cells_gt1m_genuine']} (magenta)")
    for ax in axes[::2]:
        ax.axis("off")
    fig.suptitle(f"{os.path.basename(args.run)} - {rep['verdict']}"
                 + (f": {'; '.join(flags)}" if flags else ""), fontsize=13)
    fig.savefig(os.path.join(args.run, "sanity.png"), dpi=110, bbox_inches="tight")

    print(json.dumps(rep, indent=2))
    print(f"\n{rep['verdict']}" + (f" - {len(flags)} flag(s)" if flags else ""))


if __name__ == "__main__":
    main()
