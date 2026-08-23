#!/usr/bin/env python3
"""Figure set for the Al-Masar presentation: solver, benchmarks, Beirut result.

One script so the whole deck re-renders from one command when a run finishes -
the corridor numbers in particular are expected to change when the HLLC study
lands, and hand-assembled figures would silently go stale.

  python scripts/make_presentation_figs.py --runs output/corridor_runs_v3
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap, TwoSlopeNorm
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))

# One palette across the deck so slides read as a set. Teal = "after"/green
# infrastructure, warm grey-brown = "before", brick = hazard/loss.
C_BEFORE, C_AFTER, C_ACCENT, C_WARN = "#8C7B6B", "#0E7C86", "#3FB3BD", "#A03E3E"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

STORMS = [("t2", "T2 frequent"), ("v1_nov2025", "25 Nov 2025 observed"),
          ("t50", "T50 severe")]

# bake_corridor.PROPS depression depth per material class. Cells dug to detain
# water (bioswale, bioretention pond, terrace) hold water BY DESIGN, so counting
# them as "flooded" penalises the corridor for working. The published study made
# the same distinction by reporting walkable surfaces separately.
DEPRESSION = {1: 0.0, 2: 0.0, 3: 0.15, 4: 0.0, 5: 0.0, 6: 0.40, 7: 0.10}


def corridor_masks(material_path):
    """(whole ribbon, walkable surface only)."""
    mat = np.load(material_path)
    ribbon = mat > 0
    dug = np.zeros(mat.shape, bool)
    for cls, d in DEPRESSION.items():
        if d > 0:
            dug |= (mat == cls)
    return ribbon, ribbon & ~dug


# Corridor surfaces, coloured as a landscape plan rather than by the sandbox's
# UI swatches: vegetated surfaces green, paved warm-neutral, standing water
# blue. Keeps the deck's palette coherent and makes the cross-section read
# left-to-right the way the design drawing does.
MATERIALS = [
    (1, "vehicular lane",    "#9AA5A8"),
    (2, "porous bikelane",   "#C2A878"),
    (4, "porous sidewalk",   "#D9C7A7"),
    (3, "bioswale",          "#2E7D5B"),
    (5, "rain garden",       "#6FB08A"),
    (7, "terrace",           "#B08A46"),
    (6, "bioretention pond", "#2A7FA6"),
]


def load(runs, name):
    p = os.path.join(runs, name, "max_depth.npy")
    return np.load(p) if os.path.exists(p) else None


def meta(runs, name):
    p = os.path.join(runs, name, "run_meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def fig_beirut_maps(a, T, runs, out):
    """Before / after / difference, on the observed storm."""
    m = np.load(os.path.join(T, "masks.npz"))
    dem = np.load(os.path.join(T, "dem.npy"))
    b = load(runs, "before_v1_nov2025")
    aft = load(runs, "after_v1_nov2025")
    if b is None or aft is None:
        return
    ribbon = np.load(a.material) > 0
    valid = m["valid"]
    # crop to the corridor's own bounding box plus a margin - the full domain
    # is mostly untouched blocks and reads as empty on a slide
    rr, cc = np.where(ribbon)
    r0, r1 = max(rr.min() - 120, 0), min(rr.max() + 120, dem.shape[0])
    c0, c1 = max(cc.min() - 120, 0), min(cc.max() + 120, dem.shape[1])
    sl = (slice(r0, r1), slice(c0, c1))
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(np.where(valid, dem, np.nanmedian(dem[valid]))[sl],
                      vert_exag=2, dx=0.5, dy=0.5)

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 6.4))
    for k, (d, ttl) in enumerate(((b, "Before — today's streets"),
                                  (aft, "After — green corridor"))):
        ax[k].imshow(hs, cmap="gray", vmin=0, vmax=1.4)
        dd = np.ma.masked_where(~(d[sl] > 0.05), d[sl])
        im = ax[k].imshow(dd, cmap="turbo", vmin=0.05, vmax=0.6)
        ax[k].contour(ribbon[sl], levels=[0.5], colors=C_ACCENT, linewidths=0.7)
        ax[k].set_title(ttl)
        ax[k].set_xticks([]); ax[k].set_yticks([]); ax[k].grid(False)
    cb = fig.colorbar(im, ax=ax[:2], fraction=0.03, pad=0.01)
    cb.set_label("max water depth (m)")

    diff = (aft - b)[sl]
    dd = np.ma.masked_where(np.abs(diff) < 0.02, diff)
    ax[2].imshow(hs, cmap="gray", vmin=0, vmax=1.4)
    im2 = ax[2].imshow(dd, cmap="RdBu_r", norm=TwoSlopeNorm(0, -0.3, 0.3))
    ax[2].contour(ribbon[sl], levels=[0.5], colors="k", linewidths=0.7)
    ax[2].set_title("Change (blue = drier after)")
    ax[2].set_xticks([]); ax[2].set_yticks([]); ax[2].grid(False)
    fig.colorbar(im2, ax=ax[2], fraction=0.03, pad=0.01).set_label("Δ depth (m)")
    fig.suptitle("Al-Masar green corridor — 25 Nov 2025 observed storm, 0.5 m grid",
                 fontsize=12)
    fig.savefig(os.path.join(out, "beirut_before_after.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def fig_bands(a, T, runs, out):
    """Reduction by distance from the corridor - the '2 blocks' claim."""
    m = np.load(os.path.join(T, "masks.npz"))
    street = m["valid"] & ~m["building"] & ~m["water"]
    if "courtyard" in m:
        street = street & ~m["courtyard"]
    ribbon = np.load(a.material) > 0
    dist = ndimage.distance_transform_edt(~ribbon) * 0.5
    bands = [("on\ncorridor", ribbon), ("0–25 m", (dist > 0) & (dist <= 25) & street),
             ("25–50 m", (dist > 25) & (dist <= 50) & street),
             ("50–100 m", (dist > 50) & (dist <= 100) & street)]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    w = 0.26
    xs = np.arange(len(bands))
    for i, (key, lab) in enumerate(STORMS):
        b, aft = load(runs, f"before_{key}"), load(runs, f"after_{key}")
        if b is None or aft is None:
            continue
        vals = []
        for _, mask in bands:
            fb = ((b > 0.10) & mask).sum()
            fa = ((aft > 0.10) & mask).sum()
            vals.append(100 * (fb - fa) / fb if fb else np.nan)
        ax.bar(xs + (i - 1) * w, vals, w, label=lab,
               color=[C_BEFORE, C_ACCENT, C_AFTER][i])
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bands])
    ax.set_ylabel("reduction in flooded area (%)")
    ax.set_title("The benefit reaches about two blocks, then fades")
    ax.legend(frameon=False, fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "beirut_bands.png"), dpi=140)
    plt.close(fig)


def fig_water_balance(a, runs, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    labs = [l for _, l in STORMS]
    xs = np.arange(len(STORMS))
    ib = [meta(runs, f"before_{k}").get("vol_infiltrated_m3", np.nan) for k, _ in STORMS]
    ia = [meta(runs, f"after_{k}").get("vol_infiltrated_m3", np.nan) for k, _ in STORMS]
    ax[0].bar(xs - 0.19, ib, 0.38, label="before", color=C_BEFORE)
    ax[0].bar(xs + 0.19, ia, 0.38, label="after", color=C_AFTER)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(labs, fontsize=8)
    ax[0].set_ylabel("infiltrated volume (m³)")
    ax[0].set_title("The corridor absorbs what the street shed")
    ax[0].legend(frameon=False, fontsize=8)
    for i, (x, y) in enumerate(zip(xs, ia)):
        if np.isfinite(y):
            ax[0].annotate(f"+{y - ib[i]:,.0f} m³", (x + 0.19, y), ha="center",
                           va="bottom", fontsize=7.5, color=C_AFTER)
    ob = [meta(runs, f"before_{k}").get("vol_outflow_m3", np.nan) for k, _ in STORMS]
    oa = [meta(runs, f"after_{k}").get("vol_outflow_m3", np.nan) for k, _ in STORMS]
    ax[1].bar(xs - 0.19, ob, 0.38, label="before", color=C_BEFORE)
    ax[1].bar(xs + 0.19, oa, 0.38, label="after", color=C_AFTER)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(labs, fontsize=8)
    ax[1].set_ylabel("volume leaving toward the port (m³)")
    ax[1].set_title("…and sends less of it downstream")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "beirut_water_balance.png"), dpi=140)
    plt.close(fig)


def fig_validation(out):
    """Cross-engine agreement: the case for trusting the model."""
    import glob
    s = "output/crosscheck/synx_cut_v3/max_depth.npy"
    i = "output/crosscheck/ours_cut_v3/max_depth.npy"
    h = "output/crosscheck/ours_cut_v3_hllc/max_depth.npy"
    if not all(os.path.exists(p) for p in (s, i, h)):
        return
    S, I, H = np.load(s), np.load(i), np.load(h)
    m = np.load("output/terrain_cut_0.5_v3/masks.npz")
    v = m["valid"] & ~m["building"] & ~m["water"]
    v[:10] = v[-10:] = False; v[:, :10] = v[:, -10:] = False
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for k, (X, nm) in enumerate(((I, "inertial (3-term)"), (H, "HLLC (shock-capturing)"))):
        both = v & (X > 0.05) & (S > 0.05)
        x, y = S[both], X[both]
        ax[k].hexbin(x, y, gridsize=55, bins="log", cmap="Blues", mincnt=1)
        lim = np.percentile(np.concatenate([x, y]), 99.5)
        ax[k].plot([0, lim], [0, lim], color=C_WARN, lw=1, ls="--")
        iou = ((v & (X > 0.05) & (S > 0.05)).sum()
               / (v & ((X > 0.05) | (S > 0.05))).sum())
        rmse = np.sqrt(((x - y) ** 2).mean())
        ax[k].set_xlim(0, lim); ax[k].set_ylim(0, lim)
        ax[k].set_xlabel("SynxFlow depth (m)"); ax[k].set_ylabel("our depth (m)")
        ax[k].set_title(f"{nm}\nIoU {iou:.3f} · RMSE {rmse*100:.1f} cm")
        ax[k].grid(alpha=0.2)
    fig.suptitle("Agreement with an independent full shallow-water solver, "
                 "same 0.5 m corridor grid", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "validation_synxflow.png"), dpi=140)
    plt.close(fig)


def fig_benchmarks(out):
    """EA Test 4 cross-section against the published spread."""
    p = "output/ea4/ea4_result.json"
    ph = "output/ea4_hllc/ea4_result.json"
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    dh = json.load(open(ph)) if os.path.exists(ph) else None
    xs = [r["x_m"] for r in d["cross_section_1h"]]
    lo = [r["published"][0] for r in d["cross_section_1h"]]
    hi = [r["published"][1] for r in d["cross_section_1h"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.fill_between(xs, lo, hi, color="0.75", alpha=0.65,
                    label="published spread (19 packages)")
    ax.plot(xs, [r["ours_m"] for r in d["cross_section_1h"]], "o-",
            color=C_BEFORE, ms=4, lw=1.4, label="ours — inertial")
    if dh:
        ax.plot(xs, [r["ours_m"] for r in dh["cross_section_1h"]], "s-",
                color=C_AFTER, ms=4, lw=1.4, label="ours — HLLC")
    ax.set_xlabel("distance from the inflow (m)")
    ax.set_ylabel("water depth at t = 1 h (m)")
    ax.set_title("UK EA benchmark Test 4 — flood propagation over a floodplain")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "benchmark_ea4.png"), dpi=140)
    plt.close(fig)


def fig_benchmark_ea8(out, summ="output/ea8/ours_rerun/ea8_summary.json",
                      env="docs/ea8_published_envelope.json"):
    """EA Test 8A peak depths against the published cluster, gauge by gauge."""
    if not (os.path.exists(summ) and os.path.exists(env)):
        print("  skip fig_benchmark_ea8 (no data)")
        return
    E = json.load(open(env))["peak_depth_m"]
    S = json.load(open(summ))
    ours = {k: v["ours"] for k, v in S["peak_depth_m"].items()
            if k in E and "ours" in v}
    pts = [k for k in ("P1", "P2", "P3", "P6") if k in E and k in ours]
    if not pts:
        print("  skip fig_benchmark_ea8 (no matching gauges)")
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    x = np.arange(len(pts))
    # full spread as a pale bar, the "cluster" (excluding the report's own
    # approximate 0-term packages) as the solid one - the report draws the same
    # distinction, and being inside the cluster is the stronger claim
    for i, k in enumerate(pts):
        e = E[k]
        ax.plot([i, i], [e["all_min"], e["all_max"]], color="0.82", lw=9,
                solid_capstyle="butt",
                label="all 19 packages" if i == 0 else None)
        ax.plot([i, i], [e["cluster_min"], e["cluster_max"]], color="0.6", lw=9,
                solid_capstyle="butt",
                label="main cluster" if i == 0 else None)
    ax.plot(x, [ours[k] for k in pts], "o", color=C_AFTER, ms=9, zorder=5,
            label="ours")
    ax.set_xticks(x); ax.set_xticklabels(pts)
    ax.set_xlabel("output point"); ax.set_ylabel("peak water depth (m)")
    ax.set_title("UK EA benchmark Test 8A — rainfall flooding in an urban street network")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_xlim(-0.5, len(pts) - 0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "benchmark_ea8.png"), dpi=140)
    plt.close(fig)


def fig_storms(out):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, (key, lab) in enumerate(STORMS):
        p = f"storms/{key}.json"
        if not os.path.exists(p):
            continue
        st = json.load(open(p))
        t, q = [], []
        for t0, t1, mmh in st["steps"]:
            t += [t0 / 60, t1 / 60]; q += [mmh, mmh]
        ax.step(t, q, where="post", lw=1.6, label=f"{lab} ({st['total_mm']} mm)",
                color=[C_BEFORE, C_ACCENT, C_WARN][i])
    ax.set_xlabel("time (minutes)"); ax.set_ylabel("rainfall intensity (mm/h)")
    ax.set_title("Design storms")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "storms.png"), dpi=140)
    plt.close(fig)


def fig_materials(a, T, out):
    """What is actually being built, surface by surface."""
    mat = np.load(a.material)
    dem = np.load(os.path.join(T, "dem.npy"))
    m = np.load(os.path.join(T, "masks.npz"))
    valid = m["valid"]
    rr, cc = np.where(mat > 0)
    r0, r1 = max(rr.min() - 60, 0), min(rr.max() + 60, dem.shape[0])
    c0, c1 = max(cc.min() - 60, 0), min(cc.max() + 60, dem.shape[1])
    sl = (slice(r0, r1), slice(c0, c1))
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(np.where(valid, dem, np.nanmedian(dem[valid]))[sl],
                      vert_exag=2, dx=0.5, dy=0.5)
    sub = mat[sl]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 7.4),
                           gridspec_kw={"width_ratios": [1.35, 1]})
    ax[0].imshow(hs, cmap="gray", vmin=0, vmax=1.5)
    rgba = np.zeros(sub.shape + (4,))
    for cls, _, col in MATERIALS:
        r, g, b = [int(col[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        rgba[sub == cls] = (r, g, b, 0.95)
    ax[0].imshow(rgba)
    ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].grid(False)
    ax[0].set_title("The corridor as built, cell by cell")

    areas = [(lab, float((mat == cls).sum()) * 0.25, col)
             for cls, lab, col in MATERIALS]
    labs = [x[0] for x in areas][::-1]
    vals = [x[1] for x in areas][::-1]
    cols = [x[2] for x in areas][::-1]
    ax[1].barh(range(len(labs)), vals, color=cols, edgecolor="none")
    ax[1].set_yticks(range(len(labs))); ax[1].set_yticklabels(labs)
    ax[1].set_xlabel("area (m²)")
    ax[1].set_title(f"{sum(vals):,.0f} m² of permeable and paved surface")
    for i, v in enumerate(vals):
        ax[1].annotate(f"{v:,.0f}", (v, i), xytext=(4, 0),
                       textcoords="offset points", va="center", fontsize=8,
                       color=C_BEFORE)
    ax[1].grid(axis="y", alpha=0)
    fig.suptitle("Al-Masar Al-Akhdar — the designed cross-section, rasterised at 0.5 m",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "corridor_materials.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def fig_resolution(out, path="output/resolution/resolution_summary.json"):
    """Does a finer grid change the answer? - the EA's own open question."""
    if not os.path.exists(path):
        print("  skip fig_resolution (no data yet)")
        return
    d = json.load(open(path))["rows"]
    if len(d) < 2:
        print("  skip fig_resolution (need >=2 resolutions)")
        return
    d = sorted(d, key=lambda r: r["res_m"])
    res = [r["res_m"] for r in d]
    red = [r["reduction_pct"] for r in d]
    vel = [r["p99_velocity_ms"] for r in d]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    ax[0].plot(res, red, "o-", color=C_AFTER, lw=1.6, ms=5)
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(res)
    ax[0].set_xticklabels([f"{r:g}" for r in res])
    ax[0].set_xlabel("grid resolution (m)")
    ax[0].set_ylabel("corridor benefit (% reduction)")
    ax[0].set_title("Does the answer depend on the grid?")
    ax[1].plot(res, vel, "s-", color=C_WARN, lw=1.6, ms=5)
    ax[1].set_xscale("log", base=2); ax[1].set_xticks(res)
    ax[1].set_xticklabels([f"{r:g}" for r in res])
    ax[1].set_xlabel("grid resolution (m)")
    ax[1].set_ylabel("p99 street velocity (m/s)")
    ax[1].set_title("Velocity is the variable that suffers")
    for a_ in ax:
        a_.axvline(2.0, color=C_BEFORE, ls="--", lw=1)
        a_.annotate("EA benchmark\nresolution", (2.0, a_.get_ylim()[0]),
                    xytext=(4, 6), textcoords="offset points", fontsize=7.5,
                    color=C_BEFORE)
    fig.suptitle("Grid-resolution convergence on a real dense-city catchment",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "resolution.png"), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="output/terrain_cut_0.5_v3")
    ap.add_argument("--runs", default="output/corridor_runs_v3")
    ap.add_argument("--material", default="output/corridor_gi_cut_v3/material.npy")
    ap.add_argument("--out", default="output/presentation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T = a.terrain
    for fn, args in ((fig_materials, (a, T, a.out)),
                     (fig_resolution, (a.out,)),
                     (fig_beirut_maps, (a, T, a.runs, a.out)),
                     (fig_bands, (a, T, a.runs, a.out)),
                     (fig_water_balance, (a, a.runs, a.out)),
                     (fig_validation, (a.out,)),
                     (fig_benchmarks, (a.out,)),
                     (fig_benchmark_ea8, (a.out,)),
                     (fig_storms, (a.out,))):
        try:
            fn(*args)
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
