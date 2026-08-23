#!/usr/bin/env python3
"""Score the attribution study: is the corridor's benefit robust to modelling choices?

Reports, for each variant, the flooded-street-area REDUCTION (before -> after)
rather than either absolute. That is the quantity the study actually claims,
and the quantity that can survive a scheme bias which affects both scenarios
equally.

The headline output is the spread of reductions across variants. A tight
spread means the claim is robust and the modelling choices are second-order.
A wide one means the published number carries an error bar it does not
currently admit to.

  python scripts/analyze_sensitivity.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def flooded_m2(run_dir, street, res, thresh=0.05):
    p = os.path.join(run_dir, "max_depth.npy")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return float(((d > thresh) & street).sum()) * res * res


def water_balance(run_dir):
    p = os.path.join(run_dir, "run_meta.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="output/terrain_cut_0.5_v3")
    ap.add_argument("--base-runs", default="output/corridor_runs_v3")
    ap.add_argument("--sens", default="output/sensitivity")
    ap.add_argument("--out", default="output/sensitivity/summary.json")
    args = ap.parse_args()

    m = np.load(os.path.join(args.terrain, "masks.npz"))
    res = json.load(open(os.path.join(args.terrain, "dem_transform.json")))["res"]
    # street = walkable open ground; courtyards are excluded from the published
    # statistic and must stay excluded here or the variants are not comparable
    street = m["valid"] & ~m["building"] & ~m["water"]
    if "courtyard" in m:
        street = street & ~m["courtyard"]

    B, S = args.base_runs, args.sens
    variants = [
        ("base",    "inertial / constant infil / capacity gullies", "v1_nov2025",
         f"{B}/before_v1_nov2025", f"{B}/after_v1_nov2025"),
        ("hllc",    "+ shock-capturing scheme",                      "v1_nov2025",
         f"{S}/hllc_before", f"{S}/hllc_after"),
        ("green-ampt", "+ Green-Ampt infiltration",                  "v1_nov2025",
         f"{S}/ga_before_v1_nov2025", f"{S}/ga_after_v1_nov2025"),
        ("base",    "inertial / constant infil / capacity gullies", "t50",
         f"{B}/before_t50", f"{B}/after_t50"),
        ("green-ampt", "+ Green-Ampt infiltration",                  "t50",
         f"{S}/ga_before_t50", f"{S}/ga_after_t50"),
    ]

    print("REDUCTION IN FLOODED STREET AREA (before -> after), by modelling variant")
    print(f"{'storm':<11} {'variant':<12} {'before':>10} {'after':>10} {'reduction':>11}")
    rows, by_storm = [], {}
    for name, desc, storm, bdir, adir in variants:
        fb, fa = flooded_m2(bdir, street, res), flooded_m2(adir, street, res)
        if fb is None or fa is None:
            print(f"{storm:<11} {name:<12} {'(missing)':>10}")
            continue
        red = 100.0 * (fb - fa) / fb if fb > 0 else float("nan")
        print(f"{storm:<11} {name:<12} {fb:10.0f} {fa:10.0f} {red:10.1f}%")
        rows.append({"storm": storm, "variant": name, "desc": desc,
                     "before_m2": fb, "after_m2": fa, "reduction_pct": red})
        by_storm.setdefault(storm, []).append((name, red))

    print("\nSPREAD OF THE CLAIM ACROSS MODELLING CHOICES")
    for storm, vs in by_storm.items():
        if len(vs) < 2:
            continue
        reds = [r for _, r in vs]
        base = dict(vs).get("base")
        print(f"  {storm:<11} reductions {min(reds):.1f}% - {max(reds):.1f}% "
              f"(spread {max(reds) - min(reds):.1f} points"
              + (f", base {base:.1f}%)" if base is not None else ")"))

    # --- erosion ensemble ---------------------------------------------------
    ero = [f"{S}/ero_after_t50_s{s}" for s in (1, 2, 3)]
    vals = [flooded_m2(d, street, res) for d in ero]
    vals = [v for v in vals if v is not None]
    if vals:
        base_after = flooded_m2(f"{B}/after_t50", street, res)
        print(f"\nEROSION ENSEMBLE (after-corridor, T50, {len(vals)} seeds)")
        print(f"  flooded area {min(vals):.0f} - {max(vals):.0f} m2 "
              f"(mean {np.mean(vals):.0f}, spread {100*(max(vals)-min(vals))/np.mean(vals):.1f}%)")
        if base_after:
            print(f"  vs no-erosion {base_after:.0f} m2 -> "
                  f"{100*(np.mean(vals)-base_after)/base_after:+.1f}% mean effect")
        print("  NOTE: erosion parameters are illustrative, NOT calibrated - this is a "
              "sensitivity, not a prediction")

    # --- drainage variant ---------------------------------------------------
    hd, cap = f"{S}/hd_afterdrains", f"{S}/cap_afterdrains"
    fh, fc = flooded_m2(hd, street, res), flooded_m2(cap, street, res)
    if fh is not None and fc is not None:
        wh, wc = water_balance(hd), water_balance(cap)
        print(f"\nGULLY MODEL (after + optimised inlets, observed storm)")
        print(f"  fixed capacity   flooded {fc:8.0f} m2   drained "
              f"{wc.get('vol_drained_m3', float('nan')):8.0f} m3")
        print(f"  head-discharge   flooded {fh:8.0f} m2   drained "
              f"{wh.get('vol_drained_m3', float('nan')):8.0f} m3")
        print(f"  -> head-discharge captures "
              f"{100*(wh.get('vol_drained_m3',0)-wc.get('vol_drained_m3',0))/max(wc.get('vol_drained_m3',1),1):+.1f}% "
              f"of the fixed-capacity volume")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"rows": rows,
               "erosion_ensemble_m2": vals,
               "note": "reductions are the claim; absolutes are scheme-sensitive"},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
