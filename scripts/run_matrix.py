#!/usr/bin/env python3
"""Run the scenario x storm matrix and summarize (PLAN.md Phases C/F).

Scenarios:
  S0_baseline      drains clogged (none), city as-is
  S1_drains        working inlets (terrain/drains.npz)
  S2_masar         Al-Masar green corridor (scenarios/masar_corridor.json)
  S3_masar_drains  both

Each run: solver -> sanity report -> max-depth + hazard maps; then
difference maps vs S0 and a summary table (markdown + csv).

Usage:
  python scripts/run_matrix.py --terrain output/terrain_1.0 \
      --storms v1_nov2025 t10 [--scenarios S0_baseline S2_masar] [--animate]
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from matplotlib.path import Path as MplPath

sys.path.insert(0, os.path.dirname(__file__))
from las_common import load_transform, pixel_to_utm
from flood_gpu import simulate, load_terrain

SCENARIOS = {
    "S0_baseline": {"drains": False, "edits": None},
    "S1_drains": {"drains": True, "edits": None},
    "S2_masar": {"drains": False, "edits": "MASAR"},
    "S3_masar_drains": {"drains": True, "edits": "MASAR"},
}


def polygon_mask(t, shape, poly):
    h, w = shape
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    ux, uy = pixel_to_utm(t, cols.ravel(), rows.ravel())
    return MplPath(np.asarray(poly)).contains_points(
        np.column_stack([ux, uy])).reshape(h, w)


def apply_edits(edits_file, t, dem, manning, infil):
    dem, manning, infil = dem.copy(), manning.copy(), infil.copy()
    with open(edits_file) as f:
        scn = json.load(f)
    for e in scn.get("edits", []):
        m = polygon_mask(t, dem.shape, e["polygon"])
        op = e["op"]
        if op in ("raise", "wall"):
            dem[m] += e["meters"]
        elif op == "lower":
            dem[m] -= e["meters"]
        elif op == "infiltrate":
            infil[m] = np.maximum(infil[m], e["mmh"])
        elif op == "manning":
            manning[m] = e["n"]
        else:
            sys.exit(f"unknown op {op}")
        print(f"    edit {op}: {m.sum()} cells")
    return dem, manning, infil


def sh(cmd):
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def write_summary(out_dir, rows):
    import csv
    keys = list(rows[0].keys())
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=keys)
        wcsv.writeheader()
        wcsv.writerows(rows)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("| " + " | ".join(keys) + " |\n")
        f.write("|" + "---|" * len(keys) + "\n")
        for r in rows:
            f.write("| " + " | ".join(
                f"{r[k]:.0f}" if isinstance(r[k], float) and k != "p99_depth_m"
                else str(r[k]) for k in keys) + " |\n")


def row_from_disk(run_dir, scen, storm):
    with open(os.path.join(run_dir, "run_meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(run_dir, "sanity_report.json")) as f:
        srep = json.load(f)
    return {
        "scenario": scen, "storm": storm,
        "verdict": srep["verdict"],
        "p99_depth_m": srep.get("street_depth_p99", 0),
        "wet_gt10cm_m2": srep.get("wet_area_m2_gt10cm", 0),
        "wet_gt30cm_m2": srep.get("wet_area_m2_gt30cm", 0),
        "infil_m3": meta["vol_infiltrated_m3"],
        "drained_m3": meta["vol_drained_m3"],
        "outflow_m3": meta["vol_outflow_m3"],
        "stored_m3": meta["vol_stored_end_m3"],
        "wall_s": meta["wall_s"],
    }


def collect(out_dir):
    """Rebuild summary.md/csv from every completed run on disk."""
    rows = []
    for d in sorted(os.listdir(out_dir)):
        run_dir = os.path.join(out_dir, d)
        if "__" not in d or not os.path.isdir(run_dir):
            continue
        if not (os.path.exists(os.path.join(run_dir, "run_meta.json")) and
                os.path.exists(os.path.join(run_dir, "sanity_report.json"))):
            print(f"  skipping incomplete run {d}")
            continue
        scen, storm = d.split("__", 1)
        rows.append(row_from_disk(run_dir, scen, storm))
    if not rows:
        sys.exit("no completed runs found")
    write_summary(out_dir, rows)
    print(f"summary rebuilt from {len(rows)} runs -> {out_dir}/summary.md")
    for r in rows:
        print(f"  {r['scenario']:16s} x {r['storm']:11s} [{r['verdict']}] "
              f"p99 {r['p99_depth_m']:.2f} m, >10cm {r['wet_gt10cm_m2']:9.0f} m2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true",
                    help="rebuild summary from completed runs on disk, then exit")
    ap.add_argument("--terrain")
    ap.add_argument("--storms", nargs="+")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--out", default="output/runs")
    ap.add_argument("--masar-edits", default="scenarios/masar_corridor.json")
    ap.add_argument("--save-every", type=float, default=60.0)
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.collect:
        collect(args.out)
        return
    if not args.terrain or not args.storms:
        sys.exit("--terrain and --storms are required (or use --collect)")

    dem0, t, masks, manning0, infil0, rain_w, gauges = load_terrain(args.terrain)
    drains_path = os.path.join(args.terrain, "drains.npz")
    py = sys.executable
    here = os.path.dirname(os.path.abspath(__file__))
    rows = []

    for storm_name in args.storms:
        with open(os.path.join("storms", f"{storm_name}.json")) as f:
            storm = json.load(f)
        for scen in args.scenarios:
            cfg = SCENARIOS[scen]
            run_dir = os.path.join(args.out, f"{scen}__{storm_name}")
            print(f"\n=== {scen} x {storm_name} -> {run_dir} ===")
            dem, manning, infil = dem0, manning0, infil0
            if cfg["edits"]:
                edits = args.masar_edits if cfg["edits"] == "MASAR" else cfg["edits"]
                dem, manning, infil = apply_edits(edits, t, dem0,
                                                  manning0, infil0)
            drains = None
            if cfg["drains"]:
                if not os.path.exists(drains_path):
                    sys.exit(f"{drains_path} missing - run scripts/drains.py first")
                dz = np.load(drains_path)
                drains = (dz["rows"], dz["cols"], dz["cap"])
            meta = simulate(dem, t["res"], storm["steps"], storm["duration"],
                            run_dir, manning=manning, infil_mmh=infil,
                            valid=masks["valid"], water=masks["water"],
                            rain_weight=rain_w, drains=drains, gauges=gauges,
                            save_every=args.save_every, device=args.device)
            sh([py, os.path.join(here, "sanity.py"), "--run", run_dir,
                "--terrain", args.terrain])
            sh([py, os.path.join(here, "render2d.py"), "maxdepth",
                "--run", run_dir, "--terrain", args.terrain,
                "--out", os.path.join(run_dir, "maxdepth.png")])
            sh([py, os.path.join(here, "render2d.py"), "hazard",
                "--run", run_dir, "--terrain", args.terrain,
                "--out", os.path.join(run_dir, "hazard.png")])
            if args.animate:
                sh([py, os.path.join(here, "render2d.py"), "animate",
                    "--run", run_dir, "--terrain", args.terrain,
                    "--out", os.path.join(run_dir, "flood.mp4")])
            with open(os.path.join(run_dir, "sanity_report.json")) as f:
                srep = json.load(f)
            rows.append({
                "scenario": scen, "storm": storm_name,
                "verdict": srep["verdict"],
                "p99_depth_m": srep.get("street_depth_p99", 0),
                "wet_gt10cm_m2": srep.get("wet_area_m2_gt10cm", 0),
                "wet_gt30cm_m2": srep.get("wet_area_m2_gt30cm", 0),
                "infil_m3": meta["vol_infiltrated_m3"],
                "drained_m3": meta["vol_drained_m3"],
                "outflow_m3": meta["vol_outflow_m3"],
                "stored_m3": meta["vol_stored_end_m3"],
                "wall_s": meta["wall_s"],
            })

        # difference maps vs S0 for this storm
        base = os.path.join(args.out, f"S0_baseline__{storm_name}")
        if os.path.isdir(base):
            for scen in args.scenarios:
                if scen == "S0_baseline":
                    continue
                rd = os.path.join(args.out, f"{scen}__{storm_name}")
                if os.path.isdir(rd):
                    sh([py, os.path.join(here, "render2d.py"), "diff",
                        "--run-a", base, "--run-b", rd,
                        "--terrain", args.terrain,
                        "--out", os.path.join(rd, "diff_vs_baseline.png")])

    # summary over everything completed on disk (not just this invocation)
    os.makedirs(args.out, exist_ok=True)
    collect(args.out)
    print(f"\nsummary -> {args.out}/summary.md")
    for r in rows:
        print(f"  {r['scenario']:16s} x {r['storm']:11s} [{r['verdict']}] "
              f"p99 {r['p99_depth_m']:.2f} m, >10cm {r['wet_gt10cm_m2']:8.0f} m2, "
              f"wall {r['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
