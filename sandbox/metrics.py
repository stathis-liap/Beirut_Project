"""Per-run metrics, generalizing scripts/analyze_corridor.py's band/flooded-
area logic to whatever material footprint (ribbon) a given run actually
used, cached to <run>/metrics.json on first request.
"""
import json
import os

import numpy as np
from scipy import ndimage


def flooded_area(depth, mask, cell_area, thr=0.10):
    return float(((depth > thr) & mask).sum()) * cell_area


def compute_metrics(base, run_material, run_dir):
    res = base.t["res"]
    area = res * res
    valid, building, water = base.masks["valid"], base.masks["building"], base.masks["water"]
    courtyard = base.masks["courtyard"]
    street = valid & ~building & ~courtyard & ~water
    ribbon = run_material > 0
    dist = ndimage.distance_transform_edt(~ribbon) * res

    max_depth = np.load(os.path.join(run_dir, "max_depth.npy"))
    with open(os.path.join(run_dir, "run_meta.json")) as f:
        meta = json.load(f)

    bands = [
        ("on_ribbon", ribbon),
        ("0_25m", (dist > 0) & (dist <= 25) & street),
        ("25_50m", (dist > 25) & (dist <= 50) & street),
        ("50_100m", (dist > 50) & (dist <= 100) & street),
    ]
    m = {"flooded_streets_m2": round(flooded_area(max_depth, street, area))}
    for label, mask in bands:
        m[f"flooded_{label}_m2"] = round(flooded_area(max_depth, mask, area))

    wet = max_depth[ribbon & street]
    wet = wet[wet > 0.01]
    m["p99_depth_ribbon_cm"] = round(100 * float(np.percentile(wet, 99)), 1) if wet.size else 0.0
    m["mean_depth_ribbon_cm"] = round(100 * float(wet.mean()), 1) if wet.size else 0.0

    m["vol_rain_m3"] = meta["vol_rain_m3"]
    m["vol_infiltrated_m3"] = meta["vol_infiltrated_m3"]
    m["infil_pct"] = round(100 * meta["vol_infiltrated_m3"] / max(meta["vol_rain_m3"], 1e-9), 1)
    m["vol_outflow_m3"] = meta["vol_outflow_m3"]
    m["vol_stored_end_m3"] = meta["vol_stored_end_m3"]
    m["closure_rel"] = meta["closure_rel"]

    street_wet = max_depth[street & (max_depth > 0.05)]
    m["default_vmax_m"] = round(float(np.clip(np.percentile(street_wet, 99), 0.3, 5.0)), 3) if street_wet.size else 0.3
    return m


def get_or_compute_metrics(base, run_material, run_dir):
    cache_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    m = compute_metrics(base, run_material, run_dir)
    with open(cache_path, "w") as f:
        json.dump(m, f, indent=2)
    return m
