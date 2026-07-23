#!/usr/bin/env python3
"""Corridor-focused before/after analysis and figures.

Consumes output/corridor_runs/{before,after,afterdrains,drainsonly}_<storm>
plus terrain_cut_0.5 + the corridor material raster, and emits the metrics
and figures used in the green-corridor report. Everything is measured
relative to the corridor ribbon and distance bands around it.
"""
import json, os, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, "scripts")
from las_common import load_transform
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

TER = "output/terrain_cut_0.5"
RUN = "output/corridor_runs"
FIG = "output/report/figs"
os.makedirs(FIG, exist_ok=True)
t = load_transform(f"{TER}/dem_transform.json"); RES = t["res"]; A = RES*RES
masks = np.load(f"{TER}/masks.npz")
valid, building, water = masks["valid"], masks["building"], masks["water"]
court = masks["courtyard"] if "courtyard" in masks else np.zeros_like(valid)
street = valid & ~building & ~court & ~water
mat = np.load("output/corridor_gi_cut/material.npy")
ribbon = mat > 0
distband = ndimage.distance_transform_edt(~ribbon) * RES
ortho = np.asarray(Image.open(f"{TER}/ortho.png"))

def load(scn, storm):
    d = f"{RUN}/{scn}_{storm}"
    if not os.path.exists(f"{d}/max_depth.npy"): return None
    return {"max": np.load(f"{d}/max_depth.npy"),
            "final": np.load(f"{d}/final_depth.npy"),
            "meta": json.load(open(f"{d}/run_meta.json"))}

def flooded_area(depth, mask, thr=0.10):
    return float(((depth > thr) & mask).sum()) * A

BANDS = [("on corridor", ribbon),
         ("0-25 m", (distband > 0) & (distband <= 25) & street),
         ("25-50 m", (distband > 25) & (distband <= 50) & street),
         ("50-100 m", (distband > 50) & (distband <= 100) & street)]
STORMS = ["t2", "v1_nov2025", "t50"]
SLAB = {"t2": "T2 (2-yr, frequent)", "v1_nov2025": "25 Nov 2025 (observed)",
        "t50": "T50 (50-yr, severe)"}

def main():
    metrics = {}
    for st in STORMS:
        b, a = load("before", st), load("after", st)
        if not b or not a: 
            print(f"missing {st}"); continue
        row = {"storm": SLAB[st]}
        # flooded-area reduction by band
        for lab, m in BANDS:
            fb, fa = flooded_area(b["max"], m), flooded_area(a["max"], m)
            row[f"flood_{lab}_before_m2"] = round(fb)
            row[f"flood_{lab}_after_m2"] = round(fa)
            row[f"flood_{lab}_pct"] = round(100*(1-fa/fb), 1) if fb > 0 else 0.0
        # depth on corridor
        db, da = b["max"][ribbon & street], a["max"][ribbon & street]
        row["p99_before_cm"] = round(100*np.percentile(db[db>0.01], 99), 1) if (db>0.01).any() else 0
        row["p99_after_cm"] = round(100*np.percentile(da[da>0.01], 99), 1) if (da>0.01).any() else 0
        row["meandepth_before_cm"] = round(100*db[db>0.01].mean(),1) if (db>0.01).any() else 0
        row["meandepth_after_cm"] = round(100*da[da>0.01].mean(),1) if (da>0.01).any() else 0
        # water balance
        for who, r in [("before", b), ("after", a)]:
            mt = r["meta"]
            row[f"{who}_infil_pct"] = round(100*mt["vol_infiltrated_m3"]/max(mt["vol_rain_m3"],1),1)
            row[f"{who}_outflow_m3"] = round(mt["vol_outflow_m3"])
            row[f"{who}_stored_m3"] = round(mt["vol_stored_end_m3"])
        row["infil_gain_m3"] = round(a["meta"]["vol_infiltrated_m3"]-b["meta"]["vol_infiltrated_m3"])
        # peak outflow attenuation from storage series (proxy for discharge)
        metrics[st] = row
        print(f"{SLAB[st]}: on-corridor flood -{row['flood_on corridor_pct']}%  "
              f"p99 {row['p99_before_cm']}->{row['p99_after_cm']} cm  "
              f"infil {row['before_infil_pct']}->{row['after_infil_pct']}%")
    json.dump(metrics, open("output/corridor_runs/metrics.json","w"), indent=2)
    print("wrote output/corridor_runs/metrics.json")
    return metrics

if __name__ == "__main__":
    main()
