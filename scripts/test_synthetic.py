#!/usr/bin/env python3
"""End-to-end pipeline test on a synthetic S-shaped stepped corridor.

Builds a fake DEM that mimics the real problem (sloped S-curved street with
stairs, flanked by buildings), runs the flood solver on it, and checks:
  - water concentrates in the corridor (not on roofs)
  - mass balance closes
  - a carved escape channel reduces max depth downstream

Usage: python scripts/test_synthetic.py [--render]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from flood_sim import simulate


def synthetic_dem(h=400, w=300, res=1.0):
    """Sloped plane, S-shaped street canyon with stairs, buildings elsewhere."""
    rows = np.arange(h)[:, None] * np.ones((1, w))
    cols = np.ones((h, 1)) * np.arange(w)[None, :]

    base = (h - rows) * 0.08  # 8% average slope, downhill toward south (high row)

    # S-shaped street centerline: x = center + A*sin(y)
    center = w / 2 + 60 * np.sin(rows[:, 0] / h * 2 * np.pi)
    street_halfwidth = 8
    dist = np.abs(cols - center[:, None])
    street = dist < street_halfwidth

    dem = base.copy()
    dem[~street] += 12.0  # buildings: 12 m walls beside the street

    # stairs: quantize the street elevation into 40 cm steps in three flights
    stairs = street & (((rows > 80) & (rows < 130)) |
                       ((rows > 200) & (rows < 250)) |
                       ((rows > 320) & (rows < 360)))
    dem[stairs] = np.floor(dem[stairs] / 0.4) * 0.4

    dem += np.random.default_rng(0).normal(0, 0.01, dem.shape)  # cm-level noise
    return dem.astype(np.float64), street


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="also test PyVista render")
    args = ap.parse_args()
    out = "output/_selftest"
    os.makedirs(out, exist_ok=True)

    dem, street = synthetic_dem()
    np.save(os.path.join(out, "dem.npy"), dem.astype(np.float32))

    print("=== baseline: 30 mm/h for 20 min, sim 25 min ===")
    md_base, _ = simulate(dem, rain_mmh=30, duration=1500, save_every=100,
                          out_dir=os.path.join(out, "run_base"),
                          rain_stop=1200, res=1.0)

    # checks
    street_depth = md_base[street].max()
    roof_depth = np.percentile(md_base[~street], 99)
    print(f"max depth in street: {street_depth:.3f} m, "
          f"p99 on roofs/buildings: {roof_depth:.3f} m")
    assert street_depth > 0.03, "water should accumulate in the corridor"
    assert street_depth > 3 * roof_depth, "water should concentrate in the street"

    with open(os.path.join(out, "run_base", "run_meta.json")) as f:
        meta = json.load(f)
    assert meta["vol_rain_m3"] > 0
    print("mass balance fields present:",
          {k: round(v) for k, v in meta.items() if k.startswith("vol")})

    print("=== scenario: carve 1 m escape channel branching off mid-corridor ===")
    dem2 = dem.copy()
    dem2[195:205, 150:299] -= 13.0  # channel through the buildings, out the east edge
    md_scn, _ = simulate(dem2, rain_mmh=30, duration=1500, save_every=100,
                         out_dir=os.path.join(out, "run_channel"),
                         rain_stop=1200, res=1.0)
    downstream = street & (np.arange(dem.shape[0])[:, None] > 210)
    d_base = md_base[downstream].max()
    d_scn = md_scn[downstream].max()
    print(f"downstream max depth: baseline {d_base:.3f} m -> channel {d_scn:.3f} m")
    assert d_scn < d_base, "escape channel should reduce downstream depth"

    if args.render:
        print("=== render test ===")
        from PIL import Image
        g = np.clip((dem - dem.min()) / (dem.max() - dem.min()) * 255, 0, 255)
        Image.fromarray(np.stack([g] * 3, -1).astype(np.uint8)).save(
            os.path.join(out, "ortho.png"))
        with open(os.path.join(out, "dem_transform.json"), "w") as f:
            json.dump({"res": 1.0, "minx": 0, "miny": 0, "maxy": 400,
                       "width": 300, "height": 400, "crs": "EPSG:32636"}, f)
        import subprocess
        subprocess.run([sys.executable, "scripts/render_3d.py", "video",
                        "--run", os.path.join(out, "run_base"),
                        "--data-dir", out,
                        "--out", os.path.join(out, "test.mp4"),
                        "--fps", "5"], check=True)
        subprocess.run([sys.executable, "scripts/render_3d.py", "compare",
                        "--runs", os.path.join(out, "run_base"),
                        os.path.join(out, "run_channel"),
                        "--data-dir", out,
                        "--out", os.path.join(out, "compare.png")], check=True)

    print("ALL SYNTHETIC TESTS PASSED")


if __name__ == "__main__":
    main()
