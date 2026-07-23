#!/usr/bin/env python3
"""Design-storm library for Beirut (PLAN.md Phase C).

IDF basis: "Stormwater Network Code in Lebanon" (2022), Table 2 — Beirut
IDF curves (docs/stormwater_code_lebanon.pdf; cross-checked against
Dargham & Andraos 2026). Hyetographs are alternating-block (peak-centered)
in 5-minute blocks — constant-rate rain understates pluvial peaks.

Also includes V1, a replay of the observed 25 Nov 2025 burst
(25.4 mm/30 min with a 22.2 mm/15 min core) that flooded Sassine/the Ring
with drains clogged — the primary validation storm.

Writes storms/<name>.json:
  {"name":..., "steps": [[t0_s, t1_s, mm_per_h], ...], "duration": sim_s}
plus storms/storms_preview.png.

Usage: python scripts/storms.py [--out storms] [--drain-time 3600]
"""

import argparse
import json
import os

import numpy as np

# mm/h; durations in minutes  (source: stormwater code Table 2; 25-yr interpolated)
IDF_DUR = np.array([10, 20, 30, 60, 90, 120])
IDF = {
    2:   [74.1, 53.3, 41.9, 25.9, 20.0, 17.8],
    5:   [98.7, 70.0, 55.0, 33.3, 25.4, 23.0],
    10:  [118.4, 84.9, 66.8, 41.7, 31.7, 27.4],
    25:  [141.9, 102.4, 81.0, 51.0, 38.2, 32.0],
    50:  [159.3, 115.4, 91.6, 57.9, 43.0, 35.3],
    100: [181.0, 131.2, 102.2, 64.5, 48.8, 40.3],
}


def idf_intensity(T, dur_min):
    """log-log interpolation of intensity (mm/h) at duration dur_min."""
    i = np.array(IDF[T], dtype=float)
    return float(np.exp(np.interp(np.log(dur_min), np.log(IDF_DUR), np.log(i))))


def alternating_block(T, total_min=60, block_min=5, scale=1.0):
    """Peak-centered alternating-block hyetograph. Returns list of
    (t0_s, t1_s, mm/h) blocks."""
    nb = total_min // block_min
    durs = np.arange(1, nb + 1) * block_min
    depths = np.array([idf_intensity(T, d) * d / 60.0 for d in durs])  # mm
    inc = np.diff(np.concatenate([[0.0], depths]))                     # mm/block
    order = np.zeros(nb, dtype=int)          # largest block in the middle
    mid = nb // 2
    order[mid] = 0
    lo, hi, k = mid - 1, mid + 1, 1
    while k < nb:
        if lo >= 0:
            order[lo] = k; k += 1; lo -= 1
        if k < nb and hi < nb:
            order[hi] = k; k += 1; hi += 1
    blocks = inc[order] * scale             # mm per block, arranged
    steps = []
    for j, mm in enumerate(blocks):
        steps.append([j * block_min * 60, (j + 1) * block_min * 60,
                      round(mm / (block_min / 60.0), 2)])
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="storms")
    ap.add_argument("--drain-time", type=float, default=3600.0,
                    help="extra sim time after rain stops, s")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    storms = {}
    for T in (2, 10, 50):
        storms[f"t{T}"] = {
            "name": f"t{T}", "return_period_yr": T,
            "steps": alternating_block(T),
            "total_mm": round(idf_intensity(T, 60), 1),
        }
    s = alternating_block(10, scale=1.15)
    storms["t10cc"] = {"name": "t10cc", "return_period_yr": 10,
                       "note": "10-yr +15% climate uplift (2050, Zittis et al. 2021)",
                       "steps": s,
                       "total_mm": round(idf_intensity(10, 60) * 1.15, 1)}
    storms["v1_nov2025"] = {
        "name": "v1_nov2025",
        "note": "observed 25 Nov 2025: 25.4 mm/30 min, 22.2 mm/15 min core; "
                "flooded Sassine Sq + Ring with drains clogged",
        "steps": [[0, 450, 12.8], [450, 1350, 88.8], [1350, 1800, 12.8]],
        "total_mm": 25.4,
    }
    storms["flat30"] = {"name": "flat30",
                        "note": "legacy demo storm (constant 30 mm/h, 1 h)",
                        "steps": [[0, 3600, 30.0]], "total_mm": 30.0}

    for name, st in storms.items():
        st["duration"] = st["steps"][-1][1] + args.drain_time
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(st, f, indent=2)
        peak = max(s[2] for s in st["steps"])
        print(f"{name:12s} total {st['total_mm']:5.1f} mm, "
              f"peak {peak:6.1f} mm/h, sim {st['duration']:.0f}s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(storms), figsize=(4 * len(storms), 3),
                             sharey=True)
    for ax, (name, st) in zip(np.atleast_1d(axes), storms.items()):
        for t0, t1, i in st["steps"]:
            ax.bar((t0 + t1) / 120, i, width=(t1 - t0) / 60, color="#3d8bd4")
        ax.set_title(f"{name} ({st['total_mm']} mm)")
        ax.set_xlabel("min")
    np.atleast_1d(axes)[0].set_ylabel("mm/h")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "storms_preview.png"), dpi=130)
    print(f"wrote {len(storms)} storms + preview in {args.out}/")


if __name__ == "__main__":
    main()
