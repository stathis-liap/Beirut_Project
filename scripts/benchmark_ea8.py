#!/usr/bin/env python3
"""EA 2D benchmark Test 8A (Glasgow) - PLAN.md D2.

Neelz & Pender (2013) Test 8A is the standard rainfall-on-urban-area case:
a 0.4 km2 patch of Glasgow, 400 mm/h of rain for 3 minutes over the whole
domain plus a 2.5 m3/s point inflow hydrograph, run for 5 hours, with
depth reported at 9 fixed points. Unlike the Beirut cross-checks this is
an *external* case with published answers, so it tests correctness rather
than agreement.

Inputs: the LISFLOOD-FP-format setup from Zenodo record 6907286
(`4-Glasgow/Setup/ea8-2m.*`), which is the EA distribution repackaged.

  run    run our torch solver on it, write stage-point depth series
  plot   overlay our series, LISFLOOD-FP's, and SynxFlow's if present

Usage:
  python scripts/benchmark_ea8.py run  --setup ~/Work/engines/ea_benchmark/4-Glasgow/Setup \
      --res 2m --out output/ea8
  python scripts/benchmark_ea8.py plot --out output/ea8
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

NODATA = -9999.0
POINT_NAMES = [f"P{i}" for i in range(1, 10)]


def read_asc(path):
    hdr = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            hdr[k.lower()] = float(v)
        arr = np.loadtxt(f, dtype=np.float64)
    arr[arr == hdr.get("nodata_value", NODATA)] = np.nan
    return arr, hdr


def read_series(path, value_first=True):
    """LISFLOOD .rain/.bdy: a count+unit line, then value/time pairs.

    Times are given in the unit named on that line (these files use
    minutes); returns [[t_seconds, value], ...].
    """
    rows, unit = [], "seconds"
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2 and not _isnum(parts[0]):
                continue                              # series name line
            if len(parts) == 2 and _isnum(parts[0]) and not _isnum(parts[1]):
                unit = parts[1].lower()               # "23   minutes"
                continue
            if len(parts) == 2:
                a, b = float(parts[0]), float(parts[1])
                rows.append([b, a] if value_first else [a, b])
    mul = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}[unit]
    out = np.array(rows, dtype=float)
    out[:, 0] *= mul
    return out[np.argsort(out[:, 0], kind="stable")]


def _isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def xy_to_rc(hdr, x, y):
    col = int((x - hdr["xllcorner"]) / hdr["cellsize"])
    row = int(hdr["nrows"] - 1 - (y - hdr["yllcorner"]) / hdr["cellsize"])
    return row, col


def read_stage(path):
    pts = []
    with open(path) as f:
        n = int(f.readline().split()[0])
        for _ in range(n):
            x, y = f.readline().split()[:2]
            pts.append((float(x), float(y)))
    return pts


def rain_steps(series):
    """[[t, mm/h], ...] -> flood_gpu steps [(t0, t1, mm/h), ...]."""
    steps = []
    for i in range(len(series) - 1):
        t0, v = series[i]
        t1 = series[i + 1][0]
        if v > 0 and t1 - t0 > 0.01:      # skip the ramp-off duplicates
            steps.append((float(t0), float(t1), float(v)))
    return steps


def cmd_run(args):
    from flood_gpu import simulate

    tag = f"ea8-{args.res}"
    dem, hdr = read_asc(os.path.join(args.setup, f"{tag}.dem"))
    n, _ = read_asc(os.path.join(args.setup, f"{tag}.n"))
    rain = read_series(os.path.join(args.setup, f"{tag}.rain"))
    inflow = read_series(os.path.join(args.setup, f"{tag}.bdy"))
    stage = read_stage(os.path.join(args.setup, f"{tag}.stage"))
    res = hdr["cellsize"]

    # LISFLOOD .bdy values are per unit width: for a point source it applies
    # `qtmp * dx` (iterateq.cpp, QVAR5 branch). Both EA resolutions confirm
    # it - 2.5 at 2 m and 10.0 at 0.5 m both mean the 5 m3/s peak of the
    # spec's Figure (c). Forgetting this halves our inflow at 2 m.
    inflow[:, 1] *= res

    # .bci: "P <x> <y> QVAR <name>" - a point discharge source
    src = []
    with open(os.path.join(args.setup, f"{tag}.bci")) as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and p[0] == "P" and p[3].upper() == "QVAR":
                r, c = xy_to_rc(hdr, float(p[1]), float(p[2]))
                src.append({"row": r, "col": c, "series": inflow.tolist()})

    gauges = []
    for name, (x, y) in zip(POINT_NAMES, stage):
        r, c = xy_to_rc(hdr, x, y)
        gauges.append({"name": name, "row": r, "col": c})

    valid = np.isfinite(dem)
    print(f"Test 8A @ {res:.1f} m: {dem.shape[1]}x{dem.shape[0]} "
          f"({valid.sum()} valid cells), {args.sim_time:.0f} s")
    print(f"  rain: {rain[:, 1].max():.0f} mm/h peak, "
          f"inflow: {inflow[:, 1].max():.2f} m3/s peak at "
          f"{[(s['row'], s['col']) for s in src]}")

    # The EA setup has closed outer borders (only the point source enters),
    # while simulate() always drains its edge ring - so wall the domain in
    # and crop back, exactly as crosscheck.py --closed does.
    pad = 1
    demp = np.pad(dem, pad, constant_values=np.nan)
    zmax = np.nanmax(dem) + 100.0
    demp[~np.isfinite(demp)] = zmax
    np_ = np.pad(np.nan_to_num(n, nan=0.03), pad, constant_values=0.03)
    validp = np.pad(valid, pad, constant_values=False)
    rw = validp.astype(np.float32)

    meta = simulate(
        demp, res, rain_steps(rain), args.sim_time,
        os.path.join(args.out, "ours"),
        manning=np_, valid=np.ones_like(validp, dtype=bool),
        rain_weight=rw,
        gauges=[{"name": g["name"], "row": g["row"] + pad,
                 "col": g["col"] + pad} for g in gauges],
        sources=[{"row": s["row"] + pad, "col": s["col"] + pad,
                  "series": s["series"]} for s in src],
        save_every=args.save_every, save_frames=False,
        device=args.device,
    )
    np.save(os.path.join(args.out, "ours", "max_depth_cropped.npy"),
            np.load(os.path.join(args.out, "ours", "max_depth.npy"))
            [pad:-pad, pad:-pad])
    with open(os.path.join(args.out, "setup.json"), "w") as f:
        json.dump({"tag": tag, "header": hdr, "gauges": gauges,
                   "sim_time": args.sim_time,
                   "closure_rel": meta["closure_rel"]}, f, indent=2)
    print(f"stage series -> {args.out}/ours/gauges.csv")


def cmd_prep_lisflood(args):
    """Stage a CPU-runnable copy of the EA setup for our LISFLOOD-FP build.

    The shipped .par asks for `acc_nugrid` + `cuda`; our build is the CPU
    one, so the solver line becomes plain `acceleration` with the de
    Almeida damping that this kind of terrain needs (see
    docs/crosscheck_lisflood.md).
    """
    import shutil
    tag = f"ea8-{args.res}"
    dst = os.path.join(args.out, "lisflood")
    os.makedirs(dst, exist_ok=True)
    for ext in ("dem", "n", "rain", "bci", "bdy", "stage"):
        shutil.copyfile(os.path.join(args.setup, f"{tag}.{ext}"),
                        os.path.join(dst, f"{tag}.{ext}"))
    par = f"""# EA Test 8A - CPU ACC build (generated by benchmark_ea8.py)
DEMfile         {tag}.dem
bcifile         {tag}.bci
bdyfile         {tag}.bdy
stagefile       {tag}.stage
manningfile     {tag}.n
rainfall        {tag}.rain
resroot         {tag}
dirroot         results
sim_time        {args.sim_time:.1f}
initial_tstep   0.35
saveint         3600.0
massint         60.0
acceleration
theta           0.8
cfl             0.5
nodata_elevation 40.0
elevoff
"""
    with open(os.path.join(dst, f"{tag}.par"), "w") as f:
        f.write(par)
    print(f"wrote {dst}/{tag}.par")
    print(f"run:  ( cd {dst} && <lisflood> -v {tag}.par )")


def read_lisflood_stage(path, n_points=len(POINT_NAMES)):
    """LISFLOOD .stage output: a `stage,x,y,elev` preamble (4 columns) then
    the data rows `time  h1 h2 ... hN` (n_points+1 columns)."""
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) != n_points + 1 or not _isnum(p[0]):
                continue
            rows.append([float(v) for v in p])
    return np.array(rows)


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(args.out, "setup.json")) as f:
        setup = json.load(f)

    ours = {}
    with open(os.path.join(args.out, "ours", "gauges.csv")) as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        rows = list(rd)
    t_ours = np.array([float(r["t_s"]) for r in rows])
    for name in POINT_NAMES:
        if f"h_{name}" in cols:
            ours[name] = np.array([float(r[f"h_{name}"]) for r in rows])

    lfp = None
    lfp_path = os.path.join(args.out, "lisflood", "results",
                            f"{setup['tag']}.stage")
    if os.path.exists(lfp_path):
        lfp = read_lisflood_stage(lfp_path)

    ref = {}
    if args.published and os.path.exists(args.published):
        with open(args.published) as f:
            ref = json.load(f)["peak_depth_m"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for i, name in enumerate(POINT_NAMES):
        ax = axes.flat[i]
        if name in ref:
            b = ref[name]
            ax.axhspan(b["all_min"], b["all_max"], color="#94a3b8", alpha=0.22,
                       lw=0, label="published: all packages")
            ax.axhspan(b["cluster_min"], b["cluster_max"], color="#94a3b8",
                       alpha=0.40, lw=0, label="published: main cluster")
        if name in ours:
            ax.plot(t_ours / 60, ours[name], lw=2.0, color="#2b6cb0",
                    label="ours (torch)")
        if lfp is not None and lfp.shape[1] > i + 1:
            ax.plot(lfp[:, 0] / 60, lfp[:, i + 1], lw=1.6, ls="--",
                    color="#c05621", label="LISFLOOD-FP 8.1")
        ax.set_title(f"Point {i + 1}", fontsize=11, loc="left",
                     fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        if i >= 6:
            ax.set_xlabel("minutes")
        if i % 3 == 0:
            ax.set_ylabel("depth (m)")
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle("EA 2D benchmark Test 8A (Glasgow) - depth at the 9 "
                 "published stage points", fontsize=14, fontweight="bold",
                 x=0.01, ha="left")
    fig.tight_layout()
    png = os.path.join(args.out, "ea8_stages.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print(f"wrote {png}")

    summ = {"peak_depth_m": {}}
    for i, name in enumerate(POINT_NAMES):
        row = {}
        if name in ours:
            row["ours"] = round(float(ours[name].max()), 3)
        if lfp is not None and lfp.shape[1] > i + 1:
            row["lisflood"] = round(float(lfp[:, i + 1].max()), 3)
        if name in ref:
            b = ref[name]
            row["published_cluster"] = [b["cluster_min"], b["cluster_max"]]
            for who in ("ours", "lisflood"):
                if who in row:
                    v = row[who]
                    row[f"{who}_verdict"] = (
                        "in cluster" if b["cluster_min"] <= v <= b["cluster_max"]
                        else "in full spread"
                        if b["all_min"] <= v <= b["all_max"] else "OUTSIDE")
        summ["peak_depth_m"][name] = row
    with open(os.path.join(args.out, "ea8_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    for name, row in summ["peak_depth_m"].items():
        bits = [f"{k}={row[k]:.3f}" for k in ("ours", "lisflood") if k in row]
        if "published_cluster" in row:
            lo, hi = row["published_cluster"]
            bits.append(f"published {lo:.2f}-{hi:.2f}")
            bits.append(f"[ours: {row.get('ours_verdict', '?')}]")
        print(f"  {name}: " + "  ".join(bits))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--setup",
                   default=os.path.expanduser(
                       "~/Work/engines/ea_benchmark/4-Glasgow/Setup"))
    r.add_argument("--res", default="2m", choices=["2m", "0p5m"])
    r.add_argument("--out", default="output/ea8")
    r.add_argument("--sim-time", type=float, default=18000.0)
    r.add_argument("--save-every", type=float, default=60.0)
    r.add_argument("--device", default="auto")
    r.set_defaults(func=cmd_run)

    q = sub.add_parser("prep-lisflood")
    q.add_argument("--setup",
                   default=os.path.expanduser(
                       "~/Work/engines/ea_benchmark/4-Glasgow/Setup"))
    q.add_argument("--res", default="2m", choices=["2m", "0p5m"])
    q.add_argument("--out", default="output/ea8")
    q.add_argument("--sim-time", type=float, default=18000.0)
    q.set_defaults(func=cmd_prep_lisflood)

    p = sub.add_parser("plot")
    p.add_argument("--out", default="output/ea8")
    p.add_argument("--published", default="docs/ea8_published_envelope.json",
                   help="peak-depth bands from the EA benchmark report")
    p.set_defaults(func=cmd_plot)

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
