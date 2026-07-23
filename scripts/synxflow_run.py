#!/usr/bin/env python3
"""Run SynxFlow on the inputs exported for LISFLOOD-FP (PLAN.md D3, engine #2).

SynxFlow (HiPIMS successor, BSD-3, GPU) solves the *full* shallow-water
equations with a Godunov/HLLC finite-volume scheme, so it is
methodologically independent of the Bates-2010 inertial scheme that both
our solver and LISFLOOD-FP implement. Agreement across all three is a much
stronger statement than agreement with LISFLOOD alone.

It reads the very same export_ascii.py output the LISFLOOD cross-check
uses (dem.asc / n.asc / storm.rain) and writes `max_depth.asc` in ESRI
ASCII on the identical grid, so `crosscheck.py compare` consumes it with
no changes.

IMPORTANT: SynxFlow needs its own environment (Python <=3.11, numpy <2):
  ~/Work/.venv_synxflow/bin/python scripts/synxflow_run.py ...

Usage:
  ~/Work/.venv_synxflow/bin/python scripts/synxflow_run.py \
      --export output/export_cut --out output/crosscheck/synx_cut
"""

import argparse
import os
import shutil

import numpy as np

NODATA = -9999.0


def read_asc(path):
    hdr = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            hdr[k.lower()] = float(v)
        arr = np.loadtxt(f, dtype=np.float64)
    return arr, hdr


def write_asc(path, arr, hdr):
    head = (f"ncols         {int(hdr['ncols'])}\n"
            f"nrows         {int(hdr['nrows'])}\n"
            f"xllcorner     {hdr['xllcorner']:.6f}\n"
            f"yllcorner     {hdr['yllcorner']:.6f}\n"
            f"cellsize      {hdr['cellsize']:.6f}\n"
            f"NODATA_value  {NODATA:.6f}\n")
    with open(path, "w") as f:
        f.write(head)
        np.savetxt(f, np.nan_to_num(arr, nan=NODATA), fmt="%.4f")


def read_rain(path):
    """LISFLOOD .rain (value mm/h, time s) -> [[t_s, rate_m_s], ...]."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "seconds" in line:
                continue
            val, t = line.split()
            rows.append([float(t), float(val) / 1000.0 / 3600.0])
    src = np.array(rows, dtype=float)
    return src[np.argsort(src[:, 0])]


def par_sim_time(export):
    """sim_time from the generated LISFLOOD .par, so all engines run equally
    long (the rain series usually ends before the simulation does).

    Skip any probe*.par - those are short stability probes and must never
    set the production sim_time (a stray probe.par with sim_time 600 once
    made SynxFlow run only 600 s of a 3600 s storm)."""
    pars = sorted(f for f in os.listdir(export)
                  if f.endswith(".par") and not f.startswith("probe"))
    for f in pars:
        for line in open(os.path.join(export, f)):
            parts = line.split()
            if len(parts) == 2 and parts[0] == "sim_time":
                return float(parts[1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="dir with dem.asc/n.asc")
    ap.add_argument("--out", required=True, help="case folder for SynxFlow")
    ap.add_argument("--rain", default=None, help="default <export>/storm.rain")
    ap.add_argument("--sim-time", type=float, default=None,
                    help="seconds; default = end of the rain series")
    ap.add_argument("--save-every", type=float, default=900.0)
    ap.add_argument("--open-boundary", action="store_true",
                    help="let water leave the edges; default is a closed "
                         "wall, matching crosscheck.py --closed and "
                         "LISFLOOD-FP's default border")
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    from synxflow import IO, flood

    # flood.run() chdir's into the case folder, so every path we touch
    # afterwards has to be absolute.
    export = os.path.abspath(args.export)
    out = os.path.abspath(args.out)

    dem, hdr = read_asc(os.path.join(export, "dem.asc"))
    n, _ = read_asc(os.path.join(export, "n.asc"))
    rain = read_rain(args.rain or os.path.join(export, "storm.rain"))
    sim_time = args.sim_time or par_sim_time(export) or float(rain[-1, 0])
    print(f"{int(hdr['ncols'])} x {int(hdr['nrows'])} at {hdr['cellsize']} m, "
          f"{sim_time:.0f} s, rain peak "
          f"{rain[:, 1].max() * 1000 * 3600:.1f} mm/h")

    os.makedirs(out, exist_ok=True)
    sf_hdr = {"ncols": int(hdr["ncols"]), "nrows": int(hdr["nrows"]),
              "xllcorner": hdr["xllcorner"], "yllcorner": hdr["yllcorner"],
              "cellsize": hdr["cellsize"], "NODATA_value": NODATA}
    ras = IO.Raster(array=dem, header=sf_hdr)
    case = IO.InputModel(dem_data=ras, case_folder=out)
    case.set_boundary_condition(
        outline_boundary="open" if args.open_boundary else "rigid")
    # nodata cells are outside the domain anyway; fill so the grid is finite
    case.set_grid_parameter(manning=np.where(n == NODATA, 0.03, n))
    case.set_rainfall(rain_mask=0, rain_source=rain)
    case.set_runtime([0, sim_time, args.save_every, sim_time])
    case.write_input_files()
    print("inputs written, running...", flush=True)

    flood.run(out)

    outdir = os.path.join(out, "output")
    src = os.path.join(outdir, f"h_max_{sim_time:.0f}.asc")
    if not os.path.exists(src):                     # tolerate name variants
        cands = [f for f in os.listdir(outdir) if f.startswith("h_max_")]
        if not cands:
            raise SystemExit("SynxFlow produced no h_max_*.asc")
        src = os.path.join(outdir, sorted(cands)[-1])
    dst = os.path.join(out, "max_depth.asc")
    shutil.copyfile(src, dst)
    hmax, _ = read_asc(dst)
    wet = hmax[(hmax != NODATA) & np.isfinite(hmax)]
    print(f"wrote {dst}")
    print(f"  max {np.nanmax(wet):.2f} m, "
          f"p99 {np.nanpercentile(wet[wet > 0.001], 99):.2f} m, "
          f"wet>10cm {(wet > 0.1).sum()} cells")
    np.save(os.path.join(out, "max_depth.npy"),
            np.where(hmax == NODATA, np.nan, hmax).astype(np.float32))


if __name__ == "__main__":
    main()
