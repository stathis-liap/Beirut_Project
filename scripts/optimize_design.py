#!/usr/bin/env python3
"""Gradient-based green-infrastructure placement on the Beirut corridor.

Poses the planner's question as topology optimisation: given a fixed area
budget of permeable surface, WHERE should it go to minimise flooded street
area? The design is a continuous density field rho in [0,1] (see
diff_flood.py), the flood response is differentiable in it, and the budget is
imposed by a bisection projection each iteration - the standard optimality
-criterion update from structural topology optimisation.

Compared against the two heuristics this replaces:

  uniform  spread the budget evenly over the whole eligible area - what a
           blanket "make X% of the street permeable" policy gives.
  greedy   rank cells by how deep they pond in the do-nothing case and fill
           the worst ones until the budget runs out. This is the same logic
           optimize_drains.py uses to site inlets, and it is the strongest
           intuitive baseline: put the sponge where the water is.

Greedy is myopic in a specific, checkable way: it sees where water ENDS UP,
not where intercepting it does most good. On a corridor with a continuous
6% fall, that distinction is the whole design problem.

Usage:
  python scripts/optimize_design.py --terrain output/terrain_cut_0.5_v3 \\
      --rows 1000 1128 --cols 240 368 --iters 40
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from diff_flood import (forward, density_filter, flooded_area_objective,
                        interpolate_material)


def project_to_budget(rho, budget_cells, eligible, lo=-50.0, hi=50.0, iters=60):
    """Shift rho by a scalar and clip to [0,1] so it sums to the budget.

    Bisection on the shift rather than a plain rescale: rescaling a field that
    is already saturated at 1 quietly overshoots the budget once you clip,
    which lets the "optimised" design spend more area than the baselines and
    makes the comparison meaningless."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = torch.clamp(rho + mid, 0.0, 1.0) * eligible
        if float(s.sum()) > budget_cells:
            hi = mid
        else:
            lo = mid
    return torch.clamp(rho + 0.5 * (lo + hi), 0.0, 1.0) * eligible


def run_design(rho, dem, gully, rain, n_steps, dt, res, keep, street, rain_w,
               filt, chunk=0):
    d = forward(dem, density_filter(rho, filt), gully, rain, n_steps, dt, res,
                keep=keep, rain_weight=rain_w, chunk=chunk)
    return d, flooded_area_objective(d, street, res=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="output/terrain_cut_0.5_v3")
    ap.add_argument("--rows", type=int, nargs=2, default=[1000, 1128])
    ap.add_argument("--cols", type=int, nargs=2, default=[240, 368])
    ap.add_argument("--storm", default="storms/v1_nov2025.json")
    ap.add_argument("--sim-time", type=float, default=300.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--budget-frac", type=float, default=0.15,
                    help="fraction of eligible street area allowed to be permeable")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--filter-cells", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/design_opt")
    args = ap.parse_args()

    r0, r1 = args.rows
    c0, c1 = args.cols
    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    T = args.terrain
    dem_np = np.load(os.path.join(T, "dem.npy"))[r0:r1, c0:c1].astype(np.float32)
    m = np.load(os.path.join(T, "masks.npz"))
    valid = m["valid"][r0:r1, c0:c1]
    building = m["building"][r0:r1, c0:c1]
    water = m["water"][r0:r1, c0:c1]
    rain_w_np = np.load(os.path.join(T, "rain_weight.npy"))[r0:r1, c0:c1].astype(np.float32)
    res = json.load(open(os.path.join(T, "dem_transform.json")))["res"]
    with open(args.storm) as f:
        storm = json.load(f)

    dem_np = np.where(valid, dem_np, np.nanmin(dem_np[valid]) - 5.0)
    dem = torch.tensor(dem_np, dtype=torch.float32, device=dev)
    # open boundary at the crop edge and on water, exactly as flood_gpu does
    out_np = (~valid) | water
    out_np[0, :] = out_np[-1, :] = out_np[:, 0] = out_np[:, -1] = True
    keep = torch.tensor((~out_np).astype(np.float32), device=dev)
    # only open street can be made permeable, and only street counts as flooded
    eligible_np = valid & ~building & ~water
    eligible = torch.tensor(eligible_np.astype(np.float32), device=dev)
    street = eligible.clone()
    rain_w = torch.tensor(rain_w_np, device=dev)
    gully = torch.zeros_like(dem)

    n_steps = int(args.sim_time / args.dt)
    rain = [(float(a), float(b), float(c)) for a, b, c in storm["steps"]]
    budget = args.budget_frac * float(eligible.sum())
    print(f"crop {dem.shape[0]}x{dem.shape[1]} at {res} m | eligible street "
          f"{float(eligible.sum())*res*res:.0f} m2 | budget {budget*res*res:.0f} m2 "
          f"({args.budget_frac:.0%}) | {n_steps} steps of {args.dt}s")

    def evaluate(rho):
        with torch.no_grad():
            d, j = run_design(rho, dem, gully, rain, n_steps, args.dt, res, keep,
                              street, rain_w, args.filter_cells)
        return float(j), d

    # --- baseline 0: do nothing --------------------------------------------
    zero = torch.zeros_like(dem)
    j_base, d_base = evaluate(zero)
    print(f"\n  do nothing        flooded {j_base:9.1f} m2")

    # --- baseline 1: uniform ------------------------------------------------
    uni = project_to_budget(torch.full_like(dem, args.budget_frac), budget, eligible)
    j_uni, _ = evaluate(uni)

    # --- baseline 2: greedy on baseline ponding -----------------------------
    db = (d_base * eligible).flatten()
    k = int(budget)
    idx = torch.topk(db, k).indices
    greedy = torch.zeros_like(dem).flatten()
    greedy[idx] = 1.0
    greedy = (greedy.reshape(dem.shape) * eligible)
    j_greedy, _ = evaluate(greedy)

    # --- gradient-based -----------------------------------------------------
    rho = torch.full_like(dem, args.budget_frac).requires_grad_(True)
    opt = torch.optim.Adam([rho], lr=args.lr)
    hist = []
    for it in range(args.iters):
        opt.zero_grad()
        _, j = run_design(rho, dem, gully, rain, n_steps, args.dt, res, keep,
                          street, rain_w, args.filter_cells, chunk=args.chunk)
        j.backward()
        opt.step()
        with torch.no_grad():
            rho.data = project_to_budget(rho.data, budget, eligible)
        hist.append(float(j.detach()))
        if it % 5 == 0 or it == args.iters - 1:
            print(f"    iter {it:3d}  flooded {float(j):9.1f} m2")
    j_opt, d_opt = evaluate(rho.detach())

    def pct(j):
        return 100.0 * (j_base - j) / j_base

    print(f"\n  {'design':18s} {'flooded m2':>12s} {'vs do-nothing':>14s}")
    print(f"  {'do nothing':18s} {j_base:12.1f} {'-':>14s}")
    print(f"  {'uniform':18s} {j_uni:12.1f} {pct(j_uni):13.1f}%")
    print(f"  {'greedy (ponding)':18s} {j_greedy:12.1f} {pct(j_greedy):13.1f}%")
    print(f"  {'gradient-based':18s} {j_opt:12.1f} {pct(j_opt):13.1f}%")
    print(f"\n  gradient-based vs greedy: {100*(j_greedy-j_opt)/max(j_greedy,1e-9):+.1f}% "
          f"further reduction at the same area budget")

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "rho_opt.npy"), rho.detach().cpu().numpy())
    np.save(os.path.join(args.out, "rho_greedy.npy"), greedy.cpu().numpy())
    np.save(os.path.join(args.out, "depth_base.npy"), d_base.cpu().numpy())
    np.save(os.path.join(args.out, "depth_opt.npy"), d_opt.cpu().numpy())
    json.dump({"rows": args.rows, "cols": args.cols, "res": res,
               "budget_frac": args.budget_frac, "sim_time_s": args.sim_time,
               "flooded_m2": {"do_nothing": j_base, "uniform": j_uni,
                              "greedy": j_greedy, "gradient": j_opt},
               "history": hist},
              open(os.path.join(args.out, "result.json"), "w"), indent=2)
    print(f"  wrote {args.out}/")


if __name__ == "__main__":
    main()
