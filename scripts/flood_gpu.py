#!/usr/bin/env python3
"""GPU rain-on-grid flood solver (PLAN.md Phase B).

Same numerics as flood_sim.py — the Bates, Horritt & Fewtrell (2010)
inertial shallow-water scheme with semi-implicit Manning friction and the
donor-cell flux limiter — ported to torch (CUDA or CPU), fp32 state, with:

  - hyetograph rain (storms/<name>.json step functions)
  - rain-weight raster (roof/courtyard downspout rerouting, Phase A5)
  - spatially varying Manning n and infiltration rate
  - storm-drain inlets with per-inlet capacity caps (m3/s)
  - open boundary at grid edge / outside survey / water cells, with the
    outflow VOLUME accounted (mass balance closes to ~fp32 accuracy)
  - gauges: depth/velocity time series at probe points
  - hazard accumulators: max depth, max |v|, max h*(|v|+0.5)

Usage:
  python scripts/flood_gpu.py --terrain output/terrain_1.0 \
      --storm storms/v1_nov2025.json --out output/runs/S0__v1_nov2025
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from las_common import load_transform, utm_to_pixel

G = 9.81
# CFL ceiling for the unsplit 2-D HLLC update (see simulate(scheme=...)).
CFL_HLLC_MAX = 0.4

# Gully inlet hydraulics (see simulate(drain_mode=...)). A road gully is a
# weir while the water is shallow enough to spill freely over the grate bars,
# and an orifice once the grate is drowned and the throat controls; standard
# practice is to take whichever gives the smaller discharge at the current
# head. Coefficients are the usual SI values for a grated inlet.
DRAIN_WEIR_C = 1.66      # Q = C * P * h^1.5, P = grate perimeter (m)
DRAIN_ORIFICE_C = 0.6    # Q = C * A * sqrt(2 g h), A = clear opening (m2)
DRAIN_GRATE_PERIM_M = 1.8    # a ~0.6 x 0.3 m gully grate
DRAIN_GRATE_AREA_M2 = 0.09   # its clear opening after bars/blockage

# Opt-in erosion sub-model defaults (see `simulate(erosion=...)`). These are
# illustrative/tunable, not independently calibrated the way the PROPS
# infiltration/Manning table in bake_corridor.py is - say so wherever erosion
# results are surfaced.
EROSION_DEFAULTS = dict(
    k_rain=0.4,          # wetness gain per (m/s rain * erodible * s)
    k_flow=2.0,           # wetness gain per (m2/s unit-discharge excess * s)
    p_wet=0.0,             # unit-discharge (m2/s) threshold below which flow doesn't wet the soil
    p_crit=0.02,           # unit-discharge (m2/s) threshold below which no erosion occurs (Shields-like)
    k_erode=0.08,          # erosion_accum gain per (m2/s discharge excess * s), on a stochastic hit
    k_prob=40.0,           # stochastic-hit steepness: prob = 1 - exp(-erode_potential * k_prob * dt)
    k_dry=0.02,            # wetness decay per second when dry (soil drying out)
    infil_mud_floor=3.0,   # mm/h infiltration once a cell is fully "mud" (erosion_accum=1)
    manning_mud=0.03,      # Manning n once a cell is fully "mud" (smoother than vegetated/loose soil)
    update_every=20,       # erosion state + live infil/Manning refresh cadence, in solver steps
)


def _desing_vel(q, hs, h_vel):
    """hu -> u without dividing by a vanishing depth.

    Kurganov & Petrova's desingularisation: exact when h >> h_vel, and decays
    to zero (rather than exploding) as the cell dries. A plain hu/max(h,eps)
    leaves a velocity that is finite but arbitrary on a film of water, which
    on this terrain's 6-8 m/s supercritical streets is what the wave-speed
    estimates below are most sensitive to."""
    import torch
    return 2.0 * hs * q / (hs * hs + torch.clamp(hs, min=h_vel) ** 2)


def _hllc_x(hl, hr, ql, qr, pl, pr, h_vel):
    """HLLC flux across an x-face for U = (h, hu, hv).

    `ql/qr` are x-momentum (hu), `pl/pr` transverse (hv). Returns
    (F_h, F_hu, F_hv). Wave speeds follow Toro's two-rarefaction estimate
    with the dry-bed branches, which is what makes this robust across the
    wet/dry fronts that dominate a rain-on-grid start-up.
    """
    import torch
    g = G
    ul = _desing_vel(ql, hl, h_vel)
    ur = _desing_vel(qr, hr, h_vel)
    vl = _desing_vel(pl, hl, h_vel)
    vr = _desing_vel(pr, hr, h_vel)
    cl = torch.sqrt(g * torch.clamp(hl, min=0.0))
    cr = torch.sqrt(g * torch.clamp(hr, min=0.0))

    # two-rarefaction star state, then the dry-bed overrides
    h_star_c = 0.5 * (cl + cr) + 0.25 * (ul - ur)
    u_star = 0.5 * (ul + ur) + cl - cr
    sl = torch.minimum(ul - cl, u_star - h_star_c)
    sr = torch.maximum(ur + cr, u_star + h_star_c)
    dry_l, dry_r = hl <= 0.0, hr <= 0.0
    sl = torch.where(dry_l, ur - 2.0 * cr, sl)
    sr = torch.where(dry_l, ur + cr, sr)
    sl = torch.where(dry_r, ul - cl, sl)
    sr = torch.where(dry_r, ul + 2.0 * cl, sr)

    fl_h, fr_h = hl * ul, hr * ur
    fl_q = hl * ul * ul + 0.5 * g * hl * hl
    fr_q = hr * ur * ur + 0.5 * g * hr * hr

    den = sr - sl
    safe = torch.abs(den) > 1e-12
    dens = torch.where(safe, den, torch.ones_like(den))
    f_h = (sr * fl_h - sl * fr_h + sl * sr * (hr - hl)) / dens
    f_q = (sr * fl_q - sl * fr_q + sl * sr * (qr - ql)) / dens
    f_h = torch.where(sl >= 0, fl_h, torch.where(sr <= 0, fr_h, f_h))
    f_q = torch.where(sl >= 0, fl_q, torch.where(sr <= 0, fr_q, f_q))
    f_h = torch.where(safe, f_h, torch.zeros_like(f_h))
    f_q = torch.where(safe, f_q, torch.zeros_like(f_q))

    # contact wave: transverse momentum is simply advected, upwinded on the
    # sign of the mass flux. This is the "C" in HLLC and is what stops the
    # shear layers along building walls smearing out.
    f_p = f_h * torch.where(f_h >= 0, vl, vr)

    both_dry = dry_l & dry_r
    z = torch.zeros_like(f_h)
    return (torch.where(both_dry, z, f_h),
            torch.where(both_dry, z, f_q),
            torch.where(both_dry, z, f_p))


def simulate(dem, res, steps, duration, out_dir, *, manning=0.03,
             infil_mmh=None, valid=None, water=None, rain_weight=None,
             drains=None, gauges=None, sources=None, save_every=60.0, device="auto",
             alpha=0.7, h_min=1e-4, save_frames=True, init_depth=None,
             progress=True, dtype="float32", limiter="scale", progress_cb=None,
             erosion=False, erodible=None, erosion_seed=None, erosion_params=None,
             scheme="inertial", infil_mode="constant", infil_psi_m=None,
             infil_dtheta=None, drain_mode="capacity", drain_perim_m=None,
             drain_area_m2=None):
    """steps: list of (t0_s, t1_s, mm/h). drains: (rows, cols, cap_m3s)
    arrays. gauges: list of {'name','row','col'}. sources: list of
    {'row','col','series': [[t_s, q_m3s], ...]} point inflows, linearly
    interpolated in time (used by the EA Test 8A benchmark). Returns meta
    dict.

    limiter: 'scale' (default) rescales each cell's OUTflux so it never
    exports more volume than it holds - mass-conserving positivity fix
    that does not cap physical velocities. 'clip4' reproduces the legacy
    flood_sim.py clip (q <= h*res/4dt) exactly, for reference comparison
    only - it suppresses velocities whenever v > sqrt(g*hmax)/(4*alpha).

    progress_cb: optional callable(t, duration, stats_dict) invoked at each
    save_every tick (stats_dict has storage_m3, outflow_m3, max_h, wall_s);
    used by the interactive sandbox to relay live progress, no effect on
    the simulation itself.

    erosion: opt-in, default False. When False, none of the erosion-related
    parameters below have ANY effect - the exact same tensors and arithmetic
    run as before, so every existing published run and validate.py's tests
    stay bit-for-bit reproducible. When True, erodible soil/soft-material
    cells probabilistically get "wetter" and erode under rain/flow (driven
    by the cell-centred magnitude of the momentum solve's own qx/qy unit
    discharge - a standard local erosive-power proxy, so no new physics
    tensor is needed), which lowers their LIVE infiltration and Manning
    roughness toward a "mud" floor as erosion_accum -> 1. Puddling on
    eroded cells is an emergent consequence of that (lower infiltration +
    lower roughness under the same, unmodified SWE update), not separate
    bookkeeping - erosion never touches bed elevation or water volume, so
    it needs no new mass-balance term; `closure` stays exactly the same
    identity as always. erodible: (h,w) array in [0,1], per-cell erosion
    susceptibility (0 = never erodes, e.g. paved/impervious) - required
    when erosion=True. erosion_seed: seeds the stochastic erosion draw for
    reproducibility; None = nondeterministic. erosion_params: dict of
    overrides for EROSION_DEFAULTS."""
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    f32 = dict(dtype=getattr(torch, dtype), device=device)
    h, w = dem.shape
    area = res * res

    if valid is None:
        valid = np.isfinite(dem)
    dem_np = np.where(valid, dem, np.nanmin(dem[valid]) - 5.0).astype(np.float32)
    dem_t = torch.tensor(dem_np, **f32)

    n_np = np.broadcast_to(np.asarray(manning, dtype=np.float32), (h, w))
    n2x = torch.tensor(np.maximum(n_np[:, :-1], n_np[:, 1:]) ** 2, **f32)
    n2y = torch.tensor(np.maximum(n_np[:-1, :], n_np[1:, :]) ** 2, **f32)

    if rain_weight is None:
        rain_weight = valid.astype(np.float32)
    rw = torch.tensor(rain_weight.astype(np.float32), **f32)
    rw_sum = float(rain_weight.sum())

    # kept around (even when erosion is off) so the erosion branch below can
    # lerp toward it without changing infil_t's construction for the default
    # (non-erosion) path at all.
    infil_dry_np = (np.asarray(infil_mmh, np.float32) / 3.6e6
                     if infil_mmh is not None else np.zeros((h, w), np.float32))  # m/s
    infil_t = None
    if infil_mmh is not None and np.any(infil_mmh > 0):
        infil_t = torch.tensor(infil_dry_np, **f32)

    # Green-Ampt state. infil_mmh is read as the SATURATED conductivity K;
    # psi (wetting-front suction head, m) and dtheta (moisture deficit) are
    # the two extra soil parameters. Both are literature values per land-cover
    # class - like the Manning/infiltration table they sit beside, they are
    # NOT calibrated against observed infiltration here, and there is no
    # observed flood data in this study to calibrate them against.
    if infil_mode not in ("constant", "green_ampt"):
        raise ValueError(f"unknown infil_mode '{infil_mode}'")
    green_ampt = (infil_mode == "green_ampt") and infil_t is not None
    cum_infil = psi_dtheta = None
    if green_ampt:
        if infil_psi_m is None or infil_dtheta is None:
            raise ValueError("infil_mode='green_ampt' needs infil_psi_m and infil_dtheta")
        psi_np = np.broadcast_to(np.asarray(infil_psi_m, np.float32), (h, w))
        dth_np = np.broadcast_to(np.asarray(infil_dtheta, np.float32), (h, w))
        psi_dtheta = torch.tensor((psi_np * dth_np).astype(np.float32), **f32)
        cum_infil = torch.zeros((h, w), **f32)

    out_np = ~valid
    out_np[0, :] = out_np[-1, :] = out_np[:, 0] = out_np[:, -1] = True
    if water is not None:
        out_np = out_np | water
    keep = torch.tensor((~out_np).astype(np.float32), **f32)

    drain_idx = drain_cap = None
    if drains is not None and len(drains[0]) > 0:
        rr, cc, cap = drains
        drain_idx = (torch.tensor(rr.astype(np.int64), device=device),
                     torch.tensor(cc.astype(np.int64), device=device))
        drain_cap = torch.tensor(cap.astype(np.float32), **f32)
    if drain_mode not in ("capacity", "head_discharge"):
        raise ValueError(f"unknown drain_mode '{drain_mode}'")
    drain_head = (drain_mode == "head_discharge") and drain_idx is not None
    drain_perim = drain_area = None
    if drain_head:
        n_in = drain_cap.numel()
        drain_perim = torch.tensor(
            np.broadcast_to(np.asarray(drain_perim_m if drain_perim_m is not None
                                       else DRAIN_GRATE_PERIM_M, np.float32),
                            (n_in,)).copy(), **f32)
        drain_area = torch.tensor(
            np.broadcast_to(np.asarray(drain_area_m2 if drain_area_m2 is not None
                                       else DRAIN_GRATE_AREA_M2, np.float32),
                            (n_in,)).copy(), **f32)

    src_list = []
    for s in (sources or []):
        ser = np.asarray(s["series"], dtype=np.float64)
        src_list.append((int(s["row"]), int(s["col"]), ser[:, 0], ser[:, 1]))

    depth = torch.zeros((h, w), **f32)
    if init_depth is not None:
        depth = torch.tensor(init_depth.astype(np.float32), **f32)
    # HLLC carries cell-centred momentum (hu, hv) instead of the inertial
    # scheme's face-centred unit discharge; both are exposed to the rest of
    # the loop as qx/qy on faces so the erosion model, gauges and hazard
    # accumulators need no scheme-specific code.
    hllc = (scheme == "hllc")
    if hllc and alpha > CFL_HLLC_MAX:
        # The inertial scheme's default (0.7) is stable for it but not for an
        # unsplit 2-D Godunov update, which adds both sweeps' fluxes in one
        # step and so needs CFL <= ~1/2. Measured on the lake-at-rest bowl:
        # alpha 0.7 diverges to 30 m/s of spurious current, 0.4 sits at
        # 6e-5 m/s. Clamped rather than rejected so the same call works for
        # both schemes.
        alpha = CFL_HLLC_MAX
    # depth below which velocity is desingularised toward zero; a film this
    # thin carries no meaningful momentum and dividing by it is what makes
    # Godunov schemes blow up on a drying grid.
    h_vel = max(h_min, 1e-3)
    if scheme not in ("inertial", "hllc"):
        raise ValueError(f"unknown scheme '{scheme}' (expected 'inertial' or 'hllc')")
    hu = torch.zeros((h, w), **f32)
    hv = torch.zeros((h, w), **f32)
    n2c = torch.tensor(n_np ** 2, **f32)          # cell-centred, for HLLC friction
    qx = torch.zeros((h, w - 1), **f32)
    qy = torch.zeros((h - 1, w), **f32)
    max_depth = torch.zeros((h, w), **f32)
    max_vel = torch.zeros((h, w), **f32)
    max_haz = torch.zeros((h, w), **f32)

    erosion = bool(erosion)
    if erosion:
        if erodible is None:
            raise ValueError("erosion=True requires an 'erodible' array")
        ep = dict(EROSION_DEFAULTS)
        if erosion_params:
            ep.update(erosion_params)
        erod_np = np.asarray(erodible, dtype=np.float32)
        erod_t = torch.tensor(erod_np, **f32)
        wetness = torch.zeros((h, w), **f32)
        erosion_accum = torch.zeros((h, w), **f32)
        infil_dry_t = torch.tensor(infil_dry_np, **f32)
        infil_mud_floor_ms = ep["infil_mud_floor"] / 3.6e6
        manning_dry_t = torch.tensor(n_np, **f32)
        # erosion=True always makes infil_t a real tensor (even if the base
        # terrain had zero infiltration everywhere), so the infiltration
        # block below actually runs and reflects the live, eroding surface.
        infil_t = infil_dry_t.clone()
        gen = torch.Generator(device=device)
        if erosion_seed is not None:
            gen.manual_seed(int(erosion_seed))
        else:
            gen.seed()
        erosion_dt_accum = 0.0

    os.makedirs(out_dir, exist_ok=True)
    vol_in = vol_infil = vol_drain = vol_out = 0.0
    saved, gauge_rows, outflow_series, storage_series = [], [], [], []
    last_out, last_out_t = 0.0, 0.0
    t, next_save, it, si = 0.0, 0.0, 0, 0
    t0_wall = time.time()

    def rain_now(tt):
        for t0s, t1s, mmh in steps:
            if t0s <= tt < t1s:
                return mmh / 3.6e6  # m/s
        return 0.0

    while t < duration:
        hmax = float(depth.max().item())
        if hllc:
            # CFL on the true wave speed |u| + sqrt(gh), not just sqrt(gh):
            # on supercritical streets the advective part dominates and the
            # inertial scheme's estimate would be optimistic.
            umag = torch.sqrt(_desing_vel(hu, depth, h_vel) ** 2
                              + _desing_vel(hv, depth, h_vel) ** 2)
            wave = float((umag + torch.sqrt(G * torch.clamp(depth, min=0.0))).max().item())
            dt = min(alpha * res / max(wave, 1e-6), 5.0)
        else:
            dt = min(alpha * res / math.sqrt(G * max(hmax, 0.01)), 5.0)
        # quantize so fp32/fp64 runs take identical step sequences
        dt = max(math.floor(dt * 1000.0) / 1000.0, 1e-3)
        dt = min(dt, duration - t + 1e-9)

        if hllc:
            # --- Godunov finite volume, HLLC fluxes, Audusse hydrostatic
            # reconstruction. The reconstruction is what makes the scheme
            # well-balanced (lake-at-rest is preserved exactly) AND what
            # stops it fabricating volume on the retaining-wall steps that
            # made LISFLOOD-FP's 3-term scheme diverge on this same terrain.
            zb = dem_t
            # x faces
            zf = torch.maximum(zb[:, :-1], zb[:, 1:])
            hl = torch.clamp(depth[:, :-1] + zb[:, :-1] - zf, min=0.0)
            hr = torch.clamp(depth[:, 1:] + zb[:, 1:] - zf, min=0.0)
            ul = _desing_vel(hu[:, :-1], depth[:, :-1], h_vel)
            ur = _desing_vel(hu[:, 1:], depth[:, 1:], h_vel)
            vl = _desing_vel(hv[:, :-1], depth[:, :-1], h_vel)
            vr = _desing_vel(hv[:, 1:], depth[:, 1:], h_vel)
            fxh, fxq, fxp = _hllc_x(hl, hr, hl * ul, hr * ur, hl * vl, hr * vr, h_vel)
            # Audusse bed-source correction: the momentum flux each side sees
            # differs by the hydrostatic pressure of the reconstruction, and
            # that difference IS the bed slope source. Applying it this way
            # (rather than a separate centred dz term) is what keeps still
            # water still over a step.
            fxq_l = fxq + 0.5 * G * (depth[:, :-1] ** 2 - hl ** 2)
            fxq_r = fxq + 0.5 * G * (depth[:, 1:] ** 2 - hr ** 2)

            # y faces (rows increase downward; "t"=upper cell, "b"=lower)
            zf = torch.maximum(zb[:-1, :], zb[1:, :])
            ht = torch.clamp(depth[:-1, :] + zb[:-1, :] - zf, min=0.0)
            hb = torch.clamp(depth[1:, :] + zb[1:, :] - zf, min=0.0)
            vt = _desing_vel(hv[:-1, :], depth[:-1, :], h_vel)
            vb = _desing_vel(hv[1:, :], depth[1:, :], h_vel)
            ut = _desing_vel(hu[:-1, :], depth[:-1, :], h_vel)
            ub = _desing_vel(hu[1:, :], depth[1:, :], h_vel)
            # reuse the x solver with (v, u) swapped: same 1-D Riemann problem
            fyh, fyq, fyp = _hllc_x(ht, hb, ht * vt, hb * vb, ht * ut, hb * ub, h_vel)
            fyq_t = fyq + 0.5 * G * (depth[:-1, :] ** 2 - ht ** 2)
            fyq_b = fyq + 0.5 * G * (depth[1:, :] ** 2 - hb ** 2)

            lam = dt / res
            dh = torch.zeros_like(depth)
            dhu = torch.zeros_like(hu)
            dhv = torch.zeros_like(hv)
            dh[:, :-1] -= lam * fxh
            dh[:, 1:] += lam * fxh
            dhu[:, :-1] -= lam * fxq_l
            dhu[:, 1:] += lam * fxq_r
            dhv[:, :-1] -= lam * fxp
            dhv[:, 1:] += lam * fxp
            dh[:-1, :] -= lam * fyh
            dh[1:, :] += lam * fyh
            dhv[:-1, :] -= lam * fyq_t
            dhv[1:, :] += lam * fyq_b
            dhu[:-1, :] -= lam * fyp
            dhu[1:, :] += lam * fyp

            depth = torch.clamp(depth + dh, min=0.0)
            hu = hu + dhu
            hv = hv + dhv

            # semi-implicit Manning friction, same form as the inertial path
            spd = torch.sqrt(_desing_vel(hu, depth, h_vel) ** 2
                             + _desing_vel(hv, depth, h_vel) ** 2)
            wet = depth > h_min
            denom = 1.0 + dt * G * n2c * spd / torch.clamp(depth, min=h_vel) ** (4.0 / 3.0)
            hu = torch.where(wet, hu / denom, torch.zeros_like(hu))
            hv = torch.where(wet, hv / denom, torch.zeros_like(hv))

            # face-centred unit discharge for the shared downstream code
            qx = 0.5 * (hu[:, :-1] + hu[:, 1:])
            qy = 0.5 * (hv[:-1, :] + hv[1:, :])
            act_x = (hl > h_min) | (hr > h_min)
            act_y = (ht > h_min) | (hb > h_min)
            # face flow depths, so the shared velocity/hazard block below is
            # scheme-agnostic (it divides q by the depth momentum actually used)
            hfx = torch.clamp(0.5 * (hl + hr), min=h_vel)
            hfy = torch.clamp(0.5 * (ht + hb), min=h_vel)
        else:
            # --- Bates/Horritt/Fewtrell inertial scheme (default) ---
            # momentum x
            zl = dem_t[:, :-1] + depth[:, :-1]
            zr = dem_t[:, 1:] + depth[:, 1:]
            hflow = torch.clamp(torch.maximum(zl, zr) -
                                torch.maximum(dem_t[:, :-1], dem_t[:, 1:]), min=0.0)
            active = hflow > h_min
            hfx = torch.where(active, hflow, torch.ones_like(hflow))
            qn = (qx + G * hfx * dt * (zl - zr) / res) / (
                1.0 + G * dt * n2x * torch.abs(qx) / hfx ** (7.0 / 3.0))
            qx = torch.where(active, qn, torch.zeros_like(qn))
            act_x = active

            # momentum y
            zt = dem_t[:-1, :] + depth[:-1, :]
            zb = dem_t[1:, :] + depth[1:, :]
            hflow = torch.clamp(torch.maximum(zt, zb) -
                                torch.maximum(dem_t[:-1, :], dem_t[1:, :]), min=0.0)
            active = hflow > h_min
            hfy = torch.where(active, hflow, torch.ones_like(hflow))
            qn = (qy + G * hfy * dt * (zt - zb) / res) / (
                1.0 + G * dt * n2y * torch.abs(qy) / hfy ** (7.0 / 3.0))
            qy = torch.where(active, qn, torch.zeros_like(qn))
            act_y = active

            if limiter == "clip4":      # legacy flood_sim.py behavior
                qx = torch.clamp(qx, min=-depth[:, 1:] * res / dt / 4,
                                 max=depth[:, :-1] * res / dt / 4)
                qy = torch.clamp(qy, min=-depth[1:, :] * res / dt / 4,
                                 max=depth[:-1, :] * res / dt / 4)
            else:                       # outflux scaling: positivity, mass exact
                pos_x = torch.clamp(qx, min=0)
                neg_x = torch.clamp(-qx, min=0)
                pos_y = torch.clamp(qy, min=0)
                neg_y = torch.clamp(-qy, min=0)
                outr = torch.zeros_like(depth)
                outr[:, :-1] += pos_x
                outr[:, 1:] += neg_x
                outr[:-1, :] += pos_y
                outr[1:, :] += neg_y
                s = torch.clamp(depth * res / (outr * dt + 1e-12), max=1.0)
                qx = pos_x * s[:, :-1] - neg_x * s[:, 1:]
                qy = pos_y * s[:-1, :] - neg_y * s[1:, :]

            # continuity
            dv = torch.zeros_like(depth)
            dv[:, :-1] -= qx * dt / res
            dv[:, 1:] += qx * dt / res
            dv[:-1, :] -= qy * dt / res
            dv[1:, :] += qy * dt / res
            depth = depth + dv

        i_ms = rain_now(t)
        if i_ms > 0:
            depth = depth + i_ms * dt * rw
            vol_in += i_ms * dt * rw_sum * area
        for r, c, ts, qs in src_list:
            q = float(np.interp(t, ts, qs))
            if q != 0.0:
                depth[r, c] += q * dt / area
                vol_in += q * dt
        h_pre_sink = depth if hllc else None
        if infil_t is not None:
            if green_ampt:
                # Green-Ampt: the potential rate starts effectively unbounded
                # on dry ground and decays toward K as the wetting front
                # advances, f = K (1 + psi*dtheta/F). This is what turns
                # "the corridor's soils saturate in a 50-year storm" from an
                # interpretation of a fixed rate into something the run
                # actually does - infiltration falls as F grows, so a long or
                # repeated storm gets progressively less absorption.
                # Solve the Green-Ampt cumulative form implicitly over the
                # step rather than stepping the rate explicitly:
                #     F - F_n - B ln((F+B)/(F_n+B)) = K dt,   B = psi*dtheta
                # The explicit form is unusable at the start of a storm, where
                # F_n = 0 makes the rate unbounded and the first step
                # infiltrates whatever happens to be ponded - measured +116%
                # over the analytic solution at t=900 s. Newton from the
                # F_n + K dt lower bound converges in a couple of iterations
                # because g is monotone and smooth for F > 0.
                # Impervious cells (asphalt, roofs, water) have psi*dtheta = 0,
                # and Green-Ampt is undefined there: the Newton step evaluates
                # log(0/0) and 0/0 and returns NaN, which then propagates into
                # depth and poisons the whole domain on the first step. A
                # uniform-soil test cannot catch this - it only appears on real
                # heterogeneous land cover. Solve only where there IS soil and
                # fall back to the constant rate (which is 0 for impervious)
                # everywhere else.
                b = psi_dtheta
                has_soil = b > 0
                b_safe = torch.where(has_soil, b, torch.ones_like(b))
                fn = cum_infil
                f_new = fn + infil_t * dt
                for _ in range(4):
                    g_ = (f_new - fn
                          - b_safe * torch.log((f_new + b_safe) / (fn + b_safe))
                          - infil_t * dt)
                    gp = 1.0 - b_safe / (f_new + b_safe)
                    f_new = f_new - g_ / torch.clamp(gp, min=1e-6)
                    f_new = torch.maximum(f_new, fn)
                f_new = torch.where(has_soil, f_new, fn + infil_t * dt)
                di = torch.minimum(depth, f_new - fn)
                # supply-limited cells only advance the front by what actually
                # went in, so a dry spell does not credit them with soil moisture
                cum_infil = fn + di
            else:
                di = torch.minimum(depth, infil_t * dt)
            depth = depth - di
            vol_infil += float(di.sum().item()) * area
        if drain_idx is not None:
            d = depth[drain_idx]
            if drain_head:
                # A fixed capacity drains a puddle 1 mm deep as hard as one
                # 300 mm deep, which is exactly backwards: a gully takes what
                # the head over its grate allows. Making inlet performance
                # depend on the ponding a design produces is also what lets
                # inlet placement be optimised against that design.
                dpos = torch.clamp(d, min=0.0)
                q_weir = DRAIN_WEIR_C * drain_perim * dpos ** 1.5
                q_orif = DRAIN_ORIFICE_C * drain_area * torch.sqrt(2.0 * G * dpos)
                q = torch.minimum(q_weir, q_orif)
                if drain_cap is not None:
                    q = torch.minimum(q, drain_cap)   # the pipe still limits it
                take = torch.minimum(d, q * dt / area)
            else:
                take = torch.minimum(d, drain_cap * dt / area)
            depth[drain_idx] = d - take
            vol_drain += float(take.sum().item()) * area
        if hllc and h_pre_sink is not None:
            # water that infiltrates or drops down a gully leaves at the local
            # velocity, so it takes its share of momentum with it. Scaling
            # (hu, hv) by the depth ratio keeps u fixed across the sink; not
            # doing so would leave the old momentum on less water and spin the
            # remaining film up to a fictitious speed.
            hu = hu * depth / torch.clamp(h_pre_sink, min=1e-12)
            hv = hv * depth / torch.clamp(h_pre_sink, min=1e-12)

        if erosion:
            # accumulate elapsed sim-time between erosion ticks so the
            # (expensive) Manning-face rebuild can be throttled without
            # under-counting the exposure that happened on skipped steps.
            erosion_dt_accum += dt
            if it % ep["update_every"] == 0:
                dtw = erosion_dt_accum
                erosion_dt_accum = 0.0
                # cell-centred unit-discharge magnitude (m2/s): qx/qy are
                # already exactly this at each face, from the momentum solve
                # above - a standard local erosive/stream-power proxy, so no
                # new physics tensor is needed.
                qx_c = torch.zeros_like(depth)
                qx_c[:, :-1] += qx.abs() * 0.5
                qx_c[:, 1:] += qx.abs() * 0.5
                qy_c = torch.zeros_like(depth)
                qy_c[:-1, :] += qy.abs() * 0.5
                qy_c[1:, :] += qy.abs() * 0.5
                power = torch.sqrt(qx_c * qx_c + qy_c * qy_c)

                wet_gain = erod_t * (ep["k_rain"] * i_ms +
                                      ep["k_flow"] * torch.clamp(power - ep["p_wet"], min=0.0))
                wetness = torch.clamp(wetness + wet_gain * dtw - ep["k_dry"] * dtw, 0.0, 1.0)

                # probabilistic detachment: a Poisson-style "did an erosion
                # event happen here this tick" draw, driven by how far local
                # discharge exceeds the critical threshold - gives a patchy,
                # organic pattern instead of a smooth deterministic field,
                # while still being grounded in the real local flow.
                erode_potential = erod_t * torch.clamp(power - ep["p_crit"], min=0.0)
                prob = 1.0 - torch.exp(-erode_potential * ep["k_prob"] * dtw)
                hit = torch.rand(prob.shape, generator=gen, device=device,
                                  dtype=prob.dtype) < prob
                gain = torch.where(hit, erode_potential * ep["k_erode"] * dtw,
                                    torch.zeros_like(erosion_accum))
                erosion_accum = torch.clamp(erosion_accum + gain, 0.0, 1.0)

                # live property feedback: erosion_accum -> 1 slides infil
                # and Manning toward the "mud" floor. Puddling on eroded
                # cells then falls straight out of the same unmodified SWE
                # update (lower infiltration + lower roughness), no separate
                # bookkeeping needed.
                infil_t = infil_dry_t * (1.0 - erosion_accum) + infil_mud_floor_ms * erosion_accum
                manning_live = manning_dry_t * (1.0 - erosion_accum) + ep["manning_mud"] * erosion_accum
                n2x = torch.maximum(manning_live[:, :-1], manning_live[:, 1:]) ** 2
                n2y = torch.maximum(manning_live[:-1, :], manning_live[1:, :]) ** 2

        # open boundary: count then remove
        esc = depth * (1.0 - keep)
        vol_out += float(esc.sum().item()) * area
        depth = torch.clamp(depth * keep, min=0.0)
        if hllc:
            # momentum leaves with the water it belonged to; leaving it behind
            # on an emptied outflow cell would re-inject it next step
            hu = hu * keep
            hv = hv * keep

        torch.maximum(max_depth, depth, out=max_depth)
        if it % 5 == 0:
            # face velocity = q / face flow depth (what momentum actually
            # used) - dividing by the cell's post-update depth explodes in
            # freshly drained cells and poisons the hazard maps
            vfx = torch.where(act_x, qx / hfx, torch.zeros_like(qx))
            vfy = torch.where(act_y, qy / hfy, torch.zeros_like(qy))
            vx = torch.zeros_like(depth)
            vy = torch.zeros_like(depth)
            vx[:, :-1] += vfx / 2
            vx[:, 1:] += vfx / 2
            vy[:-1, :] += vfy / 2
            vy[1:, :] += vfy / 2
            vel = torch.hypot(vx, vy)
            vel = torch.where(depth > 0.01, vel, torch.zeros_like(vel))
            torch.maximum(max_vel, vel, out=max_vel)
            torch.maximum(max_haz, depth * (vel + 0.5), out=max_haz)

        if t >= next_save:
            if save_frames:
                fn = f"depth_{int(round(t)):06d}.npy"
                np.save(os.path.join(out_dir, fn),
                        depth.cpu().numpy().astype(np.float16))
                saved.append(fn)
            storage = float(depth.sum().item()) * area
            storage_series.append([round(t, 1), round(storage, 1)])
            if t > 0:
                rate = (vol_out - last_out) / max(t - last_out_t, 1e-9)
                outflow_series.append([round(t, 1), round(rate, 3)])
            last_out, last_out_t = vol_out, t
            if gauges:
                dc = depth.cpu().numpy()
                vc = max_vel.cpu().numpy()
                gauge_rows.append(
                    [round(t, 1)] +
                    [round(float(dc[g["row"], g["col"]]), 4) for g in gauges] +
                    [round(float(vc[g["row"], g["col"]]), 3) for g in gauges])
            if progress:
                print(f"  t={t:7.1f}s dt={dt:5.2f}s rain={i_ms * 3.6e6:5.1f}mm/h "
                      f"max_h={hmax:5.2f} storage={storage:9.0f}m3 "
                      f"out={vol_out:8.0f}m3 [{time.time() - t0_wall:5.0f}s wall]",
                      flush=True)
            if progress_cb is not None:
                progress_cb(t, duration, {"storage_m3": storage, "outflow_m3": vol_out,
                                          "max_h": hmax, "wall_s": time.time() - t0_wall})
            next_save += save_every
            si += 1
        t += dt
        it += 1

    # NB: name must not match the depth_*.npy frame glob
    np.save(os.path.join(out_dir, "final_depth.npy"),
            depth.cpu().numpy().astype(np.float32))
    np.save(os.path.join(out_dir, "max_depth.npy"),
            max_depth.cpu().numpy().astype(np.float32))
    np.save(os.path.join(out_dir, "max_vel.npy"),
            max_vel.cpu().numpy().astype(np.float32))
    np.save(os.path.join(out_dir, "max_hazard.npy"),
            max_haz.cpu().numpy().astype(np.float32))
    if erosion:
        np.save(os.path.join(out_dir, "final_erosion.npy"),
                erosion_accum.cpu().numpy().astype(np.float32))
        np.save(os.path.join(out_dir, "final_wetness.npy"),
                wetness.cpu().numpy().astype(np.float32))

    stored = float(depth.sum().item()) * area
    closure = vol_in - (vol_infil + vol_drain + vol_out + stored)
    meta = {
        "steps_mmh": steps, "duration": duration, "res": res,
        "device": device, "n_iterations": it,
        "wall_s": round(time.time() - t0_wall, 1),
        "frames": saved,
        "vol_rain_m3": round(vol_in, 2),
        "vol_infiltrated_m3": round(vol_infil, 2),
        "vol_drained_m3": round(vol_drain, 2),
        "vol_outflow_m3": round(vol_out, 2),
        "vol_stored_end_m3": round(stored, 2),
        "closure_m3": round(closure, 2),
        "closure_rel": round(closure / max(vol_in, 1e-9), 6),
        "outflow_series_m3s": outflow_series,
        "storage_series_m3": storage_series,
    }
    meta["erosion"] = erosion
    if erosion:
        erosion_np = erosion_accum.cpu().numpy()
        wetness_np = wetness.cpu().numpy()
        meta["erosion_params"] = ep
        meta["max_erosion_frac"] = round(float(erosion_np.max()), 4)
        meta["eroded_area_m2"] = round(
            float(((erod_np > 0) & (erosion_np > 0.5)).sum()) * area, 1)
        meta["wet_area_m2"] = round(float((wetness_np > 0.5).sum()) * area, 1)
    if gauges:
        meta["gauges"] = [{"name": g["name"], "row": g["row"], "col": g["col"]}
                          for g in gauges]
        import csv
        with open(os.path.join(out_dir, "gauges.csv"), "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["t_s"] + [f"h_{g['name']}" for g in gauges] +
                          [f"vmax_{g['name']}" for g in gauges])
            wcsv.writerows(gauge_rows)
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if progress:
        print(f"done in {meta['wall_s']:.0f}s wall, {it} steps. mass balance: "
              f"rain {vol_in:.0f} = infil {vol_infil:.0f} + drains {vol_drain:.0f} "
              f"+ outflow {vol_out:.0f} + stored {stored:.0f} "
              f"(closure {meta['closure_rel'] * 100:.3f}%)")
    return meta


def load_terrain(terrain):
    dem = np.load(os.path.join(terrain, "dem.npy")).astype(np.float32)
    t = load_transform(os.path.join(terrain, "dem_transform.json"))
    masks = np.load(os.path.join(terrain, "masks.npz"))
    manning = np.load(os.path.join(terrain, "manning.npy"))
    infil = np.load(os.path.join(terrain, "infil_mmh.npy"))
    rain_w = np.load(os.path.join(terrain, "rain_weight.npy"))
    epath = os.path.join(terrain, "erodible.npy")
    # older terrain dirs (built before the erosion sub-model) simply have no
    # erodible cells - erosion=True on one of them is then a harmless no-op.
    erodible = np.load(epath) if os.path.exists(epath) else np.zeros_like(dem)
    gauges = []
    gpath = os.path.join(terrain, "gauges.json")
    if os.path.exists(gpath):
        with open(gpath) as f:
            for g in json.load(f):
                c, r = utm_to_pixel(t, g["x"], g["y"])
                gauges.append({"name": g["name"], "row": int(r), "col": int(c)})
    return dem, t, masks, manning, infil, rain_w, gauges, erodible


def load_soil(terrain):
    """Green-Ampt soil rasters, or (None, None) if this terrain predates them.

    Kept out of load_terrain's return tuple deliberately: that tuple is
    unpacked positionally by the sandbox and the batch scripts, and widening
    it would break every caller for a layer only the green_ampt path needs."""
    psi_p = os.path.join(terrain, "infil_psi_m.npy")
    dth_p = os.path.join(terrain, "infil_dtheta.npy")
    if os.path.exists(psi_p) and os.path.exists(dth_p):
        return np.load(psi_p), np.load(dth_p)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", required=True)
    ap.add_argument("--storm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drains", help="drains.npz (rows, cols, cap)")
    ap.add_argument("--duration", type=float, help="override storm sim duration")
    ap.add_argument("--save-every", type=float, default=60.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--erosion", action="store_true",
                     help="opt-in erosion sub-model (see EROSION_DEFAULTS); "
                          "off by default, no effect on existing results")
    ap.add_argument("--erosion-seed", type=int, default=None)
    ap.add_argument("--scheme", default="inertial", choices=["inertial", "hllc"],
                     help="'inertial' = Bates et al. 3-term (default, fast); "
                          "'hllc' = shock-capturing Godunov, for supercritical "
                          "street flow and stepped terrain")
    ap.add_argument("--infil-mode", default="constant",
                     choices=["constant", "green_ampt"],
                     help="'green_ampt' needs infil_psi_m.npy/infil_dtheta.npy "
                          "in the terrain dir and makes infiltration decay as "
                          "the wetting front advances")
    ap.add_argument("--drain-mode", default="capacity",
                     choices=["capacity", "head_discharge"],
                     help="'head_discharge' makes each gully take what the head "
                          "over its grate allows (weir/orifice), capped by its pipe")
    args = ap.parse_args()

    dem, t, masks, manning, infil, rain_w, gauges, erodible = load_terrain(args.terrain)
    psi, dtheta = load_soil(args.terrain)
    if args.infil_mode == "green_ampt" and psi is None:
        sys.exit(f"--infil-mode green_ampt needs infil_psi_m.npy + "
                 f"infil_dtheta.npy in {args.terrain} (rebuild with build_terrain.py)")
    with open(args.storm) as f:
        storm = json.load(f)
    drains = None
    if args.drains:
        dz = np.load(args.drains)
        drains = (dz["rows"], dz["cols"], dz["cap"])
        print(f"{len(dz['rows'])} drain inlets, "
              f"total capacity {dz['cap'].sum():.2f} m3/s")

    print(f"DEM {dem.shape[1]} x {dem.shape[0]} at {t['res']} m | "
          f"storm {storm['name']} ({storm['total_mm']} mm) | "
          f"sim {args.duration or storm['duration']:.0f}s"
          + f" | scheme {args.scheme} | infil {args.infil_mode} | drains {args.drain_mode}"
          + (" | erosion ON" if args.erosion else ""))
    simulate(dem, t["res"], storm["steps"],
             args.duration or storm["duration"], args.out,
             manning=manning, infil_mmh=infil, valid=masks["valid"],
             water=masks["water"], rain_weight=rain_w, drains=drains,
             gauges=gauges, save_every=args.save_every, device=args.device,
             erosion=args.erosion, erodible=erodible, erosion_seed=args.erosion_seed,
             save_frames=not args.no_frames, scheme=args.scheme,
             infil_mode=args.infil_mode, infil_psi_m=psi, infil_dtheta=dtheta,
             drain_mode=args.drain_mode)


if __name__ == "__main__":
    main()
