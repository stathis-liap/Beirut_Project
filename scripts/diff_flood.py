#!/usr/bin/env python3
"""Differentiable rain-on-grid shallow-water solver, for gradient-based
green-infrastructure and gully design.

Why a second solver rather than autograd over flood_gpu.simulate(): that one
writes state in place (`depth[drain_idx] = ...`), pulls scalars out with
`.item()` every step to pick a CFL-limited dt, and writes frames to disk. All
three are fine for a forward run and fatal for a tape. This module keeps the
same physics - Bates/Horritt/Fewtrell inertial momentum, semi-implicit Manning
friction, outflux-scaling positivity limiter, rain, Green-Ampt-free constant
infiltration, head-discharge gullies - but written functionally, with a fixed
dt so the graph does not depend on the state through a `max()`.

The point is dJ/d(design), where the design is continuous fields rather than a
discrete layout:

  rho   in [0,1] per cell: "greenness". Material properties are interpolated
        between the paved and green endpoints, exactly as SIMP interpolates
        stiffness in structural topology optimisation. rho=0 is today's
        asphalt, rho=1 is the corridor's engineered permeable build-up.
  gully in [0,1] per cell: inlet density, scaling the grate perimeter/opening
        of a notional gully at that cell.

Both are continuous so the flood response is differentiable in them; the final
answer is thresholded back to a buildable layout. Together with an area budget
this is topology optimisation for urban drainage - the design question the
corridor study currently answers with a greedy heuristic (optimize_drains.py)
and a fixed cross-section.

Gradients are exact to autograd, and `gradcheck` below verifies them against
central finite differences of the same forward model.
"""

import math

import torch
import torch.nn.functional as F

G = 9.81

# Material endpoints the density field interpolates between. Paved values are
# build_terrain.py's PAVED row; green values are bake_corridor.py's PROPS for
# the corridor's garden/bioswale build-up. Interpolation is linear in rho for
# infiltration and depression, and in rho for Manning too - a designed surface
# is a physical blend of the two, not a stiffness proxy, so the usual SIMP
# penalty exponent would misrepresent it.
PAVED = dict(infil_mmh=5.0, manning=0.016, depression_m=0.0)
GREEN = dict(infil_mmh=150.0, manning=0.100, depression_m=0.10)

# Gully hydraulics, mirroring flood_gpu.py's head-discharge inlets.
DRAIN_WEIR_C = 1.66
DRAIN_ORIFICE_C = 0.6
GRATE_PERIM_M = 1.8
GRATE_AREA_M2 = 0.09


def interpolate_material(rho):
    """Design density -> (infiltration m/s, Manning n, depression m)."""
    infil = (PAVED["infil_mmh"] + rho * (GREEN["infil_mmh"] - PAVED["infil_mmh"])) / 3.6e6
    manning = PAVED["manning"] + rho * (GREEN["manning"] - PAVED["manning"])
    depr = PAVED["depression_m"] + rho * (GREEN["depression_m"] - PAVED["depression_m"])
    return infil, manning, depr


def density_filter(rho, radius_cells):
    """Blur the density field before it is used.

    Standard in topology optimisation and not cosmetic: without it the
    optimiser finds checkerboard solutions that exploit the discretisation
    (alternating paved/green cells) and are not buildable. The filter makes
    the design mesh-independent at the length scale you can actually
    construct - here, a few metres of streetscape."""
    if radius_cells <= 0:
        return rho
    k = int(2 * radius_cells + 1)
    kern = torch.ones((1, 1, k, k), dtype=rho.dtype, device=rho.device) / (k * k)
    return F.conv2d(rho[None, None], kern, padding=k // 2)[0, 0]


def _pad_x(t):
    """(H, W-1) face field -> (inflow, outflow) as (H, W) cell fields."""
    return F.pad(t, (1, 0)), F.pad(t, (0, 1))


def _pad_y(t):
    return F.pad(t, (0, 0, 1, 0)), F.pad(t, (0, 0, 0, 1))


def forward(dem, rho, gully, rain_steps, n_steps, dt, res, *, keep=None,
            rain_weight=None, h_min=1e-4, chunk=0):
    """Run the design forward and return the final depth field.

    `rho`/`gully` are the design tensors gradients are wanted for. `chunk > 0`
    turns on gradient checkpointing every `chunk` steps: the tape then stores
    one state per chunk instead of per step, trading a second forward pass for
    O(n_steps/chunk) memory. Without it a 3600 s storm at dt=0.05 s would need
    72,000 saved states and no GPU holds that.
    """
    h, w = dem.shape
    depth = torch.zeros_like(dem)
    qx = torch.zeros((h, w - 1), dtype=dem.dtype, device=dem.device)
    qy = torch.zeros((h - 1, w), dtype=dem.dtype, device=dem.device)
    if keep is None:
        keep = torch.ones_like(dem)
    if rain_weight is None:
        rain_weight = torch.ones_like(dem)

    infil, manning, depr = interpolate_material(rho)
    bed = dem - depr                      # detention lowers the cell
    n2x = torch.maximum(manning[:, :-1], manning[:, 1:]) ** 2
    n2y = torch.maximum(manning[:-1, :], manning[1:, :]) ** 2

    def step(depth, qx, qy, t_scalar):
        t = float(t_scalar)
        # --- momentum ---
        zl = bed[:, :-1] + depth[:, :-1]
        zr = bed[:, 1:] + depth[:, 1:]
        hf = torch.clamp(torch.maximum(zl, zr)
                         - torch.maximum(bed[:, :-1], bed[:, 1:]), min=0.0)
        act = hf > h_min
        hfs = torch.where(act, hf, torch.ones_like(hf))
        qx = torch.where(act,
                         (qx + G * hfs * dt * (zl - zr) / res)
                         / (1.0 + G * dt * n2x * torch.abs(qx) / hfs ** (7.0 / 3.0)),
                         torch.zeros_like(qx))

        zt = bed[:-1, :] + depth[:-1, :]
        zb = bed[1:, :] + depth[1:, :]
        hf = torch.clamp(torch.maximum(zt, zb)
                         - torch.maximum(bed[:-1, :], bed[1:, :]), min=0.0)
        act = hf > h_min
        hfs = torch.where(act, hf, torch.ones_like(hf))
        qy = torch.where(act,
                         (qy + G * hfs * dt * (zt - zb) / res)
                         / (1.0 + G * dt * n2y * torch.abs(qy) / hfs ** (7.0 / 3.0)),
                         torch.zeros_like(qy))

        # --- positivity: scale each cell's outflux to what it actually holds ---
        px, nx = torch.clamp(qx, min=0), torch.clamp(-qx, min=0)
        py, ny = torch.clamp(qy, min=0), torch.clamp(-qy, min=0)
        in_x, out_x = _pad_x(px)
        in_nx, out_nx = _pad_x(nx)
        in_y, out_y = _pad_y(py)
        in_ny, out_ny = _pad_y(ny)
        outr = out_x + in_nx + out_y + in_ny
        s = torch.clamp(depth * res / (outr * dt + 1e-12), max=1.0)
        qx = px * s[:, :-1] - nx * s[:, 1:]
        qy = py * s[:-1, :] - ny * s[1:, :]

        # --- continuity ---
        fx, fy = qx * dt / res, qy * dt / res
        in_x, out_x = _pad_x(fx)
        in_y, out_y = _pad_y(fy)
        depth = depth + (in_x - out_x) + (in_y - out_y)

        # --- sources and sinks ---
        rain = 0.0
        for t0, t1, mmh in rain_steps:
            if t0 <= t < t1:
                rain = mmh / 3.6e6
                break
        if rain > 0:
            depth = depth + rain * dt * rain_weight
        depth = depth - torch.minimum(depth, infil * dt)

        dpos = torch.clamp(depth, min=0.0)
        q_weir = DRAIN_WEIR_C * (GRATE_PERIM_M * gully) * dpos ** 1.5
        # sqrt(0) has an infinite derivative, and `minimum` below routes ZERO
        # gradient to whichever branch it did not select - so on a dry cell the
        # orifice term contributes 0 * inf = NaN to the tape. The offset keeps
        # the derivative finite; it shifts the discharge by <1e-6 m3/s, far
        # below any depth the solver resolves.
        q_orif = DRAIN_ORIFICE_C * (GRATE_AREA_M2 * gully) * torch.sqrt(2.0 * G * dpos + 1e-12)
        q_in = torch.minimum(q_weir, q_orif)
        depth = depth - torch.minimum(depth, q_in * dt / (res * res))

        return torch.clamp(depth * keep, min=0.0), qx, qy

    if chunk and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        i = 0
        while i < n_steps:
            m = min(chunk, n_steps - i)
            t0 = i * dt

            def run(depth, qx, qy, m=m, t0=t0):
                for j in range(m):
                    depth, qx, qy = step(depth, qx, qy, t0 + j * dt)
                return depth, qx, qy

            depth, qx, qy = checkpoint(run, depth, qx, qy, use_reentrant=False)
            i += m
    else:
        for i in range(n_steps):
            depth, qx, qy = step(depth, qx, qy, i * dt)
    return depth


def flooded_area_objective(depth, street, thresh=0.05, eps=0.01, res=1.0):
    """Smooth surrogate for "flooded street area".

    A hard count of cells over a depth threshold has zero gradient almost
    everywhere, so it cannot drive an optimiser. The sigmoid is that indicator
    with a finite width: `eps` is how many metres of depth it takes to go from
    'dry' to 'flooded', and as eps -> 0 this converges to the hard count."""
    ind = torch.sigmoid((depth - thresh) / eps)
    return (ind * street).sum() * res * res


def gradcheck(device="cpu", n=24, n_steps=120, verbose=True):
    """Verify dJ/drho against central finite differences of the same forward.

    This is the load-bearing test for the whole approach: an adjoint that is
    subtly wrong still produces plausible-looking designs, because the
    optimiser will happily descend a wrong gradient to a wrong optimum."""
    torch.manual_seed(0)
    dt, res = 0.02, 1.0
    yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float64),
                            torch.arange(n, dtype=torch.float64), indexing="ij")
    dem = (0.05 * yy + 0.01 * xx).to(device)          # a tilted street
    keep = torch.ones_like(dem)
    keep[0, :] = keep[-1, :] = keep[:, 0] = keep[:, -1] = 0.0
    street = torch.ones_like(dem)
    rain = [(0.0, n_steps * dt * 0.7, 120.0)]

    rho = torch.full((n, n), 0.4, dtype=torch.float64, device=device, requires_grad=True)
    gully = torch.full((n, n), 0.05, dtype=torch.float64, device=device, requires_grad=True)

    def J(rho_, gully_):
        d = forward(dem, density_filter(rho_, 1), gully_, rain, n_steps, dt, res,
                    keep=keep)
        return flooded_area_objective(d, street, res=res)

    j = J(rho, gully)
    j.backward()
    g_rho, g_gully = rho.grad.clone(), gully.grad.clone()

    # Sweep the step size rather than trusting one. A central difference has
    # truncation error O(eps^2) and round-off error O(machine_eps * |J| / eps),
    # so the agreement is V-shaped in eps and the floor is set by how big the
    # gradient is relative to J. The gully gradients here are ~1e-7 against a
    # J of order 100, so a step that suits rho is pure noise for gully; taking
    # the best over a sweep is what actually tests the adjoint.
    worst = 0.0
    rows = []
    for name, var, grad in (("rho", rho, g_rho), ("gully", gully, g_gully)):
        for (r, c) in [(n // 2, n // 2), (n // 3, 2 * n // 3), (2 * n // 3, n // 3)]:
            ad = float(grad[r, c])
            best_rel, best_fd, best_eps = float("inf"), float("nan"), None
            for eps in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
                with torch.no_grad():
                    var[r, c] += eps
                jp = float(J(rho.detach(), gully.detach()))
                with torch.no_grad():
                    var[r, c] -= 2 * eps
                jm = float(J(rho.detach(), gully.detach()))
                with torch.no_grad():
                    var[r, c] += eps
                fd = (jp - jm) / (2 * eps)
                rel = abs(fd - ad) / max(abs(fd), abs(ad), 1e-30)
                if rel < best_rel:
                    best_rel, best_fd, best_eps = rel, fd, eps
            worst = max(worst, best_rel)
            rows.append((name, r, c, ad, best_fd, best_rel, best_eps))
    if verbose:
        print(f"  {'field':6s} {'cell':>9s} {'autograd':>13s} {'finite-diff':>13s} "
              f"{'rel err':>9s} {'best eps':>9s}")
        for name, r, c, ad, fd, rel, eps in rows:
            print(f"  {name:6s} ({r:3d},{c:3d}) {ad:13.6e} {fd:13.6e} {rel:9.2e} {eps:9.0e}")
        print(f"  worst relative error: {worst:.2e}")
    return worst


def checkpoint_equivalence(device="cpu", n=24, n_steps=120):
    """Checkpointing must not change the answer, only the memory it costs.

    Worth testing rather than assuming: the recomputed forward inside a
    checkpoint has to see the same rain schedule and the same dt as the
    original, and an off-by-one in the chunk's start time would give plausible
    but subtly wrong gradients."""
    torch.manual_seed(0)
    dt, res = 0.02, 1.0
    yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float64),
                            torch.arange(n, dtype=torch.float64), indexing="ij")
    dem = (0.05 * yy + 0.01 * xx).to(device)
    keep = torch.ones_like(dem)
    keep[0, :] = keep[-1, :] = keep[:, 0] = keep[:, -1] = 0.0
    street = torch.ones_like(dem)
    rain = [(0.0, n_steps * dt * 0.7, 120.0)]

    grads = {}
    for chunk in (0, 10):
        rho = torch.full((n, n), 0.4, dtype=torch.float64, device=device,
                         requires_grad=True)
        gully = torch.full((n, n), 0.05, dtype=torch.float64, device=device,
                           requires_grad=True)
        d = forward(dem, density_filter(rho, 1), gully, rain, n_steps, dt, res,
                    keep=keep, chunk=chunk)
        flooded_area_objective(d, street, res=res).backward()
        grads[chunk] = (rho.grad.clone(), gully.grad.clone())
    dr = float((grads[0][0] - grads[10][0]).abs().max())
    dg = float((grads[0][1] - grads[10][1]).abs().max())
    print(f"  max |d(grad rho)|   plain vs checkpointed: {dr:.3e}")
    print(f"  max |d(grad gully)| plain vs checkpointed: {dg:.3e}")
    return max(dr, dg)


if __name__ == "__main__":
    print("gradient check: autograd vs central finite differences")
    worst = gradcheck()
    print("\ncheckpointing equivalence")
    dmax = checkpoint_equivalence()
    ok = worst < 1e-4 and dmax < 1e-12
    print("\nPASS" if ok else "\nFAIL")
