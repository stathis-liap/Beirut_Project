#!/usr/bin/env python3
"""Extra figures for the two audience-specific decks.

make_presentation_figs.py builds the shared result figures. This adds the two
things those did not need: the solver internals a technical audience asks about
(equations, the dam-break verification, the infiltration and gully laws, the
attribution and the optimiser), and the plain-language input/output picture a
non-technical audience needs instead.

  python scripts/make_deck_figs.py
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LightSource, ListedColormap, BoundaryNorm

sys.path.insert(0, os.path.dirname(__file__))
from build_deck import corridor_masks, STORMS

# Same palette as the shared figures so the two decks read as one project.
C_BEFORE, C_AFTER, C_ACCENT, C_WARN = "#8C7B6B", "#0E7C86", "#3FB3BD", "#A03E3E"
NAVY, GREY, PAPER = "#0E2841", "#5C6B73", "#F2F4F3"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "mathtext.fontset": "cm",
})

G = 9.81


# ---------------------------------------------------------------- equations --
def eq_card(path, title, lines, w=6.4, lh=0.62, size=15):
    """A titled block of display equations, rendered as an image.

    python-pptx has no equation support, so the governing equations go on the
    slides as figures. mathtext with the Computer Modern set keeps them looking
    like the paper they come from rather than like slide furniture."""
    h = 0.62 + lh * len(lines)
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_facecolor(PAPER)
    fig.text(0.035, 1 - 0.34 / h, title, fontsize=11, color=TEAL_T, weight="bold",
             va="top")
    for i, (eq, note) in enumerate(lines):
        y = 1 - (0.62 + lh * (i + 0.55)) / h
        fig.text(0.055, y, eq, fontsize=size, color=NAVY, va="center")
        if note:
            fig.text(0.97, y, note, fontsize=8.5, color=GREY, va="center",
                     ha="right", style="italic")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.16,
                facecolor=PAPER)
    plt.close(fig)


TEAL_T = "#156082"


def figs_equations(out):
    """Equation cards.

    The annotation on the right of each line says what the term *does*, not
    what it is called. The audience for these slides builds software and
    understands maths; it does not read the hydraulics literature, so a name
    like "Audusse reconstruction" tells them nothing and "keeps still water
    still over a stepped bed" tells them everything."""
    eq_card(f"{out}/eq_swe.png", "The physics: three quantities per cell", [
        (r"$\partial_t h + \partial_x(hu) + \partial_y(hv) = R - I - D$",
         "depth changes by flow in/out, plus rain,\nminus what soaks in and what drains take"),
        (r"$\partial_t(hu) + \partial_x\left(hu^2 + \frac{1}{2}gh^2\right)"
         r" + \partial_y(huv) = -gh \partial_x z_b - \tau_x$",
         "east-west momentum: pushed by its own\npressure and by the downhill slope, held\nback by friction"),
        (r"$\partial_t(hv) + \partial_x(huv) + \partial_y\left(hv^2 + "
         r"\frac{1}{2}gh^2\right) = -gh \partial_y z_b - \tau_y$",
         "the same, north-south"),
        (r"$\tau = g n^2 |\mathbf{u}| \mathbf{u} / h^{1/3}$",
         "friction. $n$ is Manning's roughness —\nthe one number per surface type"),
    ], w=8.4, lh=0.86, size=14)

    eq_card(f"{out}/eq_numerics.png", "How it is computed: a loop over cells", [
        (r"$U_i^{n+1} = U_i^{n} - \frac{\Delta t}{\Delta x}"
         r"\left(F_{i+\frac{1}{2}} - F_{i-\frac{1}{2}}\right) + \Delta t S_i$",
         "new state = old state − (what left − what\narrived) + rain, infiltration, drains"),
        (r"$F^{hllc} = \frac{S_R F_L - S_L F_R + S_L S_R (U_R - U_L)}{S_R - S_L}$",
         "flow across one face, worked out from the\ncells either side — handles a sharp front\nwithout smearing it"),
        (r"$S_L, S_R \sim u \pm \sqrt{gh}$",
         "how fast a disturbance travels. Sets both\nthe flux above and the timestep below"),
        (r"$\tilde h_{L} = \max\left(0, h_L + z_L - \max(z_L, z_R)\right)$",
         "measure depth from the higher of the two\nbeds, so still water stays still over a step"),
        (r"$\Delta t \leq 0.4 \Delta x / \max\left(|u| + \sqrt{gh}\right)$",
         "timestep short enough that water cannot\ncross a cell in one step, or it goes unstable"),
    ], w=8.4, lh=0.92, size=14)

    eq_card(f"{out}/eq_infil.png", "Infiltration: the ground fills up", [
        (r"$f(t) = K\left(1 + \frac{\psi \Delta\theta}{F(t)}\right)$",
         "soak-up rate. $K$ is the steady rate; the\nsecond term is the extra suction of dry soil,\nand it shrinks as total soaked-in $F$ grows"),
        (r"$F - F_n - \psi\Delta\theta \ln\frac{F + \psi\Delta\theta}"
         r"{F_n + \psi\Delta\theta} = K \Delta t$",
         "the same law as a total rather than a rate.\nThis is what we actually step forward"),
        (r"$F^{(k+1)} = F^{(k)} - g(F^{(k)}) / g'(F^{(k)})$",
         "you cannot rearrange that for $F$, so:\nNewton's method, 4 iterations, every cell,\nevery step"),
    ], w=8.4, lh=0.94, size=14)

    eq_card(f"{out}/eq_drain.png", "Gullies: how much a grate can swallow", [
        (r"$Q_{weir} = C_w P h^{3/2}$",
         "shallow water spilling over the grate edge.\n$P$ is the perimeter"),
        (r"$Q_{orifice} = C_o A \sqrt{2gh}$",
         "deep water squeezing through the opening.\n$A$ is the hole"),
        (r"$Q = \min\left(Q_{weir}, Q_{orifice}, h \Delta x^2/\Delta t\right)$",
         "whichever is the bottleneck — and never\nmore water than the cell actually has"),
    ], w=8.4, lh=0.94, size=14)


def fig_compute_loop(out):
    """What one timestep actually does, for people who write code."""
    fig, ax = plt.subplots(figsize=(11.6, 4.5))
    ax.set_xlim(0, 116); ax.set_ylim(0, 45); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, fc, ec, lw=1.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0,rounding_size=1.8",
                                    fc=fc, ec=ec, lw=lw, zorder=2))

    def arrow(x0, y0, x1, y1, col=C_ACCENT, rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=15, lw=1.6, color=col,
                                     connectionstyle=f"arc3,rad={rad}", zorder=3))

    # --- the cell stencil -------------------------------------------------
    cx, cy, sz = 12, 16, 8.5
    for dx, dy, fc in ((0, 0, "#CFE9EC"), (-1, 0, "white"), (1, 0, "white"),
                       (0, 1, "white"), (0, -1, "white")):
        box(cx + dx * sz, cy + dy * sz, sz, sz, fc, "#9FB3B8", lw=1.0)
    ax.text(cx + sz / 2, cy + sz / 2, "$h$\n$hu, hv$", fontsize=10, color=NAVY,
            ha="center", va="center", linespacing=1.5)
    for dx, dy in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        x0 = cx + sz / 2 + dx * sz * 0.42
        y0 = cy + sz / 2 + dy * sz * 0.42
        arrow(x0 + dx * sz * 0.26, y0 + dy * sz * 0.26, x0, y0, C_AFTER)
    ax.text(cx + sz / 2, cy - sz - 3.2,
            "Each cell holds three numbers.\nEach step, compute the flow across\nits four faces.",
            fontsize=9.5, color=NAVY, ha="center", va="top", linespacing=1.6)
    ax.text(cx + sz / 2, 43, "ONE CELL", fontsize=10, color=C_ACCENT,
            weight="bold", ha="center")

    # --- the step ---------------------------------------------------------
    steps = [("1.  Face fluxes", "how much water and momentum crosses each\nface, from the two cells either side"),
             ("2.  Update the cell", "add what arrived, subtract what left"),
             ("3.  Sources", "add rain; subtract infiltration (Newton solve)\nand whatever the gully swallows"),
             ("4.  Pick the next $\\Delta t$", "short enough that nothing crosses a whole cell")]
    for i, (t, sub) in enumerate(steps):
        y = 34 - i * 8.6
        box(40, y, 43, 7.2, "white", C_ACCENT)
        ax.text(42, y + 4.9, t, fontsize=10.5, color=NAVY, weight="bold",
                va="center")
        ax.text(42, y + 2.1, sub, fontsize=8.4, color=GREY, va="center",
                linespacing=1.35)
        if i < 3:
            arrow(61.5, y, 61.5, y - 1.4)
    ax.text(61.5, 43, "ONE TIMESTEP", fontsize=10, color=C_ACCENT,
            weight="bold", ha="center")
    ax.add_patch(FancyArrowPatch((84.5, 7.0), (84.5, 37.0), arrowstyle="-|>",
                                 mutation_scale=15, lw=1.6, color=C_AFTER,
                                 connectionstyle="arc3,rad=0.55", zorder=3))
    ax.text(93.5, 22, "repeat", fontsize=10, color=C_AFTER, weight="bold",
            ha="center")

    box(98, 12, 18, 20, PAPER, NAVY)
    ax.text(107, 29, "PER RUN", fontsize=9.5, color=NAVY, weight="bold",
            ha="center")
    ax.text(107, 21.5,
            "2.3 M cells\n~75,000 steps\n1 hour of storm\n\n"
            "all of it as tensor\nops on the GPU",
            fontsize=9.2, color=NAVY, ha="center", va="center", linespacing=1.6)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{out}/tech_loop.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------- verification --
def fig_dambreak(out, path="output/verification/dambreak.json"):
    """Ritter dam break: where the two schemes visibly part company."""
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    x = np.array(d["x"]); ex = np.array(d["exact"])
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9),
                           gridspec_kw=dict(width_ratios=[1.45, 1]))

    ax[0].fill_between(x, 0, ex, color=NAVY, alpha=0.10, lw=0)
    ax[0].plot(x, ex, color=NAVY, lw=2.0, label="the exact answer")
    ax[0].plot(x, np.array(d["schemes"]["hllc"]["h"]), color=C_AFTER, lw=1.5,
               label="the full method")
    ax[0].plot(x, np.array(d["schemes"]["inertial"]["h"]), color=C_BEFORE,
               lw=1.5, ls="--", label="the cheap method")
    ax[0].set_xlabel("distance from the dam (m)")
    ax[0].set_ylabel("water depth (m)")
    ax[0].set_title(f"Dry-bed dam break, {d['t_end']:.0f} s after release "
                    f"({d['profile_res']} m grid)")
    ax[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    ax[0].set_xlim(x.min(), x.max())

    rows = sorted(d["convergence"], key=lambda r: -r["res_m"])
    r = [q["res_m"] for q in rows]
    for key, lab, col, ls in (("hllc", "the full method", C_AFTER, "-"),
                              ("inertial", "the cheap method", C_BEFORE, "--")):
        ax[1].plot(r, [q[key]["rmse_pct_h0"] for q in rows], ls, color=col,
                   marker="o", ms=4, label=lab)
    ax[1].set_xscale("log"); ax[1].set_xticks(r)
    ax[1].set_xticklabels([f"{v:g}" for v in r])
    ax[1].set_xticks([], minor=True)          # log minor labels collide with ours
    ax[1].invert_xaxis()
    ax[1].set_xlabel("grid resolution (m)  →  finer")
    ax[1].set_ylabel("depth error, % of initial depth")
    ax[1].set_title("Halve the grid, halve the error — for one of them")
    ax[1].legend(frameon=False, fontsize=8.5)
    ax[1].set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(f"{out}/tech_dambreak.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ process models --
def fig_greenampt(out):
    """Why a wetting front is not the same as a constant rate."""
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.7))
    t = np.linspace(1.0, 3600, 4000)                       # s

    soils = [("sandy loam  (grass, shrub, canopy)", 11.0, 0.11, 0.28, C_AFTER),
             ("silt loam  (bare soil)", 8.0, 0.12, 0.30, C_ACCENT),
             ("loamy sand  (gravel, unpaved)", 15.0, 0.08, 0.25, NAVY)]
    for lab, K_mmh, psi, dth, col in soils:
        K = K_mmh / 3.6e6                                  # m/s
        b = psi * dth
        # invert the cumulative form F - b ln(1 + F/b) = K t by bisection
        F = np.zeros_like(t)
        lo, hi = np.zeros_like(t), np.full_like(t, 10.0)
        for _ in range(80):
            F = 0.5 * (lo + hi)
            g = F - b * np.log1p(F / b) - K * t
            hi = np.where(g > 0, F, hi); lo = np.where(g > 0, lo, F)
        f = K * (1 + b / np.maximum(F, 1e-9)) * 3.6e6      # mm/h
        ax[0].plot(t / 60, f, color=col, lw=1.7, label=lab)
        ax[0].axhline(K_mmh, color=col, lw=0.9, ls=":", alpha=0.85)

    ax[0].set_xlabel("time into the storm (min)")
    ax[0].set_ylabel("infiltration capacity (mm/h)")
    ax[0].set_title("Capacity falls as the ground wets\n"
                    "dotted = the saturated rate it decays toward")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_xlim(0, 60); ax[0].set_ylim(0, 70)

    # Verification: the solver's 4-iteration implicit step against the exact
    # cumulative solution. The whole discrepancy is a fixed sub-millimetre
    # offset picked up in the first seconds, where the analytic rate is
    # unbounded; it does not grow, so it washes out as the storm proceeds.
    K, b = 11.0 / 3.6e6, 0.11 * 0.28

    def exact_F(tt):
        lo, hi = 0.0, 10.0
        for _ in range(90):
            mid = 0.5 * (lo + hi)
            if mid - b * np.log1p(mid / b) - K * tt > 0:
                hi = mid
            else:
                lo = mid
        return hi

    for dt, col in ((1.0, C_AFTER), (10.0, C_ACCENT), (60.0, C_WARN)):
        n = int(3600 / dt)
        Fn = 0.0; ts = []; err = []; last = 0.0
        for k in range(1, n + 1):
            f_new = Fn + K * dt
            for _ in range(4):                              # the solver's Newton
                gg = f_new - Fn - b * np.log((f_new + b) / (Fn + b)) - K * dt
                gp = 1.0 - b / (f_new + b)
                f_new = max(f_new - gg / max(gp, 1e-6), Fn)
            Fn = f_new
            tt = k * dt
            if tt - last >= 15.0:
                last = tt
                e = exact_F(tt)
                ts.append(tt / 60); err.append(100 * abs(Fn - e) / e)
        ax[1].plot(ts, err, color=col, lw=1.6, label=f"$\\Delta t$ = {dt:g} s")
        if dt == 1.0:
            offs = abs(Fn - exact_F(3600.0)) * 1000
    ax[1].set_yscale("log")
    ax[1].axhline(1.0, color=GREY, lw=0.9, ls="--")
    ax[1].text(58, 1.15, "1 %", fontsize=8, color=GREY, ha="right")
    ax[1].set_xlabel("time into the storm (min)")
    ax[1].set_ylabel("error in cumulative infiltration (%)")
    ax[1].set_title("Verified against the exact solution\n"
                    f"the entire discrepancy is a fixed {offs:.1f} mm, out of "
                    f"{exact_F(3600.0)*1000:.0f} mm infiltrated")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_xlim(0, 60)
    fig.tight_layout()
    fig.savefig(f"{out}/tech_greenampt.png", dpi=150)
    plt.close(fig)


def fig_gully(out):
    """Head-discharge capture versus the fixed-capacity assumption."""
    P, A, Cw, Co = 1.8, 0.09, 1.66, 0.6
    h = np.linspace(0, 0.40, 400)
    qw = Cw * P * h ** 1.5
    qo = Co * A * np.sqrt(2 * G * h)
    q = np.minimum(qw, qo)
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.plot(h * 100, qw * 1000, color=C_ACCENT, lw=1.2, ls="--",
            label=r"weir  $C_w P h^{3/2}$")
    ax.plot(h * 100, qo * 1000, color=NAVY, lw=1.2, ls=":",
            label=r"orifice  $C_o A \sqrt{2gh}$")
    ax.plot(h * 100, q * 1000, color=C_AFTER, lw=2.4, label="capture (the smaller)")
    ax.axhline(30, color=C_BEFORE, lw=1.4, label="fixed-capacity assumption")
    cross = h[np.argmin(np.abs(qw - qo))] * 100
    ax.axvline(cross, color=GREY, lw=0.8, alpha=0.6)
    ax.annotate(f"grate drowns\nat {cross:.0f} cm of head", xy=(cross, 8),
                xytext=(cross + 4, 6), fontsize=8, color=GREY)
    ax.set_xlabel("ponded depth over the grate (cm)")
    ax.set_ylabel("capture per inlet (L/s)")
    ax.set_title("An inlet captures what the head over it allows")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(0, 40); ax.set_ylim(0, 60)
    fig.tight_layout()
    fig.savefig(f"{out}/tech_gully.png", dpi=150)
    plt.close(fig)


# -------------------------------------------------------------- attribution --
def _walk_reduction(run, storm, mask, thr=0.10):
    b = f"{run}/before_{storm}/max_depth.npy"
    a = f"{run}/after_{storm}/max_depth.npy"
    if not (os.path.exists(b) and os.path.exists(a)):
        return None
    B, A = np.load(b), np.load(a)
    fb = ((B > thr) & mask).sum(); fa = ((A > thr) & mask).sum()
    return 100.0 * (fb - fa) / fb if fb else float("nan")


def fig_attribution(out, base, hllc, best, material):
    """Which model change moved the headline, and by how much."""
    _, walk = corridor_masks(material)
    cfg = [("baseline\ninertial + constant rate", base, C_BEFORE),
           ("+ shock capturing", hllc, C_ACCENT),
           ("+ Green–Ampt\n(best validated)", best, C_AFTER)]
    vals = [[_walk_reduction(run, k, walk) for k, _ in STORMS] for _, run, _ in cfg]
    if any(v is None for row in vals for v in row):
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    xs = np.arange(len(STORMS)); w = 0.26
    for i, ((lab, _, col), row) in enumerate(zip(cfg, vals)):
        bars = ax.bar(xs + (i - 1) * w, row, w, color=col, label=lab)
        for b_, v in zip(bars, row):
            ax.text(b_.get_x() + b_.get_width() / 2,
                    v + (1.2 if v >= 0 else -3.0), f"{v:.0f}",
                    ha="center", fontsize=8.5, color=NAVY)
    ax.axhline(0, color=NAVY, lw=0.9)
    ax.set_xticks(xs); ax.set_xticklabels([lab for _, lab in STORMS])
    ax.set_ylabel("reduction in flooded walkable area (%)")
    ax.set_title("Which model change moved the answer\n"
                 "Numerics dominate in the severe storm; soil physics in the light ones")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{out}/tech_attribution.png", dpi=150)
    plt.close(fig)


def fig_optimizer(out, path="output/design_opt/result.json"):
    """Gradient-based placement against the heuristics it has to beat."""
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    fa = d["flooded_m2"]
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.8),
                           gridspec_kw=dict(width_ratios=[1, 1.15]))

    keys = ["do_nothing", "uniform", "greedy", "gradient"]
    labs = ["do nothing", "spread evenly", "greedy\n(lowest cells first)",
            "gradient\ndescent"]
    cols = [GREY, C_BEFORE, C_ACCENT, C_AFTER]
    bars = ax[0].bar(labs, [fa[k] for k in keys], color=cols, width=0.62)
    for b_, k in zip(bars, keys):
        ax[0].text(b_.get_x() + b_.get_width() / 2, fa[k] + 4,
                   f"{fa[k]:.0f}", ha="center", fontsize=9, color=NAVY)
    ax[0].set_ylabel("flooded area after the storm (m²)")
    ax[0].set_title(f"Same budget of permeable surface ({d['budget_frac']:.0%} of the patch)")
    ax[0].tick_params(axis="x", labelsize=8)

    hist = d["history"]
    ax[1].plot(np.arange(1, len(hist) + 1), hist, color=C_AFTER, lw=1.8)
    ax[1].axhline(fa["greedy"], color=C_ACCENT, lw=1.1, ls="--",
                  label="greedy (lowest cells first)")
    ax[1].axhline(fa["uniform"], color=C_BEFORE, lw=1.1, ls=":", label="spread evenly")
    ax[1].set_xlabel("optimiser iteration")
    ax[1].set_ylabel("flooded area (m²)")
    ax[1].set_title("Every step needs the gradient of flooding\n"
                    "with respect to all 16,384 design cells")
    ax[1].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{out}/tech_optimizer.png", dpi=150)
    plt.close(fig)


def fig_massbalance(out, runs):
    """The audit every run carries: nothing is created or lost."""
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.7),
                           gridspec_kw=dict(width_ratios=[1.15, 1]))
    metas = []
    for key, lab in STORMS:
        for phase, style in (("before", "--"), ("after", "-")):
            p = f"{runs}/{phase}_{key}/run_meta.json"
            if os.path.exists(p):
                metas.append((lab, phase, style, json.load(open(p))))

    cols = {lab: c for (_, lab), c in zip(STORMS, [C_ACCENT, C_AFTER, NAVY])}
    for lab, phase, style, m in metas:
        s = np.array(m["storage_series_m3"])
        ax[0].plot(s[:, 0] / 60, s[:, 1] / 1000, style, lw=1.5,
                   color=cols[lab], alpha=1.0 if phase == "after" else 0.55,
                   label=f"{lab} — {phase}")
    ax[0].set_xlabel("time (min)")
    ax[0].set_ylabel("water standing on the surface (thousand m³)")
    ax[0].set_title("Storage through the storm")
    ax[0].legend(frameon=False, fontsize=7.2, ncol=2)

    short = {"frequent (T2)": "T2", "observed 25 Nov 2025": "25 Nov",
             "severe (T50)": "T50"}
    labs = [f"{short.get(lab, lab)}\n{phase}" for lab, phase, _, _ in metas]
    rel = [abs(m["closure_rel"]) for _, _, _, m in metas]
    ax[1].bar(range(len(rel)), np.maximum(rel, 1e-9), color=C_AFTER, width=0.6)
    ax[1].set_yscale("log")
    ax[1].axhline(1e-3, color=C_WARN, lw=1.2, ls="--",
                  label="1e-3 — a typical acceptance threshold")
    ax[1].set_xticks(range(len(labs)))
    ax[1].set_xticklabels(labs, fontsize=8, rotation=0)
    ax[1].set_ylabel("|rain − (infiltrated + drained + out + stored)| / rain")
    ax[1].set_title("Mass balance closes on every run")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_ylim(1e-9, 1e-2)
    fig.tight_layout()
    fig.savefig(f"{out}/tech_massbalance.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------- non-technical set --
def fig_blackbox(out):
    """What goes in, what the model is, what comes out."""
    fig, ax = plt.subplots(figsize=(12.2, 5.0))
    ax.set_xlim(0, 122); ax.set_ylim(4.5, 50.5); ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, fc, ec, r=2.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=1.4, zorder=2))

    def arrow(x0, y0, x1, y1, col=C_ACCENT):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=16, lw=1.8, color=col, zorder=3))

    inputs = [("Ground surface", "39 GB airborne laser scan\nevery half metre of the city"),
              ("What the ground is made of", "asphalt, concrete, soil, grass,\nbuildings, trees — 9 types"),
              ("Rain", "how hard, for how long\n(a real storm or a design storm)"),
              ("The proposed design", "where the corridor goes and\nwhat each surface is made of")]
    for i, (t, s) in enumerate(inputs):
        y = 39 - i * 10.2
        box(1.5, y, 30, 8.4, "white", C_ACCENT)
        ax.text(3.2, y + 5.6, t, fontsize=10.5, color=NAVY, weight="bold", va="center")
        ax.text(3.2, y + 2.5, s, fontsize=8.2, color=GREY, va="center", linespacing=1.35)
        arrow(31.8, y + 4.2, 43.5, y + 4.2)

    box(44, 6, 33, 38, PAPER, NAVY, r=3)
    ax.text(60.5, 40, "THE MODEL", fontsize=11, color=NAVY, weight="bold", ha="center")
    ax.text(60.5, 34.5,
            "Rain falls on every half-metre\ncell of the city, and the water is\n"
            "moved downhill by the physics\nof shallow flowing water.",
            fontsize=9.2, color=NAVY, ha="center", va="center", linespacing=1.5)
    ax.text(60.5, 24.5,
            "Every cell separately soaks up\nwater, slows it down, ponds it,\n"
            "or drops it into a gully —\naccording to what it is made of.",
            fontsize=9.2, color=NAVY, ha="center", va="center", linespacing=1.5)
    ax.text(60.5, 13.5,
            "Nothing is decided in advance.\nWhere the flooding appears is\n"
            "an answer, not an assumption.",
            fontsize=9.2, color=C_AFTER, ha="center", va="center", linespacing=1.5,
            style="italic")

    outputs = [("Where it floods, and how deep", "a map of the whole area,\nstreet by street"),
               ("How dangerous it is", "depth and speed together —\nwhere you could be swept off your feet"),
               ("Where the water ends up", "how much soaks in, how much\nreaches the sea"),
               ("What the design changes", "the same maps with and without\nthe corridor, subtracted")]
    for i, (t, s) in enumerate(outputs):
        y = 39 - i * 10.2
        box(90.5, y, 30, 8.4, "white", C_AFTER)
        ax.text(92.2, y + 5.6, t, fontsize=10.5, color=NAVY, weight="bold", va="center")
        ax.text(92.2, y + 2.5, s, fontsize=8.2, color=GREY, va="center", linespacing=1.35)
        arrow(77.3, y + 4.2, 89, y + 4.2, C_AFTER)

    ax.text(16.5, 48.4, "WHAT GOES IN", fontsize=10.5, color=C_ACCENT,
            weight="bold", ha="center")
    ax.text(105.5, 48.4, "WHAT COMES OUT", fontsize=10.5, color=C_AFTER,
            weight="bold", ha="center")
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{out}/nt_blackbox.png", dpi=150)
    plt.close(fig)


def fig_inputs(out, terrain, material):
    """The four inputs, as the actual data rather than as words."""
    dem = np.load(f"{terrain}/dem.npy")
    lc = np.load(f"{terrain}/landcover.npy")
    m = np.load(f"{terrain}/masks.npz")
    valid = m["valid"]
    mat = np.load(material)

    fig, ax = plt.subplots(1, 4, figsize=(11.2, 5.0),
                           gridspec_kw=dict(width_ratios=[1, 1, 1.05, 1],
                                            wspace=0.26))
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)

    z = np.where(valid, dem, np.nan)
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(np.nan_to_num(z, nan=np.nanmedian(z)), vert_exag=2.0)
    ax[0].imshow(hs, cmap="gray")
    ax[0].imshow(z, cmap="terrain", alpha=0.45)
    ax[0].set_title("1. The ground\nlaser-scanned, half-metre detail", fontsize=9.5)

    names = ["paved", "building", "trees", "grass", "soil", "water",
             "gravel", "shrub", "concrete"]
    cols = ["#B9B4AC", "#7A6A5D", "#2E6B43", "#79B85B", "#B08A5A", "#3E7CB1",
            "#C9BA9B", "#5E9A56", "#D6D2CB"]
    ax[1].imshow(np.where(valid, lc, np.nan), cmap=ListedColormap(cols),
                 norm=BoundaryNorm(np.arange(-0.5, 9.5), 9), interpolation="nearest")
    ax[1].set_title("2. What it is made of\nsoaks up water, or sheds it", fontsize=9.5)
    ax[1].legend(handles=[plt.Line2D([], [], marker="s", ls="", ms=5, color=c, label=n)
                          for c, n in zip(cols, names)],
                 loc="upper center", bbox_to_anchor=(0.5, -0.01), fontsize=6.4,
                 frameon=False, ncol=3, handletextpad=0.35, columnspacing=0.9,
                 labelspacing=0.3)

    t = np.linspace(0, 60, 400)
    r = np.where(t < 7.5, 12.8, np.where(t < 22.5, 88.8, np.where(t < 30, 12.8, 0)))
    ax[2].set_xticks([0, 15, 30, 45, 60]); ax[2].set_yticks([0, 30, 60, 90])
    ax[2].fill_between(t, 0, r, color=C_ACCENT, alpha=0.75, step="pre")
    ax[2].set_xlabel("minutes", fontsize=8.5)
    ax[2].set_ylabel("rain (mm/h)", fontsize=8.5)
    ax[2].grid(alpha=0.25)
    ax[2].set_title("3. The storm\n25 Nov 2025: 25 mm in half an hour", fontsize=9.5)
    ax[2].set_box_aspect(1.75)

    grn = np.where(mat > 0, 1.0, np.nan)
    ax[3].imshow(hs, cmap="gray", alpha=0.75)
    ax[3].imshow(grn, cmap=ListedColormap([C_AFTER]), alpha=0.95,
                 interpolation="nearest")
    ax[3].set_title("4. The proposed corridor\nseven surface types, cell by cell",
                    fontsize=9.5)
    fig.suptitle("Everything the model is told. Nothing else.", fontsize=12,
                 color=NAVY, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{out}/nt_inputs.png", dpi=140)
    plt.close(fig)


def fig_headline(out, runs, material):
    """The three storms as three plain statements."""
    _, walk = corridor_masks(material)
    fig, ax = plt.subplots(figsize=(12.2, 4.4))
    ax.set_xlim(0, 120); ax.set_ylim(0, 44); ax.axis("off"); ax.grid(False)

    cards = []
    for key, lab in STORMS:
        w = _walk_reduction(runs, key, walk)
        mb = json.load(open(f"{runs}/before_{key}/run_meta.json"))
        ma = json.load(open(f"{runs}/after_{key}/run_meta.json"))
        cards.append((key, lab, w, ma["vol_infiltrated_m3"] - mb["vol_infiltrated_m3"],
                      100 * (mb["vol_outflow_m3"] - ma["vol_outflow_m3"])
                      / max(mb["vol_outflow_m3"], 1e-9)))

    blurb = {"t2": "the rain Beirut sees\nmost winters",
             "v1_nov2025": "the storm that flooded\nSassine and the Ring",
             "t50": "a once-in-50-years\ndownpour"}
    for i, (key, lab, w, inf, outp) in enumerate(cards):
        x = 2 + i * 39
        good = w > 5
        ax.add_patch(FancyBboxPatch((x, 2), 35, 40,
                                    boxstyle="round,pad=0,rounding_size=2.5",
                                    fc="white", ec=C_AFTER if good else C_BEFORE,
                                    lw=1.8))
        ax.text(x + 17.5, 37.5, lab, fontsize=11, color=NAVY, weight="bold",
                ha="center")
        ax.text(x + 17.5, 33.2, blurb.get(key, ""), fontsize=8.4, color=GREY,
                ha="center", va="center", linespacing=1.4)
        ax.text(x + 17.5, 23.5, f"{w:.0f}%" if good else "no change",
                fontsize=34 if good else 20,
                color=C_AFTER if good else C_BEFORE, ha="center", va="center",
                weight="bold")
        ax.text(x + 17.5, 16.5,
                "less flooding where\npeople walk" if good
                else "the corridor fills and holds\nthe water instead",
                fontsize=8.6, color=NAVY, ha="center", va="center", linespacing=1.4)
        ax.plot([x + 4, x + 31], [12.5, 12.5], color="#E4E8E7", lw=1.2)
        ax.text(x + 17.5, 8.6, f"+{inf:,.0f} m³ of rain soaked into the ground",
                fontsize=8.4, color=GREY, ha="center")
        ax.text(x + 17.5, 5.2, f"{outp:.0f}% less water sent toward the port",
                fontsize=8.4, color=GREY, ha="center")
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{out}/nt_headline.png", dpi=150)
    plt.close(fig)


def fig_depth_classes(out, runs, terrain, material, storm="v1_nov2025", pad=90):
    """Depth in terms a person can stand in, cropped to where the corridor is.

    The whole-domain view hides the effect: the corridor is a thin ribbon in a
    large catchment, so the two panels look identical at that scale. Cropping to
    the ribbon and its immediate streets is what a resident actually wants to
    see."""
    b = f"{runs}/before_{storm}/max_depth.npy"
    a = f"{runs}/after_{storm}/max_depth.npy"
    if not (os.path.exists(b) and os.path.exists(a)):
        return
    B, A = np.load(b), np.load(a)
    m = np.load(f"{terrain}/masks.npz")
    dem = np.load(f"{terrain}/dem.npy")
    ribbon, walk = corridor_masks(material)

    rr, cc = np.where(ribbon)
    r0, r1 = max(rr.min() - pad, 0), min(rr.max() + pad, B.shape[0])
    c0, c1 = max(cc.min() - pad, 0), min(cc.max() + pad, B.shape[1])
    sl = (slice(r0, r1), slice(c0, c1))

    show = (m["valid"] & ~m["building"])[sl]
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(np.nan_to_num(dem[sl], nan=float(np.nanmedian(dem))),
                      vert_exag=2.0)
    edge = ribbon[sl]

    edges = [0.02, 0.10, 0.30, 0.60, 99.0]
    labs = ["a puddle  (2–10 cm)", "over your shoe  (10–30 cm)",
            "shin to knee  (30–60 cm)", "waist or worse  (over 60 cm)"]
    cols = ["#BFE3E8", "#63BBC7", "#1F8A99", "#0B3C4A"]

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 7.4))
    for k, (X, ttl) in enumerate(((B, "Today"), (A, "With the green corridor"))):
        Xc = X[sl]
        cls = np.full(Xc.shape, np.nan)
        for i in range(4):
            cls[show & (Xc >= edges[i]) & (Xc < edges[i + 1])] = i
        ax[k].imshow(hs, cmap="gray", vmin=0, vmax=1.6)
        ax[k].imshow(cls, cmap=ListedColormap(cols),
                     norm=BoundaryNorm(np.arange(-0.5, 4.5), 4),
                     interpolation="nearest")
        ax[k].contour(edge.astype(float), levels=[0.5], colors=["#0E2841"],
                      linewidths=0.9)
        wet = float((walk & (X > 0.10)).sum()) * 0.25
        ax[k].set_title(f"{ttl}\n{wet:,.0f} m² of the walking surface\nunder water",
                        fontsize=10.5, color=NAVY)
        ax[k].set_xticks([]); ax[k].set_yticks([]); ax[k].grid(False)

    fig.legend(handles=[plt.Line2D([], [], marker="s", ls="", ms=10, color=c, label=l)
                        for c, l in zip(cols, labs)],
               loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.004))
    fig.suptitle("How deep the water gets — the observed storm of 25 November 2025",
                 fontsize=13, color=NAVY)
    fig.tight_layout(rect=(0, 0.075, 1, 0.955))
    fig.savefig(f"{out}/nt_depth_classes.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/presentation")
    ap.add_argument("--runs", default="output/corridor_runs_v3_best")
    ap.add_argument("--base", default="output/corridor_runs_v3")
    ap.add_argument("--hllc", default="output/corridor_runs_v3_hllc")
    ap.add_argument("--terrain", default="output/terrain_cut_0.5_v3")
    ap.add_argument("--material", default="output/corridor_gi_cut_v3/material.npy")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    jobs = {
        "equations": lambda: figs_equations(a.out),
        "loop": lambda: fig_compute_loop(a.out),
        "dambreak": lambda: fig_dambreak(a.out),
        "greenampt": lambda: fig_greenampt(a.out),
        "gully": lambda: fig_gully(a.out),
        "attribution": lambda: fig_attribution(a.out, a.base, a.hllc, a.runs,
                                               a.material),
        "optimizer": lambda: fig_optimizer(a.out),
        "massbalance": lambda: fig_massbalance(a.out, a.runs),
        "blackbox": lambda: fig_blackbox(a.out),
        "inputs": lambda: fig_inputs(a.out, a.terrain, a.material),
        "headline": lambda: fig_headline(a.out, a.runs, a.material),
        "depths": lambda: fig_depth_classes(a.out, a.runs, a.terrain, a.material),
    }
    want = a.only.split(",") if a.only else list(jobs)
    for name in want:
        jobs[name]()
        print(f"  {name}")
    print(f"figures -> {a.out}")


if __name__ == "__main__":
    main()
