#!/usr/bin/env python3
"""Two audience-specific decks from the same results.

The single deck in make_pptx.py tries to serve everyone and therefore pitches
at the middle. These split it: one for people who build simulators, with the
equations, the verification and the statistics; one for people who need to know
what goes in and what comes out.

  python scripts/make_decks.py

Both read every number from the run outputs, so neither can drift from the
results, and both use the Dynacity template's own layouts and fonts.
"""

import argparse
import json
import os
import sys

import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt

sys.path.insert(0, os.path.dirname(__file__))
from build_deck import corridor_masks, STORMS
from pptx.dml.color import RGBColor
from make_pptx import (strip_slides, drop_ph, fit, set_text, caption, title_of,
                       numbers, NAVY, TEAL, GREY,
                       L_INTRO, L_CHAPTER, L_CONTENT, L_TEXT_IMG, L_TWO_IMG,
                       L_BULLETS, L_OUTRO)

WHITE = RGBColor(0xFF, 0xFF, 0xFF)
L_TWO_TEXT = 4          # title + two text columns
L_COMPARE = 5           # title + two labelled columns

# Geometry of the layouts we place into, in inches.
BOX_IMG_RIGHT = (6.95, 1.31, 6.05, 5.08)      # picture half of L_TEXT_IMG
BOX_FULL = (0.41, 1.20, 12.60, 4.62)          # content area of L_CONTENT
BOX_FULL_TALL = (0.41, 1.05, 12.60, 4.85)
CAP_BOX = (0.55, 5.98, 12.20, 0.55)           # above the dark footer


def deck(template):
    prs = Presentation(template)
    strip_slides(prs)
    return prs


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def body(slide, idx, lines, size=14, space=10):
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == idx:
            set_text(ph, lines, size=size, space=space)
            return ph
    return None


def img_slide(prs, path, title, cap=None, box=BOX_FULL, layout=L_CONTENT):
    s = prs.slides.add_slide(prs.slide_layouts[layout])
    title_of(s, title)
    drop_ph(s, 1); drop_ph(s, 10)
    fit(prs, s, path, box)
    if cap:
        caption(s, cap, CAP_BOX, size=10.5)
    return s


def text_img(prs, title, lines, img, size=14, box=BOX_IMG_RIGHT):
    s = prs.slides.add_slide(prs.slide_layouts[L_TEXT_IMG])
    title_of(s, title)
    body(s, 1, lines, size=size)
    drop_ph(s, 10)
    fit(prs, s, img, box)
    return s


def two_image(prs, title, left, right, cap_left=None, cap_right=None,
              stacked=False):
    """Two figures on one slide.

    Built on the plain content layout rather than the template's two-image
    layout: that layout carries no title placeholder and its caption boxes are
    dark-filled, so text placed in them disappears."""
    s = prs.slides.add_slide(prs.slide_layouts[L_CONTENT])
    title_of(s, title)
    drop_ph(s, 1); drop_ph(s, 10)
    if stacked:
        fit(prs, s, left, (3.4, 1.10, 6.5, 1.95))
        fit(prs, s, right, (0.35, 3.20, 12.6, 3.05))
    else:
        fit(prs, s, left, (0.45, 1.15, 6.15, 4.30))
        fit(prs, s, right, (6.75, 1.15, 6.15, 4.30))
        for txt, x in ((cap_left, 0.55), (cap_right, 6.85)):
            if txt:
                caption(s, txt, (x, 5.62, 5.95, 0.95), size=10)
    return s


def chapter(prs, title):
    s = prs.slides.add_slide(prs.slide_layouts[L_CHAPTER])
    title_of(s, title)
    return s


# =========================================================== technical deck ==
def build_technical(prs, F, n, out):
    s = prs.slides.add_slide(prs.slide_layouts[L_INTRO])
    title_of(s, "A differentiable flood model")
    tb = s.shapes.add_textbox(Inches(0.92), Inches(4.62), Inches(7.6), Inches(1.2))
    set_text(tb, ["Rain-on-grid shallow water at 0.5 m, with infiltration, "
                  "detention and head-dependent drainage",
                  "Formulation · verification · benchmarks · Beirut"],
             size=13.5, color=WHITE)
    notes(s, "A 2-D shallow-water code built for one question: does a green "
             "corridor reduce street flooding, and by how much. Three claims "
             "today — it solves the equations correctly, it agrees with codes "
             "that were written independently, and it can differentiate itself.")

    chapter(prs, "FORMULATION")

    text_img(prs, "Governing equations and source terms", [
        [("Depth-averaged conservation of mass and momentum", True), ("", False)],
        "Rain R applied to every cell; infiltration I and inlet capture D are "
        "per-cell sinks — this is what makes it a pluvial model rather than a "
        "routed one",
        "Bed slope treated as a source term, friction by Manning",
        "",
        [("Domain:", True), ("  2,001 × 1,142 cells at 0.5 m = 2.3 M cells, "
                             "GPU, float32", False)],
        [("Wetting and drying:", True),
         ("  Kurganov–Petrova velocity desingularisation; a flux limiter "
          "rescales each cell's outflux so it never exports more than it holds",
          False)],
    ], f"{F}/eq_swe.png", size=13, box=(6.95, 1.9, 6.05, 3.6))
    notes(prs.slides[-1],
          "Standard SWE, deliberately. That is what lets us benchmark against "
          "everyone else. The three mass source terms are where this departs "
          "from a hydraulics-only code. The limiter is mass-conserving — it "
          "rescales outflux rather than capping velocity, so it does not "
          "silently suppress supercritical flow the way a depth-gradient clip "
          "does.")

    text_img(prs, "Two schemes in one code", [
        [("Inertial (three-term)", True),
         ("  — Bates, Horritt & Fewtrell (2010). Advective acceleration "
          "dropped; face discharge updated semi-implicitly against friction. "
          "The LISFLOOD-FP family.", False)],
        "",
        [("Shock-capturing (HLLC)", True),
         ("  — Godunov finite volume, HLLC approximate Riemann solver, Toro "
          "two-rarefaction wave-speed estimates, Audusse hydrostatic "
          "reconstruction for well-balancedness over an irregular bed.", False)],
        "",
        [("Cost:", True), ("  HLLC is ≈ 3.5× the inertial scheme, and needs "
                           "CFL ≤ 0.4 against 0.7 — unsplit 2-D Godunov is "
                           "unstable at the inertial number.", False)],
        [("Why it matters:", True), ("  the schemes share terrain, sources and "
                                     "mass accounting, so scheme error is "
                                     "measured rather than assumed.", False)],
    ], f"{F}/eq_numerics.png", size=12, box=(6.95, 1.5, 6.05, 4.6))
    notes(prs.slides[-1],
          "Having both in one code is the unusual part. Every difference "
          "between the two runs is the scheme and nothing else — same terrain, "
          "same rain, same infiltration, same accounting. That is what makes "
          "the attribution slide later possible.")

    img_slide(prs, f"{F}/tech_dambreak.png",
              "Verification: dry-bed dam break against Ritter",
              "Left: 15 s after release on a 0.5 m grid, frictionless, closed "
              "tank. Right: the same test refined from 2 m to 0.25 m. HLLC "
              "converges at roughly first order (5.5 → 1.9 % of h₀); the "
              "inertial scheme plateaus near 14 % and its front is 60 % too "
              "slow, because the neglected advective term is exactly the "
              "momentum flux that sets the front celerity.",
              box=BOX_FULL_TALL)
    notes(prs.slides[-1],
          "The cleanest separation we have. Note the inertial scheme does not "
          "just lose accuracy — it converges to the wrong answer, and it "
          "oscillates. This is the mechanism behind its +27 % wet-area bias on "
          "Beirut: streets there run supercritical at 6–8 m/s on a 6 % grade, "
          "which is a dam break in slow motion.")

    s = two_image(prs, "Infiltration: a wetting front, solved implicitly",
                  f"{F}/eq_infil.png", f"{F}/tech_greenampt.png",
                  stacked=True)
    notes(s, "Green–Ampt rather than a constant rate. Two reasons: capacity "
             "declines as the ground wets, which a fixed rate cannot express, "
             "and it is what dropped the corridor's infiltration multiple from "
             "3× to 1.3×. The explicit rate is unbounded at F = 0, so the "
             "cumulative form is solved implicitly — four Newton iterations "
             "per cell per step. Impervious cells have ψΔθ = 0, which is a "
             "0/0 in the log; they are branched around, not regularised.")

    s = prs.slides.add_slide(prs.slide_layouts[L_TEXT_IMG])
    title_of(s, "Gully inlets that respond to the head over them")
    body(s, 1, [
        "Weir control while the grate is free-flowing, orifice control once "
        "drowned; capture is the smaller of the two, and never more water than "
        "the cell holds",
        "A 0.6 × 0.3 m grate: perimeter 1.8 m, clear opening 0.09 m² after bars "
        "and blockage",
        "",
        [("Why not a fixed capacity:", True),
         ("  capture then depends on the ponding the design actually produces, "
          "so an intervention that reduces ponding is correctly credited with "
          "reducing inlet capture too.", False)],
        "",
        [("Not calibrated.", True), ("  No per-inlet data exists for Beirut. "
                                     "Results are reported as differences, "
                                     "which are robust to a common bias.",
                                     False)],
    ], size=13)
    drop_ph(s, 10)
    fit(prs, s, f"{F}/eq_drain.png", (6.95, 1.35, 6.05, 1.9))
    fit(prs, s, f"{F}/tech_gully.png", (7.25, 3.35, 5.5, 3.0))
    notes(s, "Small piece of physics, but it closes a feedback loop the "
             "capacity model breaks: less ponding means less capture, so the "
             "corridor is not double-counted.")

    chapter(prs, "VERIFICATION AND VALIDATION")

    s = prs.slides.add_slide(prs.slide_layouts[L_TEXT_IMG])
    title_of(s, "Analytic tests and the mass audit")
    body(s, 1, [
        [("Five analytic tests, both schemes, all passing", True), ("", False)],
        "Closed box — volume conservation, no leakage:  exact",
        "Planar runoff — steady depth vs Manning normal depth:  14.1 vs 14.7 mm",
        "Lake at rest — well-balancedness over an uneven bed:  6 × 10⁻⁵ m/s",
        "Reference-port equivalence — independent implementation:  0.027 mm",
        "Erosion sanity — bit-identical when disabled, closes when on",
        "",
        [("Every production run carries its own audit:", True),
         ("  rain = infiltrated + drained + outflow + stored, closing to "
          "better than 3 × 10⁻⁵.", False)],
    ], size=12.5)
    drop_ph(s, 10)
    fit(prs, s, f"{F}/tech_massbalance.png", (6.95, 1.9, 6.05, 3.8))
    notes(s, "Lake at rest is the one to mention if pressed — it is the test "
             "many schemes fail, and this domain is terraced with retaining "
             "walls, so a non-well-balanced scheme would generate currents out "
             "of nothing. On the right: LISFLOOD-FP failed exactly this audit "
             "on the same terrain, at Vol/Input = 5.95.")

    text_img(prs, "Against an independently written solver", [
        [("SynxFlow", True), ("  — a separate full shallow-water Godunov code, "
                              "same 0.5 m grid, same storm, same terrain", False)],
        "",
        [("HLLC:", True), ("   IoU 0.913 · correlation 0.985 · RMSE 3.7 cm · "
                           "wet-area bias +1 %", False)],
        [("Inertial:", True), ("   IoU 0.690 · correlation 0.938 · RMSE 7.8 cm · "
                               "wet-area bias +27 %", False)],
        "",
        [("LISFLOOD-FP: unusable on this terrain.", True),
         ("  Generated 5.95× the input volume and failed its own mass balance "
          "— a documented limitation of its scheme where retaining walls step "
          "the bed. The raster is void, not merely different.", False)],
    ], f"{F}/validation_synxflow.png", size=13)
    notes(prs.slides[-1],
          "The strongest evidence available without field data. Two "
          "independently written Godunov schemes agreeing to 3.7 cm across a "
          "full urban catchment. Note this also quantifies the inertial "
          "scheme's bias on the real problem, and it is the same sign and "
          "cause as the dam-break result.")

    s = two_image(prs, "UK EA benchmark suite (SC120002): Tests 8A and 4",
                  f"{F}/benchmark_ea8.png", f"{F}/benchmark_ea4.png",
                  cap_left="Test 8A — rainfall on a dense urban street network. "
                           "In cluster at all four gauges with published "
                           "curves: P1 0.579 · P2 0.239 · P3 0.726 · P6 0.064 m.",
                  cap_right="Test 4 — flood propagation over a floodplain. In "
                            "cluster at all nine cross-section points, on both "
                            "schemes. Built from the written specification.")
    notes(s, "19 packages in the published comparison. 8A is the closest "
             "published analogue to what we do — direct rainfall on a real "
             "street network. Of the remaining six cases, four need terrain "
             "and boundary data we do not have and two need coupled 1-D sewer "
             "or river models we do not implement. Test 3 is the one worth "
             "acquiring: it discriminates shock-capturing schemes.")

    text_img(prs, "Resolution: answering an open question in the EA report", [
        [("The report leaves it open:", True),
         ("  “it is not currently clear that grid resolutions finer than 2 m "
          "will improve the quality of velocity predictions”, and recommends "
          "it be tested at 0.5 m.", False)],
        "",
        "Same terrain, block-aggregated — the computational grid is the only "
        "quantity varying",
        "",
        [("0.5 m: 9.4 % · 1 m: 2.4 % · 2 m: 0.6 % · 4 m: 2.4 %", True),
         ("   measured corridor benefit", False)],
        [("p99 street velocity climbs 4.32 → 14.76 m/s as the grid coarsens",
          True), ("  — the wrong direction, and physically implausible: "
                  "block-averaging manufactures artificial gradients while "
                  "erasing the kerbs that convey the water.", False)],
        "",
        [("Caveat:", True), ("  coarsening is not the same as building "
                             "natively at 2 m with recalibrated roughness. "
                             "Direction and magnitude are unambiguous.", False)],
    ], f"{F}/resolution.png", size=12)
    notes(prs.slides[-1],
          "A study done at the benchmark's own 2 m would conclude the "
          "intervention does essentially nothing. For appraising interventions "
          "at street scale, sub-metre is a precondition, not a refinement.")

    chapter(prs, "APPLICATION AND THE DIFFERENTIABLE LAYER")

    text_img(prs, "Attribution: which model change moved the answer", [
        [("Walkable-surface reduction, by configuration", True), ("", False)],
        f"T2 frequent:   {n['t2']['walk']:.1f} %   "
        f"(baseline 55.6 → HLLC 52.9 → best {n['t2']['walk']:.1f})",
        f"25 Nov observed:   {n['v1_nov2025']['walk']:.1f} %   "
        f"(44.3 → 42.8 → {n['v1_nov2025']['walk']:.1f})",
        f"T50 severe:   {n['t50']['walk']:.1f} %   "
        f"(26.4 → −9.0 → {n['t50']['walk']:.1f})",
        "",
        [("The severe-storm sign flip is numerics, not soil physics:", True),
         ("  shock capturing alone moves it −35.4 points, Green–Ampt returns "
          "+8.3. The old scheme was over-spreading water in the before-case, "
          "where flow is fastest and most supercritical, and inflating the "
          "apparent benefit.", False)],
    ], f"{F}/tech_attribution.png", size=12.5)
    notes(prs.slides[-1],
          "This is the slide that justifies having built the HLLC scheme. "
          "Without it the headline for a 50-year storm would still read +26 %, "
          "and it would be wrong.")

    text_img(prs, "Beirut: what the corridor actually does", [
        [("Water balance, best-validated configuration", True), ("", False)],
        f"T2:   +{n['t2']['infil']:,.0f} m³ infiltrated · "
        f"−{n['t2']['out']:.0f} % outflow toward the port",
        f"25 Nov:   +{n['v1_nov2025']['infil']:,.0f} m³ · "
        f"−{n['v1_nov2025']['out']:.0f} %",
        f"T50:   +{n['t50']['infil']:,.0f} m³ · −{n['t50']['out']:.0f} %",
        "",
        [("In a 50-year storm the corridor stops reducing its own flooded "
          "area but keeps doing hydrological work", True),
         (" — its swales and ponds fill and hold. Two distinct functions; only "
          "the downstream one survives an extreme event.", False)],
        "",
        [("Correction:", True),
         (f"  the infiltration multiple is {n['v1_nov2025']['infil_x']:.2f}×, "
          "not the 3× previously claimed. Ordinary Beirut ground absorbs far "
          "more than a constant rate implied. Better physics, smaller claim.",
          False)],
    ], f"{F}/beirut_water_balance.png", size=12)
    notes(prs.slides[-1],
          "The absolute infiltration gain held up; the ratio did not. Say the "
          "correction plainly — it is the kind of thing a referee finds.")

    opt = json.load(open("output/design_opt/result.json"))
    text_img(prs, "The solver differentiates itself", [
        [("Written in torch, so ∂(flooded area)/∂(design) comes from autograd",
          True), ("", False)],
        "Verified against central finite differences to 2 × 10⁻⁸ over an "
        "epsilon sweep; gradient checkpointing makes full-length storms "
        "tractable and is bit-identical to the direct backward pass",
        "SIMP-style continuous relaxation between paved and green endpoints, "
        "with a density filter and a bisection projection onto an area budget",
        "",
        [(f"Result at equal budget ({opt['budget_frac']:.0%} of the patch):",
          True), ("", False)],
        [(f"do nothing {opt['flooded_m2']['do_nothing']:.0f} m²  ·  "
          f"spread evenly {opt['flooded_m2']['uniform']:.0f}  ·  "
          f"greedy {opt['flooded_m2']['greedy']:.0f}  ·  ", False),
         (f"gradient {opt['flooded_m2']['gradient']:.0f} m²", True)],
        "",
        [("Not yet a claim.", True),
         ("  Thresholded to a buildable binary layout the optimum degrades to "
          "159 m² against greedy's 149 — the relaxation needs a Heaviside "
          "projection with continuation before this stands up.", False)],
    ], f"{F}/tech_optimizer.png", size=12)
    notes(prs.slides[-1],
          "The differentiator for a publication. None of the 19 packages in "
          "the EA comparison is differentiable, and none models infiltration "
          "or green infrastructure at all. Be honest about the binary gap — "
          "the continuous result is real, the buildable one is not there yet. "
          "The objective is also still instantaneous; peak-over-time and a "
          "cost on gullies are the next two changes.")

    s = prs.slides.add_slide(prs.slide_layouts[L_TWO_TEXT])
    title_of(s, "Where it stands, and what is missing")
    body(s, 1, [
        [("Established", True), ("", False)],
        "Solves the equations correctly — five analytic tests, both schemes",
        "Agrees with an independent Godunov code to 3.7 cm / IoU 0.913",
        "Inside the published cluster on two EA benchmark cases",
        "Mass closes to 3 × 10⁻⁵ on every run",
        "Sub-metre resolution demonstrably changes the conclusion",
        "Gradients verified to 2 × 10⁻⁸",
    ], size=12.5)
    body(s, 2, [
        [("Missing", True), ("", False)],
        [("No observed flood data for Beirut.", True),
         ("  Every comparison above is model-to-model. Verified, not "
          "validated.", False)],
        "Material properties are literature values, not site measurements; "
        "Green–Ampt adds four uncalibrated parameters per cover class",
        "Drain inlet capacity assumed; no per-inlet data exists",
        "Corridor cross-section interpreted from published drawings",
        "Erosion sub-model illustrative, and disabled in all results shown",
        "No coupled 1-D sewer or river model — two EA cases out of scope",
    ], size=12)
    notes(s, "Lead with the missing column if the audience is sceptical. The "
             "honest framing is that the numerics are in good shape and the "
             "parameterisation is not, and only field measurement fixes the "
             "second.")

    s = prs.slides.add_slide(prs.slide_layouts[L_OUTRO])
    title_of(s, "Questions")
    prs.save(out)
    print(f"wrote {out}  ({len(prs.slides._sldIdLst)} slides)")


# ======================================================= non-technical deck ==
def build_plain(prs, F, n, out, runs, material):
    # The two areas quoted on the depth slide, computed exactly as the figure
    # computes them so the words and the picture cannot disagree.
    _, walk = corridor_masks(material)
    res2 = 0.25
    wet_b = float((walk & (np.load(f"{runs}/before_v1_nov2025/max_depth.npy")
                           > 0.10)).sum()) * res2
    wet_a = float((walk & (np.load(f"{runs}/after_v1_nov2025/max_depth.npy")
                           > 0.10)).sum()) * res2

    s = prs.slides.add_slide(prs.slide_layouts[L_INTRO])
    title_of(s, "Where the rain goes")
    tb = s.shapes.add_textbox(Inches(0.92), Inches(4.62), Inches(7.6), Inches(1.2))
    set_text(tb, ["What a green corridor would do to flooding in Beirut",
                  "A model of the Al-Masar Al-Akhdar right-of-way, built from "
                  "a laser survey of the city"], size=13.5, color=WHITE)
    notes(s, "No equations in this talk. What the model is told, what it "
             "works out, and what it says about the corridor.")

    s = prs.slides.add_slide(prs.slide_layouts[L_TWO_TEXT])
    title_of(s, "The problem is not extreme rain")
    body(s, 1, [
        [("25 November 2025", True), ("", False)],
        "25 mm of rain fell in half an hour.",
        "Sassine Square and the Ring flooded. Traffic stopped. Basements "
        "filled.",
        "",
        [("By Lebanon's own stormwater code that burst is a 2-to-5 year "
          "event", True), ("  — the kind of rain the city should expect "
                           "several times a decade.", False)],
    ], size=15)
    body(s, 2, [
        [("So why does it flood?", True), ("", False)],
        "The ground is sealed. Rain that would once have soaked into soil now "
        "runs straight off asphalt and concrete.",
        "The slopes are steep. Water arriving from Achrafieh reaches the lower "
        "streets fast and deep.",
        "The drains take what they can and the rest stays on the street.",
        "",
        [("A green corridor is an attempt to give some of that water "
          "somewhere to go.", True), ("", False)],
    ], size=14)
    notes(s, "Set the frame: this is not about a freak storm, it is about "
             "ordinary rain on a surface that cannot absorb it. That is what "
             "makes it fixable.")

    img_slide(prs, f"{F}/nt_blackbox.png",
              "What the model is, in one picture",
              "The model is told four things and works out the rest. It is "
              "not given a drainage plan or a list of places that flood.",
              box=(0.30, 1.32, 12.75, 5.10))
    notes(prs.slides[-1],
          "THE slide of this talk. The point to land: nothing about where the "
          "flooding happens is put in by hand. Rain is placed on every "
          "half-metre square of the city and the physics moves it. That is "
          "why the model can be trusted to say something about a design that "
          "has never been built.")

    img_slide(prs, f"{F}/nt_inputs.png", "What goes in",
              "A laser survey of the ground surface; what each patch of ground "
              "is made of; a storm; and the proposed design. Nothing else.",
              box=(0.30, 1.35, 12.75, 5.05))
    notes(prs.slides[-1],
          "39 GB of airborne laser data, gridded to half a metre — about 2.3 "
          "million squares. The land-cover map is what decides whether a "
          "square soaks water up or sheds it. The storm is a real recorded "
          "one. If the design team gives us their real drawings, that is a "
          "data swap and a re-run, not a rebuild.")

    img_slide(prs, f"{F}/beirut_before_after.png", "What comes out",
              "Peak water depth during the observed storm — as things are "
              "today, with the corridor built, and the difference. Blue on the "
              "right-hand map means drier.",
              box=(0.41, 1.45, 12.60, 4.85))
    notes(prs.slides[-1],
          "Note the effect reaches beyond the corridor itself into the "
          "surrounding streets — the corridor sits on a natural drainage line, "
          "so it intercepts water coming down from Achrafieh. The large square "
          "at top left is a real deep construction excavation, surveyed by the "
          "laser and modelled as a basin. Genuine terrain, not an error.")

    s = prs.slides.add_slide(prs.slide_layouts[L_TEXT_IMG])
    title_of(s, "How deep, in terms you can picture")
    body(s, 1, [
        "The model reports depth everywhere, every few seconds. Translated "
        "into what it means for a person on foot:",
        "",
        [("Today:", True), (f"  {wet_b:,.0f} m² of the corridor's walking "
                            "surface goes under water in the observed storm.",
                            False)],
        [("With the corridor:", True),
         (f"  {wet_a:,.0f} m² — {100*(wet_b-wet_a)/wet_b:.0f} % of it removed.",
          False)],
        "",
        "The dark outline is the corridor. Water that still shows inside it is "
        "mostly in the rain gardens and ponds, which are dug out to hold water "
        "on purpose.",
    ], size=14)
    drop_ph(s, 10)
    fit(prs, s, f"{F}/nt_depth_classes.png", (7.05, 1.00, 5.95, 5.50))
    notes(s, "Depth in centimetres means little to most people; 'over your "
             "shoe' and 'shin to knee' mean a lot. If asked why water remains "
             "inside the corridor: that is the design working, not failing.")

    s = prs.slides.add_slide(prs.slide_layouts[L_TEXT_IMG])
    title_of(s, "The corridor, as the model builds it")
    body(s, 1, [
        "31,828 m² in total, laid down half a metre at a time",
        "Seven different surfaces — rain garden, bioswale, bioretention pond, "
        "terrace, porous sidewalk and bikelane, vehicular lane — each with its "
        "own ability to absorb, slow and hold water",
        "Rain garden is just over half of it; the engineered bioswales and "
        "ponds together are under 4,000 m²",
        "Buildings in the corridor's path are removed from the terrain first, "
        "so the comparison is genuinely before-and-after",
        "",
        [("This is our reading of the design.", True),
         ("  The route and the zone come from the published dataset. The "
          "cross-section — how wide each band is, where the swales and ponds "
          "sit — is what we could make out from the drawings and what seemed "
          "sensible to us. Real dimensions would be a straight swap.", False)],
    ], size=12.5)
    drop_ph(s, 10)
    fit(prs, s, f"{F}/corridor_materials.png", BOX_IMG_RIGHT)
    notes(s, "Say the caveat unprompted — it costs nothing and it is the first "
             "thing a designer in the room will want to know. Frame it as an "
             "invitation: give us the real numbers and we re-run in a day.")

    img_slide(prs, f"{F}/nt_headline.png", "What it says",
              "Measured on the surfaces people walk on. The fifth of the "
              "corridor dug out as swales and ponds is excluded — it holds "
              "water by design, and counting that as flooding would penalise "
              "the design for working.",
              box=(0.30, 1.42, 12.75, 4.85))
    notes(prs.slides[-1],
          "Three storms, three answers. The honest headline is that it works "
          "well for the rain Beirut actually gets most years, and that in a "
          "genuine 50-year storm it stops keeping its own surface dry and "
          "instead fills up and protects everywhere downstream. Two different "
          "jobs. Do not oversell the third card.")

    text_img(prs, "The benefit reaches about two blocks", [
        "The corridor is a thin ribbon, but the water it takes in would "
        "otherwise have run through the streets around it.",
        "",
        "Measured by distance from the corridor edge, the improvement is "
        "strong on the corridor itself, clear within 25 m, still visible at "
        "50 m, and gone by about 100 m.",
        "",
        [("This is a useful planning number.", True),
         ("  It says which streets a corridor helps and which it does not — "
          "and that a single corridor is a local remedy, not a city-wide one.",
          False)],
    ], f"{F}/beirut_bands.png", size=14)
    notes(prs.slides[-1],
          "Useful for managing expectations. Someone will ask whether this "
          "fixes flooding in Beirut. It does not — it fixes flooding along "
          "this axis, and the model can say exactly how far.")

    text_img(prs, "Where the water ends up", [
        [("Two things happen to rain that the corridor catches", True),
         ("", False)],
        "",
        f"It soaks into the ground: an extra "
        f"{n['v1_nov2025']['infil']:,.0f} m³ in the observed storm — roughly "
        f"{n['v1_nov2025']['infil']/2.5:,.0f} bathtubs' worth that used to run "
        "down the street.",
        "",
        f"It stops flowing toward the port: {n['v1_nov2025']['out']:.0f} % less "
        "water reaches the sea outlet, which is where the city's drainage is "
        "under most strain.",
        "",
        [("In the 50-year storm the corridor absorbs the most of all — "
          f"+{n['t50']['infil']:,.0f} m³", True),
         (" — even though its own surface stays wet. It is doing its job "
          "for everyone downstream.", False)],
    ], f"{F}/beirut_water_balance.png", size=13)
    notes(prs.slides[-1],
          "The bathtub comparison is only there to make cubic metres "
          "imaginable — use it or drop it. The serious point is the second "
          "one: less water reaching the port outlet is a benefit that does "
          "not show up on a flood map at all.")

    s = prs.slides.add_slide(prs.slide_layouts[L_TWO_TEXT])
    title_of(s, "How much should you believe it?")
    body(s, 1, [
        [("What has been checked", True), ("", False)],
        "It gets the textbook problems right — cases where the correct answer "
        "can be worked out on paper, all five of them.",
        "It was run against a flood simulator written by a different team, on "
        "the same streets and the same storm. The two agree to within about "
        "4 cm of water depth.",
        "It was run against two international benchmark tests used to compare "
        "flood models, and sits inside the range of the nineteen commercial "
        "packages in that comparison.",
        "It accounts for every drop: rain in equals water soaked up, drained, "
        "flowed away, or still standing — to five decimal places, every run.",
    ], size=12.5)
    body(s, 2, [
        [("What has not", True), ("", False)],
        [("Nobody has measured a flood in Beirut to compare it against.",
          True), ("", False)],
        "Everything above compares the model to other models, or to problems "
        "with known answers. That is real evidence, and it is not the same as "
        "being right about this city.",
        "",
        "The absorption rates of soils and paving are taken from published "
        "engineering tables, not measured on site.",
        "",
        [("So: use the comparisons with confidence — before versus after, "
          "which streets, roughly how much. Treat the absolute depths with "
          "more caution.", True), ("", False)],
    ], size=12)
    notes(s, "Do not skip the right-hand column. Volunteering the limitation "
             "is what makes the left-hand column credible, and it sets up the "
             "ask on the next slide.")

    s = prs.slides.add_slide(prs.slide_layouts[L_TWO_TEXT])
    title_of(s, "What would settle it")
    body(s, 1, [
        [("A modest field campaign, and nothing else, closes the gap", True),
         ("", False)],
        "",
        "After the next significant storm: photographs and levelled marks "
        "where the water reached, at a few dozen street locations.",
        "",
        "A handful of low-cost depth loggers at points that pond reliably.",
        "",
        "A note of when key streets became impassable, and when they cleared.",
        "",
        "No specialist equipment. No large team.",
    ], size=13)
    body(s, 2, [
        [("What it would buy", True), ("", False)],
        "",
        "A measured error instead of a model-to-model one — the difference "
        "between a study that is defensible and one that can carry a planning "
        "decision.",
        "",
        [("We can help make it efficient.", True),
         ("  The model already predicts where flooding concentrates, so it can "
          "site the loggers and the survey points where they will be most "
          "informative. We would be glad to do that work alongside the "
          "project.", False)],
    ], size=13)
    notes(s, "Land this politely and concretely. It is an offer, not a "
             "complaint: we bring the model, they bring the city, and the "
             "measurement is cheap. Do not leave it as a vague 'more data "
             "would be nice'.")

    s = prs.slides.add_slide(prs.slide_layouts[L_OUTRO])
    title_of(s, "Thank you")
    prs.save(out)
    print(f"wrote {out}  ({len(prs.slides._sldIdLst)} slides)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template",
                    default="output/presentation/Dynacity_Presentation_Template.pptx")
    ap.add_argument("--figs", default="output/presentation")
    ap.add_argument("--runs", default="output/corridor_runs_v3_best")
    ap.add_argument("--material", default="output/corridor_gi_cut_v3/material.npy")
    ap.add_argument("--out-technical",
                    default="output/presentation/almasar_technical_deck.pptx")
    ap.add_argument("--out-plain",
                    default="output/presentation/almasar_overview_deck.pptx")
    a = ap.parse_args()

    n = numbers(a.runs, a.material)
    build_technical(deck(a.template), a.figs, n, a.out_technical)
    build_plain(deck(a.template), a.figs, n, a.out_plain, a.runs, a.material)


if __name__ == "__main__":
    main()
