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
        fit(prs, s, left, (1.55, 1.02, 10.2, 2.35))
        fit(prs, s, right, (0.60, 3.55, 12.1, 2.80))
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
    """Pitched at an engineer or a software developer, not a hydraulics
    specialist: the equations stay, the literature names go, and every term is
    said in terms of what it does."""
    s = prs.slides.add_slide(prs.slide_layouts[L_INTRO])
    title_of(s, "How the flood model works")
    tb = s.shapes.add_textbox(Inches(0.92), Inches(4.62), Inches(7.6), Inches(1.2))
    set_text(tb, ["A physics simulation of rain running through a city, at "
                  "half-metre resolution",
                  "What it computes · how it is tested · what it says about the "
                  "Beirut green corridor"], size=13.5, color=WHITE)
    notes(s, "No hydraulics background assumed. Three questions: what does it "
             "actually compute, how do we know it is right, and what did it "
             "tell us. Stop me on any of the maths — none of it is deep, it is "
             "just unfamiliar.")

    s = prs.slides.add_slide(prs.slide_layouts[L_TWO_TEXT])
    title_of(s, "The idea in one paragraph")
    body(s, 1, [
        [("Cut the city into half-metre squares. Put rain on all of them. "
          "Let physics move the water.", True), ("", False)],
        "",
        "2.3 million squares, each holding three numbers: how deep the water "
        "is, and how fast it is moving in x and y.",
        "",
        "Every timestep, work out how much water crosses between neighbouring "
        "squares, add the rain, subtract what soaks into the ground and what "
        "the drains take, and repeat — about 75,000 times for one hour of "
        "storm.",
        "",
        "It runs on a GPU because every square updates independently, which is "
        "exactly what a GPU is for.",
    ], size=13.5)
    body(s, 2, [
        [("Why this rather than a drainage model", True), ("", False)],
        "",
        "The usual approach divides the city into catchments, guesses what "
        "fraction of rain runs off each, and routes the result through a pipe "
        "network you have drawn.",
        "",
        [("That requires you to decide in advance where the water goes.",
          True), ("", False)],
        "",
        "Here nothing is routed by hand. Where the flooding shows up is an "
        "output. That is the property that lets the model say something "
        "useful about a design that has never been built.",
    ], size=13.5)
    notes(s, "This slide is the whole talk in miniature. If someone takes only "
             "one thing, it should be that the flood pattern is computed, not "
             "assumed.")

    chapter(prs, "WHAT IT COMPUTES")

    img_slide(prs, f"{F}/tech_loop.png", "The main loop",
              "Nothing exotic: a stencil over a regular grid, a state update, "
              "source terms, and a stability-limited timestep. The whole thing "
              "is tensor operations — there is no per-cell Python anywhere in "
              "the hot path.",
              box=(0.35, 1.30, 12.65, 4.45))
    notes(s := prs.slides[-1],
          "Deliberately familiar territory for anyone who has written "
          "numerical code. The only two non-obvious parts are step 1, which "
          "is the next slide, and step 3's infiltration, which needs an "
          "iterative solve because the law cannot be rearranged.")

    text_img(prs, "The equations being stepped", [
        [("Conservation of water and of momentum, averaged over depth", True),
         ("", False)],
        "",
        "The first line is bookkeeping: a square's water level changes by what "
        "flows in and out, plus rain, minus what soaks in and what the gullies "
        "take.",
        "",
        "The next two are Newton's second law for the water: it accelerates "
        "under its own pressure and under gravity pulling it downhill, and "
        "friction with the ground slows it.",
        "",
        [("That is the entire physics.", True),
         ("  Everything else in this talk is about computing it accurately, "
          "and checking that we did.", False)],
    ], f"{F}/eq_swe.png", size=13, box=(6.85, 1.55, 6.2, 4.5))
    notes(prs.slides[-1],
          "Do not linger. The point is that the physics is short and standard "
          "— which is exactly why the model can be compared against other "
          "packages. The interesting engineering is downstream of here.")

    text_img(prs, "Two ways to compute the flow between squares", [
        [("The cheap way (\"inertial\")", True),
         ("  drops one term — the momentum the water carries with it as it "
          "moves. What is left can be solved almost directly. This is what "
          "most flood models in production use, including LISFLOOD-FP.",
          False)],
        "",
        [("The full way (\"shock-capturing\")", True),
         ("  keeps every term and solves a small problem at each face: given "
          "the water either side, what actually flows across? It costs about "
          "3.5× more and needs a timestep about half as long.", False)],
        "",
        [("We implemented both, in one code.", True),
         ("  Same terrain, same rain, same everything else — so the difference "
          "between two runs is the method and nothing else. Most studies "
          "cannot measure that, because they only have one solver.", False)],
    ], f"{F}/eq_numerics.png", size=12.5, box=(6.85, 1.35, 6.2, 4.9))
    notes(prs.slides[-1],
          "If asked what the dropped term does: it is the difference between "
          "water that simply spreads and water that has momentum and keeps "
          "going. It matters when flow is fast — which in Beirut it is, 6 to "
          "8 m/s on a 6 % street grade. Next slide shows how much it matters.")

    img_slide(prs, f"{F}/tech_dambreak.png",
              "The test that separates them: a dam break with a known answer",
              "Release a wall of water onto dry ground. This problem has an "
              "exact solution you can write down on paper, so there is no "
              "argument about what the right answer is. Left: both methods "
              "against it. Right: the same test on progressively finer grids.",
              box=BOX_FULL_TALL)
    notes(prs.slides[-1],
          "The single most useful slide for a sceptical engineer. Three things "
          "to point at. One: the full method sits on the exact curve. Two: the "
          "cheap method oscillates and its front is 60 % too slow — it does "
          "not just lose accuracy, it converges to the wrong answer, which is "
          "the thing you always want to catch. Three: the right panel is a "
          "convergence test — halve the grid, error should fall. It does for "
          "one method and flatlines for the other. This is why we pay 3.5× for "
          "the full solver.")

    s = two_image(prs, "Where the water goes when it is not on the street",
                  f"{F}/eq_infil.png", f"{F}/tech_greenampt.png", stacked=True)
    notes(s, "The green corridor works by soaking water up, so this term is "
             "not a detail — it is the mechanism under test. A constant "
             "soak-up rate is wrong in a way that matters: ground absorbs "
             "fast when dry and slows as it fills, like a sponge. Left panel "
             "shows that decay; the dotted lines are the constant rate a "
             "simpler model would have used throughout. Right panel is the "
             "verification — our solver against the exact answer. Switching to "
             "this is what dropped the corridor's claimed infiltration gain "
             "from 3× to 1.3×.")

    s = prs.slides.add_slide(prs.slide_layouts[L_TEXT_IMG])
    title_of(s, "Storm drains that fill up")
    body(s, 1, [
        "A gully grate has two regimes. Shallow water spills over its edge; "
        "deep water is forced through the opening. Which one limits the flow "
        "depends on how much water is standing on top of it.",
        "",
        [("Why not just give each drain a fixed capacity:", True),
         ("  because then a design that reduces ponding gets no penalty for "
          "also reducing what the drains capture. Making capture depend on "
          "depth closes that loop, so the corridor is not credited twice.",
          False)],
        "",
        [("Honest caveat:", True), ("  nobody has data on Beirut's actual "
                                    "inlets. We report differences rather "
                                    "than absolutes, which is robust to "
                                    "getting this uniformly wrong.", False)],
    ], size=13)
    drop_ph(s, 10)
    fit(prs, s, f"{F}/eq_drain.png", (6.72, 1.22, 6.35, 2.05))
    fit(prs, s, f"{F}/tech_gully.png", (7.35, 3.45, 5.3, 2.85))
    notes(s, "Small piece of physics, but a good example of the general "
             "principle: model the mechanism, not the outcome, and the "
             "feedbacks come for free.")

    chapter(prs, "HOW WE KNOW IT IS RIGHT")

    s = prs.slides.add_slide(prs.slide_layouts[L_TEXT_IMG])
    title_of(s, "A test suite, and an assertion on every run")
    body(s, 1, [
        [("Five problems with answers known in advance", True), ("", False)],
        "Sealed box — put rain in, check nothing leaks out:  exact",
        "Water running down a slope — check the steady depth matches the "
        "textbook formula:  14.1 vs 14.7 mm",
        "A still pond on bumpy ground — check the model does not invent "
        "currents:  6 × 10⁻⁵ m/s",
        "Same problem through an independent implementation:  agrees to "
        "0.027 mm",
        "Erosion off — check the code path is bit-identical to before it "
        "existed",
        "",
        [("And an invariant checked on every production run:", True),
         ("  rain in = soaked up + drained + flowed out + still standing. "
          "It closes to better than 3 parts in 100,000, every time.", False)],
    ], size=12)
    drop_ph(s, 10)
    fit(prs, s, f"{F}/tech_massbalance.png", (6.95, 1.9, 6.05, 3.8))
    notes(s, "Frame this as unit tests plus a runtime assertion, because that "
             "is what it is. The still-pond test is the one worth calling out: "
             "many methods fail it, and this terrain is full of retaining "
             "walls, so a model that fails it would generate flow out of "
             "nothing. On the right — one of the three engines we compared "
             "against failed exactly this mass check on this terrain, at six "
             "times the input volume.")

    text_img(prs, "Run against a solver somebody else wrote", [
        [("SynxFlow — a separate flood simulator, different team, different "
          "code, same job", True), ("", False)],
        "Same terrain, same grid, same storm. Then compare the depth in every "
        "cell.",
        "",
        [("Our full method:", True), ("   agrees on 91 % of the flooded "
                                      "footprint, 3.7 cm typical difference",
                                      False)],
        [("Our cheap method:", True), ("   69 % overlap, 7.8 cm, and it "
                                       "spreads water over 27 % more area — "
                                       "the same bias the dam break predicted",
                                       False)],
        "",
        [("A third engine, LISFLOOD-FP, could not be used at all:", True),
         ("  it produced six times the water that went in and failed its own "
          "conservation check on this terrain. Its output is void, not merely "
          "different.", False)],
    ], f"{F}/validation_synxflow.png", size=12.5)
    notes(prs.slides[-1],
          "This is the strongest evidence available short of measuring a real "
          "flood. Two codes written independently, agreeing to under 4 cm "
          "across a whole catchment. Note it also confirms the dam-break "
          "finding on the real problem — same sign, same cause.")

    s = two_image(prs, "Two standard benchmark cases used to compare flood models",
                  f"{F}/benchmark_ea8.png", f"{F}/benchmark_ea4.png",
                  cap_left="Rain on a real, dense street network — the closest "
                           "published case to what we do. Our answer sits "
                           "inside the range of published results at all four "
                           "measurement points.",
                  cap_right="How fast a flood front travels and how deep the "
                            "water is behind it. Inside the published range at "
                            "all nine points, on both methods.")
    notes(s, "The UK Environment Agency publishes a benchmark suite with "
             "results from 19 commercial and research packages, so 'in "
             "cluster' means we land where the established tools land. Of the "
             "other six cases, four need data files we do not have and two "
             "need a coupled sewer model we have not built.")

    text_img(prs, "Does the half-metre grid earn its cost?", [
        "Coarsen the same terrain step by step and re-run. Only the grid "
        "changes — same rain, same everything.",
        "",
        [("At 0.5 m the corridor removes 9.4 % of the flooding. At 2 m: 0.6 %.",
          True), ("", False)],
        "A study done at 2 m would conclude the intervention does essentially "
        "nothing.",
        "",
        [("Peak street velocity goes the wrong way", True),
         (" — 4.3 up to 14.8 m/s as the grid gets coarser. Averaging blocks "
          "of cells together erases the kerbs and narrow streets that actually "
          "carry the water, while inventing steep steps between the big cells "
          "that remain.", False)],
        "",
        [("Caveat:", True), ("  coarsening an existing model is not identical "
                             "to building one natively at 2 m with retuned "
                             "roughness. The size and direction of the effect "
                             "are not in doubt.", False)],
    ], f"{F}/resolution.png", size=12)
    notes(prs.slides[-1],
          "Worth stressing that the benchmark report itself lists this as an "
          "open question and asks somebody to test it at 0.5 m. We are "
          "unusually well placed to answer it, and the answer is that "
          "resolution decides the conclusion.")

    chapter(prs, "BEIRUT, AND WHAT COMES NEXT")

    text_img(prs, "Which change to the model moved the answer", [
        [("Same corridor, same storms, three versions of the model", True),
         ("", False)],
        f"Frequent storm:   55.6 → 52.9 → {n['t2']['walk']:.1f} %  reduction "
        "in flooded walkable area",
        f"Observed 25 Nov:   44.3 → 42.8 → {n['v1_nov2025']['walk']:.1f} %",
        f"Severe 50-year:   26.4 → −9.0 → {n['t50']['walk']:.1f} %",
        "",
        [("The severe storm is the interesting one.", True),
         ("  Switching to the accurate flow method alone moved it by 35 "
          "percentage points — the cheap method had been over-spreading water "
          "in the do-nothing case, where flow is fastest, and inflating the "
          "apparent benefit. Better soil physics then gave 8 points back.",
          False)],
        "",
        [("This is why we built the second solver.", True),
         ("  Without it the headline would still read +26 %, and it would be "
          "wrong.", False)],
    ], f"{F}/tech_attribution.png", size=12)
    notes(prs.slides[-1],
          "A good slide to dwell on with an engineering audience: it is a "
          "worked example of a numerical artefact masquerading as a physical "
          "result, caught by having two implementations.")

    text_img(prs, "What the corridor actually does", [
        [("Water balance, best-validated configuration", True), ("", False)],
        f"Frequent storm:   +{n['t2']['infil']:,.0f} m³ soaked up · "
        f"{n['t2']['out']:.0f} % less water sent toward the port",
        f"Observed 25 Nov:   +{n['v1_nov2025']['infil']:,.0f} m³ · "
        f"{n['v1_nov2025']['out']:.0f} %",
        f"Severe 50-year:   +{n['t50']['infil']:,.0f} m³ · "
        f"{n['t50']['out']:.0f} %",
        "",
        [("In a 50-year storm it stops reducing its own flooded area but "
          "absorbs more than ever.", True),
         ("  The swales and ponds fill and hold. Two different jobs — keeping "
          "its own surface usable, and protecting what is downstream — and "
          "only the second survives an extreme event.", False)],
        "",
        [("Correction to the earlier study:", True),
         (f"  infiltration goes up {n['v1_nov2025']['infil_x']:.2f}×, not the "
          "3× previously claimed. Ordinary Beirut ground absorbs far more than "
          "a constant-rate model implied.", False)],
    ], f"{F}/beirut_water_balance.png", size=12)
    notes(prs.slides[-1],
          "Say the correction plainly. The absolute gain held; the ratio did "
          "not, because the baseline was wrong. Better physics, smaller claim "
          "— that is how it should go.")

    opt = json.load(open("output/design_opt/result.json"))
    text_img(prs, "The model can be differentiated", [
        [("It is written in PyTorch, so autograd works through the whole "
          "simulation.", True), ("", False)],
        "Backpropagate through 75,000 timesteps and you get the derivative of "
        "flooded area with respect to every cell of the design — the same "
        "machinery that trains a neural network, applied to a physics solver.",
        "",
        "Checked against finite differences to 2 × 10⁻⁸. Gradient "
        "checkpointing trades compute for memory so full storms fit.",
        "",
        [("So you can optimise a design instead of guessing at a few:", True),
         ("", False)],
        [(f"do nothing {opt['flooded_m2']['do_nothing']:.0f} m²  ·  "
          f"spread evenly {opt['flooded_m2']['uniform']:.0f}  ·  "
          f"greedy {opt['flooded_m2']['greedy']:.0f}  ·  ", False),
         (f"gradient descent {opt['flooded_m2']['gradient']:.0f} m²", True)],
        "",
        [("Not a finished claim.", True),
         ("  Forced to a buildable yes/no layout the advantage disappears. "
          "The relaxation needs more work before this stands up.", False)],
    ], f"{F}/tech_optimizer.png", size=11.5)
    notes(prs.slides[-1],
          "The part with publication potential. None of the 19 benchmark "
          "packages is differentiable, and none models infiltration or green "
          "infrastructure at all — they are hydraulics only. Be straight about "
          "the gap: the continuous optimum is real, the buildable one is not "
          "there yet, and the objective is still instantaneous rather than "
          "peak-over-time.")

    s = prs.slides.add_slide(prs.slide_layouts[L_TWO_TEXT])
    title_of(s, "Where it stands, and what is missing")
    body(s, 1, [
        [("Solid", True), ("", False)],
        "Passes five tests with known answers, on both methods",
        "Agrees with an independently written solver to 3.7 cm across a whole "
        "catchment",
        "Lands inside the published range on two standard benchmark cases",
        "Conserves water to 3 parts in 100,000 on every run",
        "Sub-metre resolution demonstrably changes the conclusion — we can "
        "show it rather than assert it",
        "Gradients verified to 2 × 10⁻⁸",
    ], size=12.5)
    body(s, 2, [
        [("Missing", True), ("", False)],
        [("Nobody has measured a real flood in Beirut.", True),
         ("  Everything above compares the model to other models or to "
          "textbook problems. That is real evidence, and it is not the same "
          "as being right about this city.", False)],
        "Soil and paving absorption rates come from published tables, not "
        "site measurements",
        "Drain inlet capacity assumed — no per-inlet data exists",
        "The corridor cross-section is our reading of the published drawings",
        "Erosion model is illustrative, and switched off in everything shown "
        "here",
        "No coupled sewer or river model, which puts two benchmark cases out "
        "of scope",
    ], size=12)
    notes(s, "Lead with the right column if the audience is sceptical. The "
             "honest summary is that the numerics are in good shape and the "
             "input parameters are not, and only field measurement fixes the "
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
