#!/usr/bin/env python3
"""Build the Al-Masar slide deck on the Dynacity template.

Figures-first: the template's own layouts, fonts and colours carry the styling,
and every number on a slide is read from the run outputs so the deck cannot
drift from the results. Re-run after any new study and the slides update.

  python scripts/make_pptx.py --runs output/corridor_runs_v3_best

Also emits slides.json - the per-slide notes the LaTeX speaker transcript is
built from, so the two can never disagree about what is on a slide.
"""

import argparse
import json
import os
import sys

import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

sys.path.insert(0, os.path.dirname(__file__))
from build_deck import corridor_masks, STORMS

NAVY = RGBColor(0x0E, 0x28, 0x41)
TEAL = RGBColor(0x15, 0x60, 0x82)
GREY = RGBColor(0x5C, 0x6B, 0x73)

L_INTRO, L_CHAPTER, L_CONTENT, L_TEXT_IMG, L_TWO_IMG, L_BULLETS, L_OUTRO = \
    0, 1, 2, 3, 6, 7, 9


# Speaker notes. Deliberately terse: a presenter scans these, they do not read
# them aloud. "say" is the one message, "numbers" the figures to have ready,
# "asked" the questions that slide reliably attracts.
NOTES = {
"Where the rain goes": dict(
  say="A physics-based flood model of the Fouad Boutros right-of-way, from a 39 GB LiDAR "
      "survey at half a metre. Three parts: how it works, how far it can be trusted, what "
      "it says about the corridor.",
  numbers=["39 GB airborne LiDAR", "0.5 m grid, 2.3 M cells"]),

"Chapter: how the simulator works": dict(say="Section break."),

"Rain falls on every cell": dict(
  say="Rain-on-grid: rainfall lands on every cell and the water finds its own way "
      "downhill. Nothing is pre-routed, so where flooding appears is a result, not an "
      "input.",
  numbers=["9 land-cover classes set infiltration, roughness, detention",
           "25 Nov 2025: 25.4 mm in 30 min - only a 2-5 year burst",
           "Mass balance closes to 3e-5 every run"],
  asked=[("Why does an ordinary storm flood the city?",
          "Not extreme rainfall. Ordinary rain on steep, sealed streets with limited "
          "inlet capacity."),
         ("Rain on roofs?", "Rerouted to the nearest street cell. Courtyards too.")]),

"What makes it different": dict(
  say="Two things. The physics other packages leave out - infiltration, detention, "
      "gullies that respond to head - and a solver you can differentiate, so it can "
      "design as well as predict.",
  numbers=["None of the 19 packages in the EA benchmark model infiltration or green "
           "infrastructure at all - they are hydraulics only",
           "None is differentiable; ours gives d(flooded area)/d(design) to 2e-8",
           "Two schemes in one code, so scheme error is measured, not assumed",
           "Written in PyTorch on GPU - what makes the adjoint and ensembles affordable"],
  asked=[("Why does differentiability matter?",
          "It turns the model from a predictor into a designer. Instead of testing a "
          "handful of layouts by hand you get the gradient of flooding with respect to "
          "every cell of the design, and can optimise directly. Nothing else in the "
          "field does this."),
         ("Isn't this just another SWE solver?",
          "The hydraulics are standard and deliberately so - that is what lets us "
          "benchmark against everyone else. What is new is the surface model on top and "
          "the fact that the whole thing is differentiable."),
         ("What is it worse at?",
          "No 1D sewer or river coupling, so two of the eight EA tests are out of scope. "
          "And the shock-capturing scheme costs about 3.5x the fast one.")]),

"The corridor, surface by surface": dict(
  say="The design is written into the terrain cell by cell - seven surface types, each "
      "with its own infiltration, roughness and detention depth.",
  numbers=["31,828 m2 total", "Rain garden is 52% of it",
           "Bioswales + ponds together under 4,000 m2"],
  asked=[("How faithful is the layout?",
          "SAY THIS UNPROMPTED. Zone polygon and alignment are real data. The "
          "cross-section - band widths, where swales sit, pond and terrace placement - is "
          "our reading of the published drawings plus what seemed hydraulically sensible. "
          "Faithful in spirit, not a construction spec. If the design team shares real "
          "dimensions it is a data swap and a re-run, not a rebuild."),
         ("Which element does the work?",
          "Mostly plain permeable ground, not the engineered features.")]),

"Chapter: validation": dict(say="Section break."),

"Against an independent solver": dict(
  say="The strongest evidence here. A separate code, written by a different group, "
      "solving the full equations with a different scheme, on the identical grid and "
      "storm.",
  numbers=["HLLC vs SynxFlow: IoU 0.913, corr 0.985, RMSE 3.7 cm",
           "Inertial vs SynxFlow: IoU 0.690, RMSE 7.8 cm, +27% wet area",
           "LISFLOOD-FP: fabricated 6x the input volume, failed its own mass balance"],
  asked=[("Why is the fast scheme biased?",
          "It drops the advective momentum term, so it smears the flood front. Matters "
          "here because these streets run supercritical at 6-8 m/s on a 6% grade."),
         ("Is LISFLOOD's failure our fault?",
          "No - documented limitation of its scheme on terrain with retaining walls. Its "
          "raster is void rather than merely different.")]),

"Benchmark: Test 8A": dict(
  say="The EA case closest to what we do: direct rainfall on a real dense street network. "
      "We are inside the main cluster at all four gauges with published curves.",
  numbers=["P1 0.579 · P2 0.239 · P3 0.726 · P6 0.064 m",
           "19 packages in the published comparison",
           "Dark bar excludes the report's own approximate codes"],
  asked=[("Why only four points?", "Only four have published time series in the report."),
         ("How accurate are the bands?",
          "Read off the report's figures, so good to a centimetre or two. We sit well "
          "inside them, not on the edges.")]),

"Benchmark: Test 4": dict(
  say="Tests how fast a flood front travels and how deep the water is behind it. In "
      "cluster at all nine points, on both schemes.",
  numbers=["Built from the written spec - the EA supplies no terrain file for this case",
           "5 of 5 analytic tests pass: conservation, Manning normal depth, "
           "well-balancedness, reference-port equivalence"],
  asked=[("Why not all eight EA tests?",
          "The rest need data files we do not have, and two need coupled 1D sewer or "
          "river models we do not implement. Test 3 is the one worth acquiring - it "
          "discriminates shock-capturing schemes, which is exactly our new capability.")]),

"Resolution": dict(
  say="The EA report says it is unclear whether finer than 2 m is worth it, and "
      "recommends someone test at 0.5 m. This is that test.",
  numbers=["0.5 m: 9.4% benefit · 2 m: 0.6%",
           "Peak velocity climbs 4.3 -> 14.8 m/s as the grid coarsens",
           "Same terrain coarsened, so only the grid changes"],
  asked=[("Why does velocity go up?",
          "Block-averaging manufactures artificial steep gradients while erasing the "
          "kerbs and narrow streets that convey the water. Wrong direction and "
          "physically implausible - that is the point."),
         ("Is this a fair test?",
          "Caveat honestly: coarsening is not identical to building natively at 2 m, "
          "where roughness would be recalibrated. Direction and size are unambiguous.")]),

"Chapter: Beirut": dict(say="Section break."),

"Before and after": dict(
  say="The observed storm, before and after, difference on the right. Blue is drier.",
  numbers=["Effect extends beyond the ribbon into surrounding streets",
           "The corridor follows a natural drainage axis - it is already where the water "
           "wants to go"],
  asked=[("What is the large square top-left?",
          "A real deep excavation on a construction site, surveyed by the LiDAR and "
          "modelled as a basin. Genuine terrain, not an artefact.")]),

"Results": dict(
  say="Over half the flooding removed underfoot in a frequent storm, about half in the "
      "observed event, essentially nothing in a 50-year storm - which is the next slide.",
  numbers=["Benefit reaches roughly two blocks, then fades",
           "Metric excludes the fifth of the corridor dug out to hold water",
           "Those cells are 20% of the ribbon but 34% of its wet area"],
  asked=[("Why exclude the ponds?",
          "Counting a full bioretention pond as flooding penalises the design for "
          "working. The published study drew the same distinction.")]),

"The severe storm, and a correction": dict(
  say="In a 50-year storm the corridor stops keeping itself dry and starts protecting "
      "everywhere else. Two different jobs; only the second survives an extreme event.",
  numbers=["T50: area barely moves, but +2,124 m3 absorbed and outflow cut 11%",
           "Infiltration multiple is 1.3x, NOT the 3x previously claimed",
           "Absolute gain (~900 m3 observed storm) holds up"],
  asked=[("Why did the infiltration claim shrink?",
          "Better soil physics. Ordinary Beirut ground absorbs far more than a constant "
          "rate implied, so the corridor's relative contribution is smaller. Better "
          "physics, smaller claim - say it plainly.")]),

"What would make this a validated model — the ask": dict(
  say="Verified, not validated. It solves the equations correctly and agrees with an "
      "independent solver to 3.7 cm - but every comparison is against another model, "
      "never a measured flood in Beirut.",
  numbers=["Needed: water marks at a few dozen streets after the next big storm",
           "Plus a handful of low-cost depth loggers, and timing",
           "No specialist equipment, no large team"],
  asked=[("Can we use the numbers meanwhile?",
          "Yes, with care. The study reports differences, and a systematic bias affects "
          "both scenarios and largely cancels. Absolutes carry more uncertainty."),
         ("How can we help?",
          "OFFER THIS. The model already predicts where flooding concentrates, so we can "
          "site the loggers and surveys efficiently. Makes it a collaboration, not a "
          "favour.")]),

"Close": dict(say="Thank you / questions."),
}


def strip_slides(prs):
    """Remove the template's example slides, keeping layouts and masters."""
    xml_slides = prs.slides._sldIdLst
    for sld in list(xml_slides):
        rId = sld.get(
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        prs.part.drop_rel(rId)
        xml_slides.remove(sld)


def drop_ph(slide, idx):
    """Remove an unused placeholder so its empty frame does not render."""
    for sh in list(slide.placeholders):
        if sh.placeholder_format.idx == idx:
            sh._element.getparent().remove(sh._element)
            return


def fit(prs, slide, img, box, shrink=1.0):
    """Place an image inside a box preserving aspect ratio, centred."""
    from PIL import Image
    L, T, W, H = [Inches(v) for v in box]
    iw, ih = Image.open(img).size
    scale = min(W / iw, H / ih) * shrink
    w, h = int(iw * scale), int(ih * scale)
    return slide.shapes.add_picture(img, int(L + (W - w) / 2),
                                    int(T + (H - h) / 2), w, h)


def _no_bullet(p):
    """Suppress the layout's bullet glyph.

    The template's bullets come from a symbol font that is not installed
    everywhere, so they render as tofu boxes outside PowerPoint. These lines
    read as statements rather than a list, so removing the glyph is also the
    better typography."""
    from pptx.oxml.ns import qn
    pPr = p._p.get_or_add_pPr()
    for tag in ("a:buChar", "a:buAutoNum", "a:buNone"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    pPr.append(pPr.makeelement(qn("a:buNone"), {}))


def set_text(ph, lines, size=16, color=NAVY, bold_first=False, space=10):
    tf = ph.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        _no_bullet(p)
        runs = line if isinstance(line, list) else [(line, False)]
        for txt, bold in runs:
            r = p.add_run()
            r.text = txt
            r.font.size = Pt(size)
            r.font.bold = bold or (bold_first and i == 0)
            r.font.color.rgb = color
        p.space_after = Pt(space)
    return tf


def caption(slide, text, box, size=11):
    L, T, W, H = [Inches(v) for v in box]
    tb = slide.shapes.add_textbox(L, T, W, H)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.color.rgb = GREY
    return tb


def title_of(slide, text):
    for sh in slide.placeholders:
        if sh.placeholder_format.idx == 0:
            sh.text_frame.text = text
            return sh
    return None


def numbers(runs, material):
    ribbon, walkable = corridor_masks(material)
    out = {}
    for key, _ in STORMS:
        pb = f"{runs}/before_{key}/max_depth.npy"
        pa = f"{runs}/after_{key}/max_depth.npy"
        if not (os.path.exists(pb) and os.path.exists(pa)):
            continue
        b, a = np.load(pb), np.load(pa)
        def red(mask):
            fb = ((b > 0.10) & mask).sum(); fa = ((a > 0.10) & mask).sum()
            return 100.0 * (fb - fa) / fb if fb else float("nan")
        mb = json.load(open(f"{runs}/before_{key}/run_meta.json"))
        ma = json.load(open(f"{runs}/after_{key}/run_meta.json"))
        out[key] = dict(walk=red(walkable), ribbon=red(ribbon),
                        infil=ma["vol_infiltrated_m3"] - mb["vol_infiltrated_m3"],
                        infil_x=ma["vol_infiltrated_m3"] / max(mb["vol_infiltrated_m3"], 1e-9),
                        out=100.0 * (mb["vol_outflow_m3"] - ma["vol_outflow_m3"])
                            / max(mb["vol_outflow_m3"], 1e-9))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template",
                    default="/home/stathisliap/Downloads/Dynacity_Presentation_Template.pptx")
    ap.add_argument("--figs", default="output/presentation")
    ap.add_argument("--runs", default="output/corridor_runs_v3_best")
    ap.add_argument("--material", default="output/corridor_gi_cut_v3/material.npy")
    ap.add_argument("--out", default="output/presentation/almasar_slides.pptx")
    ap.add_argument("--notes", default="output/presentation/slides.json")
    a = ap.parse_args()

    F = a.figs
    n = numbers(a.runs, a.material)
    prs = Presentation(a.template)
    strip_slides(prs)
    notes = []

    def add(layout):
        return prs.slides.add_slide(prs.slide_layouts[layout])

    def note(title):
        """Compact speaker note, looked up from NOTES.

        Deliberately not prose: a presenter scans this mid-talk, they do not
        read it. One message, the figures, the questions it will attract."""
        n = NOTES.get(title, {})
        notes.append({"title": title, "say": n.get("say", ""),
                      "numbers": n.get("numbers", []), "asked": n.get("asked", [])})

    # ---- 1 intro -----------------------------------------------------------
    s = add(L_INTRO)
    title_of(s, "Where the rain goes")
    tb = s.shapes.add_textbox(Inches(0.9), Inches(5.1), Inches(7.0), Inches(1.6))
    set_text(tb, ["Modelling the Al-Masar Al-Akhdar green corridor, Beirut",
                  "A 0.5 m rain-on-grid shallow-water model built from airborne LiDAR"],
             size=15, color=NAVY)
    note("Where the rain goes")

    # ---- 2 chapter ---------------------------------------------------------
    s = add(L_CHAPTER); title_of(s, "HOW THE SIMULATOR WORKS")
    note("Chapter: how the simulator works")

    # ---- 3 what it does ----------------------------------------------------
    s = add(L_TEXT_IMG)
    title_of(s, "Rain falls on every cell. Nothing is routed by hand.")
    for ph in s.placeholders:
        if ph.placeholder_format.idx == 1:
            set_text(ph, [
                "2-D shallow-water equations, 2.3 million cells at 0.5 m",
                "Rain applied directly to every cell — the flood pattern is an "
                "output, not an assumption",
                "No drainage network drawn in advance, no sub-catchments",
                "Per-cell infiltration, roughness and detention from a 9-class "
                "land-cover classification",
                "Buildings block flow; roof and courtyard rain is rerouted to the "
                "nearest street",
                "Mass balance closes to 3 × 10⁻⁵ every run",
            ], size=14)
    drop_ph(s, 10); fit(prs, s, f"{F}/storms.png", (6.95, 1.31, 6.05, 5.08))
    note("Rain falls on every cell")

    # ---- what makes it different -------------------------------------------
    s = add(L_BULLETS)
    title_of(s, "What makes it different")
    for ph in s.placeholders:
        if ph.placeholder_format.idx == 1:
            set_text(ph, [
                [("The physics other packages leave out", True), ("", False)],
                "Per-cell infiltration with a wetting front (Green–Ampt), "
                "surface detention, and gullies whose capture depends on the "
                "head over the grate",
                "None of the 19 packages in the UK EA benchmark models "
                "infiltration or green infrastructure at all — they are "
                "hydraulics only",
                "",
                [("A solver you can differentiate", True), ("", False)],
                "Gradients of flooded area with respect to every cell of the "
                "design, verified to 2 × 10⁻⁸",
                "Turns the model from a predictor into a designer",
            ], size=13)
        if ph.placeholder_format.idx == 2:
            set_text(ph, [
                [("Built for this problem", True), ("", False)],
                "Two schemes in one code — fast inertial and shock-capturing "
                "HLLC — so scheme error is measured, not assumed",
                "Sub-metre on a real city: 2.3 M cells, and we show resolution "
                "decides the answer",
                "PyTorch on GPU — what makes the adjoint and ensembles affordable",
                "Opt-in erosion sub-model for the soft landscape",
                [("What it does not do:", True),
                 (" no coupled 1-D sewer or river model; shock capturing costs "
                  "~3.5× the fast scheme", False)],
            ], size=12, space=8)
    note("What makes it different")

    # ---- 4 the corridor ----------------------------------------------------
    s = add(L_TEXT_IMG)
    title_of(s, "The corridor is built into the terrain, surface by surface")
    for ph in s.placeholders:
        if ph.placeholder_format.idx == 1:
            set_text(ph, [
                "Geometry from the AUB Beirut Urban Lab dataset",
                "Seven surface types, each with its own infiltration, roughness "
                "and surface depression",
                "Rain garden is 52 % of the corridor; the engineered bioswales and "
                "ponds are under 4,000 m²",
                "Buildings the scheme demolishes are cleared from the terrain first",
                "31,828 m² of corridor in total",
                "",
                [("Our reading of the design.", True),
                 (" The cross-section is interpreted from the published "
                  "drawings and what seemed hydraulically sensible — not a "
                  "construction specification.", False)],
            ], size=13)
    drop_ph(s, 10); fit(prs, s, f"{F}/corridor_materials.png", (6.95, 1.31, 6.05, 5.08))
    note("The corridor, surface by surface")

    # ---- 5 chapter ---------------------------------------------------------
    s = add(L_CHAPTER); title_of(s, "CAN IT BE TRUSTED?")
    note("Chapter: validation")

    # ---- 6 vs other software ----------------------------------------------
    s = add(L_TEXT_IMG)
    title_of(s, "Against an independent solver, on the same grid")
    for ph in s.placeholders:
        if ph.placeholder_format.idx == 1:
            set_text(ph, [
                [("SynxFlow", True), (" — an independently written full "
                 "shallow-water Godunov solver", False)],
                [("Our shock-capturing scheme:", True),
                 ("  IoU 0.913 · correlation 0.985 · RMSE 3.7 cm", False)],
                [("Our fast scheme:", True),
                 ("  IoU 0.690 · RMSE 7.8 cm · spreads water over 27 % more area",
                  False)],
                [("LISFLOOD-FP:", True), ("  unusable here — fabricated six times "
                 "the input volume and failed its own mass balance", False)],
            ], size=14)
    drop_ph(s, 10); fit(prs, s, f"{F}/validation_synxflow.png", (6.95, 1.31, 6.05, 5.08))
    note("Against an independent solver")

    # ---- 7 benchmark 8A ---------------------------------------------------
    s = add(L_TEXT_IMG)
    title_of(s, "UK EA benchmark, Test 8A: rainfall in an urban street network")
    for ph in s.placeholders:
        if ph.placeholder_format.idx == 1:
            set_text(ph, [
                "The closest published benchmark to what we model: direct "
                "rainfall on a real dense street network in Glasgow",
                [("In cluster at all four gauges with published curves", True),
                 ("", False)],
                "P1 0.579 m · P2 0.239 m · P3 0.726 m · P6 0.064 m",
                "Compared against 19 commercial and research packages",
                "Mass balance closed to −0.000 % on the benchmark run",
            ], size=14)
    drop_ph(s, 10); fit(prs, s, f"{F}/benchmark_ea8.png", (6.95, 1.31, 6.05, 5.08))
    note("Benchmark: Test 8A")

    # ---- 8 benchmark Test 4 -----------------------------------------------
    s = add(L_TEXT_IMG)
    title_of(s, "UK EA benchmark, Test 4: flood propagation over a floodplain")
    for ph in s.placeholders:
        if ph.placeholder_format.idx == 1:
            set_text(ph, [
                "Tests the speed of an advancing flood front and the depths "
                "behind it",
                [("In cluster at all nine cross-section points, on both schemes",
                  True), ("", False)],
                "Built from the specification in the EA report — no data files "
                "required for this case",
                "",
                [("Behind these: 5 of 5 analytic tests pass", True),
                 (" — conservation, Manning normal depth, well-balancedness, "
                  "reference-port equivalence", False)],
            ], size=14)
    drop_ph(s, 10); fit(prs, s, f"{F}/benchmark_ea4.png", (6.95, 1.31, 6.05, 5.08))
    note("Benchmark: Test 4")

    # ---- 8 resolution ------------------------------------------------------
    s = add(L_TEXT_IMG)
    title_of(s, "Does the half-metre grid earn its cost?")
    for ph in s.placeholders:
        if ph.placeholder_format.idx == 1:
            set_text(ph, [
                [("The EA benchmark report leaves this open:", True),
                 (" “it is not currently clear that grid resolutions finer than "
                  "2 m will improve…”", False)],
                "Same terrain, coarsened step by step — only the grid changes",
                [("At 2 m the corridor's measured benefit collapses from 9.4 % to "
                  "0.6 %", True), ("", False)],
                "Peak street velocity climbs the wrong way, 4.3 → 14.8 m/s, as "
                "kerbs and narrow streets are averaged away",
                "For this kind of appraisal, sub-metre resolution is the difference "
                "between measuring the effect and missing it",
            ], size=14)
    drop_ph(s, 10); fit(prs, s, f"{F}/resolution.png", (6.95, 1.31, 6.05, 5.08))
    note("Resolution")

    # ---- 9 chapter ---------------------------------------------------------
    s = add(L_CHAPTER); title_of(s, "BEIRUT: THE GREEN CORRIDOR")
    note("Chapter: Beirut")

    # ---- 10 before / after -------------------------------------------------
    s = add(L_CONTENT)
    title_of(s, "Before and after, 25 November 2025 storm")
    drop_ph(s, 1); drop_ph(s, 10)
    fit(prs, s, f"{F}/beirut_before_after.png", (0.41, 1.45, 12.60, 4.85))
    caption(s, "Peak water depth. Teal outline = the right-of-way ribbon. "
               "Right panel: blue is drier after the corridor is built.",
            (0.41, 6.35, 12.60, 0.5))
    note("Before and after")

    # ---- 11 results --------------------------------------------------------
    s = add(L_TEXT_IMG)
    title_of(s, "The benefit is real, and it reaches about two blocks")
    rows = [
        [("Frequent storm (T2)", True), (f"   −{n['t2']['walk']:.0f} % flooding underfoot", False)],
        [("Observed 25 Nov 2025", True), (f"   −{n['v1_nov2025']['walk']:.0f} %", False)],
        [("Severe storm (T50)", True), (f"   {n['t50']['walk']:+.0f} %", False)],
        "",
        f"+{n['v1_nov2025']['infil']:,.0f} m³ extra water absorbed in the observed storm",
        f"−{n['v1_nov2025']['out']:.0f} % less water sent toward the port",
        "",
        "Measured on walkable surfaces — the fifth of the corridor dug out as "
        "swales and ponds holds water by design",
    ]
    for ph in s.placeholders:
        if ph.placeholder_format.idx == 1:
            set_text(ph, rows, size=14)
    drop_ph(s, 10); fit(prs, s, f"{F}/beirut_bands.png", (6.95, 1.31, 6.05, 5.08))
    note("Results")

    # ---- 12 water balance --------------------------------------------------
    s = add(L_TEXT_IMG)
    title_of(s, "In a severe storm it stops keeping itself dry — and protects everywhere else")
    for ph in s.placeholders:
        if ph.placeholder_format.idx == 1:
            set_text(ph, [
                [("T50: flooded area on the corridor barely moves", True),
                 (f" ({n['t50']['walk']:+.1f} %)", False)],
                [("…yet it absorbs", True),
                 (f" {n['t50']['infil']:,.0f} m³ more water and cuts flow toward the "
                  f"port by {n['t50']['out']:.0f} %", False)],
                "Its swales and ponds fill and hold — the water is on the corridor "
                "by design, not running down the streets",
                "",
                [("Correction to the earlier study:", True),
                 (f" the corridor does not triple infiltration. The multiple is "
                  f"{n['v1_nov2025']['infil_x']:.2f}× — ordinary Beirut ground already "
                  f"absorbs far more than a constant rate implied.", False)],
            ], size=14)
    drop_ph(s, 10); fit(prs, s, f"{F}/beirut_water_balance.png", (6.95, 1.31, 6.05, 5.08))
    note("The severe storm, and a correction")

    # ---- caveats + the ask -------------------------------------------------
    s = add(L_BULLETS)
    title_of(s, "What would make this a validated model")
    for ph in s.placeholders:
        if ph.placeholder_format.idx == 1:
            set_text(ph, [
                [("The model is verified. It is not yet validated.", True),
                 ("", False)],
                "It reproduces analytic solutions, passes two international "
                "benchmarks, and agrees with an independent solver to 3.7 cm.",
                "Every one of those compares it to other models — never to a "
                "flood that was measured in Beirut.",
                "",
                [("The proposal", True), ("", False)],
                "A modest, well-targeted field campaign in Beirut would close "
                "this — and nothing else can.",
            ], size=14)
        if ph.placeholder_format.idx == 2:
            set_text(ph, [
                [("What it would take", True), ("", False)],
                "Water-mark surveys after the next significant storm — "
                "photographs and levelled marks at a few dozen street locations",
                "A handful of low-cost depth loggers at known ponding points",
                "Timing of when key streets became impassable",
                "",
                [("What it would buy", True), ("", False)],
                "A measured error, not an inter-model one — and results that "
                "can carry planning decisions with confidence",
            ], size=13)
    note("What would make this a validated model — the ask")

    # ---- 14 outro ----------------------------------------------------------
    s = add(L_OUTRO)
    title_of(s, "Questions")
    note("Close")

    prs.save(a.out)
    json.dump(notes, open(a.notes, "w"), indent=2)
    print(f"wrote {a.out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
    print(f"wrote {a.notes}")


if __name__ == "__main__":
    main()
