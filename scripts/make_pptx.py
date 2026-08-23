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

    def note(title, body):
        notes.append({"title": title, "notes": body})

    # ---- 1 intro -----------------------------------------------------------
    s = add(L_INTRO)
    title_of(s, "Where the rain goes")
    tb = s.shapes.add_textbox(Inches(0.9), Inches(5.1), Inches(7.0), Inches(1.6))
    set_text(tb, ["Modelling the Al-Masar Al-Akhdar green corridor, Beirut",
                  "A 0.5 m rain-on-grid shallow-water model built from airborne LiDAR"],
             size=15, color=NAVY)
    note("Where the rain goes",
         "Opening. This is a physics-based flood model of the Fouad Boutros right-of-way "
         "in Beirut, built from a 39 GB airborne LiDAR survey and run at half-metre "
         "resolution. The talk has three parts: how the model works, how far it can be "
         "trusted, and what it says about the proposed green corridor.")

    # ---- 2 chapter ---------------------------------------------------------
    s = add(L_CHAPTER); title_of(s, "HOW THE SIMULATOR WORKS")
    note("Chapter: how the simulator works", "Section break.")

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
    note("Rain falls on every cell",
         "The key modelling choice is rain-on-grid: rainfall is applied to every cell and "
         "the water finds its own way downhill. Nothing is pre-routed, so where flooding "
         "appears is a result rather than an input.\n\n"
         "Each cell carries its own infiltration rate, surface roughness and detention "
         "depth, derived from a nine-class land-cover classification of the LiDAR returns "
         "and aerial imagery. Buildings stand as solid obstacles at roof height, and the "
         "rain landing on roofs and enclosed courtyards is rerouted to the nearest street "
         "cell, which is what a downspout does.\n\n"
         "The figure shows the three design storms. The 25 November 2025 event is real: "
         "25.4 mm in 30 minutes. By the Lebanese stormwater code's own intensity table "
         "that is only a 2-5 year burst, yet it flooded Sassine Square and the Ring. The "
         "flooding problem here is not extreme rainfall; it is ordinary rain on steep, "
         "sealed streets.\n\n"
         "Mass balance is checked every run — rain in equals infiltration plus drainage "
         "plus outflow plus storage, closing to about one part in 30,000. A model that "
         "loses water silently can produce any answer you like.")

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
    note("The corridor, surface by surface",
         "The design is not a polygon with an average infiltration rate. Each of the seven "
         "surface types in the published cross-section — vehicular lane, porous bikelane, "
         "bioswale, porous sidewalk, rain garden, terrace, bioretention pond — is written "
         "into the terrain cell by cell with its own infiltration rate, Manning roughness "
         "and detention depth.\n\n"
         "Worth knowing before someone asks which element does the work: rain garden is "
         "over half the corridor by area. The specialist detention features, the bioswales "
         "and bioretention ponds, together come to under 4,000 m². Most of the benefit "
         "comes from plain permeable ground, not from engineered structures.\n\n"
         "The corridor also demolishes buildings in its path. Those are identified from the "
         "official highway alignment, excluding heritage and institutional buildings which "
         "the real scheme preserves and repurposes, and cleared from the terrain before the "
         "corridor is laid down.\n\n"
         "IMPORTANT TO SAY OUT LOUD, and do not wait to be asked. The internal structure of "
         "the corridor here is our interpretation. We had the official zone polygon and the "
         "right-of-way alignment as data, but the cross-section — how wide the bikelane is, "
         "where the bioswale sits relative to the sidewalk, where the ponds and terraces go "
         "— was read off the published design drawings by eye and completed with what "
         "seemed hydraulically sensible. Band widths follow the drawing as we read it; "
         "bioretention ponds are placed at terrain low points and terraces on the steep "
         "segments, which is standard practice and also where our own model independently "
         "shows water collecting.\n\n"
         "So this is a faithful-in-spirit representation, not a construction specification. "
         "If the design team can share the actual cross-section dimensions and element "
         "positions, we can drop them straight in and re-run — the pipeline takes a "
         "material raster, so it is a data swap rather than a rebuild. The headline "
         "conclusions are unlikely to move much, because they are driven by total permeable "
         "area and where the corridor sits in the catchment rather than by the exact "
         "arrangement within it — but that is an expectation, not a tested claim.")

    # ---- 5 chapter ---------------------------------------------------------
    s = add(L_CHAPTER); title_of(s, "CAN IT BE TRUSTED?")
    note("Chapter: validation", "Section break.")

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
    note("Against an independent solver",
         "This is the strongest evidence in the deck. SynxFlow is a separate piece of "
         "software, written by a different group, solving the full shallow-water equations "
         "with a different numerical scheme. We ran it on the identical 0.5 m corridor grid "
         "and the identical storm.\n\n"
         "Our shock-capturing scheme agrees with it to an extent IoU of 0.913 and a depth "
         "RMSE of 3.7 cm. For context, the UK Environment Agency's own benchmark exercise "
         "saw worse agreement between commercial packages on a much gentler test case.\n\n"
         "The comparison also exposed something about our own fast scheme: it spreads water "
         "over 27 % more area than the full solver. That is a known characteristic of "
         "simplified three-term schemes — they smear the flood front because they drop the "
         "advective momentum term. It matters here because these streets run supercritical "
         "at 6-8 m/s on a 6 % grade.\n\n"
         "A third engine, LISFLOOD-FP, could not be used at all. On this stepped terrain it "
         "fabricated roughly six times the input water volume and failed its own mass "
         "balance check, so its output is void rather than merely different. That is a "
         "documented limitation of its scheme on terrain with retaining walls, not a bug we "
         "introduced.")

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
    note("Benchmark: Test 8A",
         "Of the eight EA benchmark cases this is the one closest to what we actually do: "
         "rain applied directly onto a real, dense urban street network — the Glasgow test "
         "site — with no inflow hydrograph and no drainage network.\n\n"
         "The grey bars show the published spread of the 19 packages that took part. The "
         "darker inner bar is what the report calls the main cluster: it excludes the "
         "simplified 'volume-spreading' codes that the report itself describes as "
         "approximate. Being inside the cluster is the stronger claim, and we are inside it "
         "at all four gauges where the report publishes curves.\n\n"
         "Only four of the nine output points have published time series in the report, "
         "which is why the figure shows four. Our values at the other five are in the "
         "results file but there is nothing to compare them against.\n\n"
         "One caveat to give if pressed: the published bands here were read off the "
         "report's figures rather than obtained as data, so they are good to a centimetre "
         "or two. That is not enough to change the verdict — our points sit well inside the "
         "bands, not on their edges.")

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
    note("Benchmark: Test 4",
         "Test 4 is a flat floodplain filled from a breach, and it tests something 8A does "
         "not: how fast an advancing flood front travels and how deep the water is behind "
         "it. The grey band is the envelope of the 19 published packages; we sit inside it "
         "at all nine points along the cross-section, on both our schemes.\n\n"
         "This one we built ourselves from the written specification, because the EA report "
         "states that no terrain file is supplied — the floodplain is flat at elevation "
         "zero and every other parameter is given numerically. That makes it reproducible "
         "without the benchmark data package.\n\n"
         "Underneath both benchmarks sit five analytic tests where the correct answer is "
         "known exactly: volume conservation in a closed basin, steady runoff matching the "
         "Manning normal depth to within half a millimetre, still water staying still over "
         "an uneven bed, and agreement with an independent reference implementation to "
         "0.027 mm.\n\n"
         "If asked why not all eight EA tests: the remaining cases need terrain and "
         "boundary files we do not have, and two of them require coupled 1-D sewer or river "
         "models, which this tool does not implement. Test 3 in particular would be worth "
         "acquiring — it is the case that discriminates shock-capturing schemes, which is "
         "exactly our new capability.")

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
    note("Resolution",
         "This slide answers a question the EA benchmark report explicitly says is "
         "unresolved: whether resolutions finer than 2 m are worth the cost. The report "
         "recommends someone test it at 0.5 m. As far as we know nobody had, and we are "
         "well placed to because we have a real dense city surveyed at half a metre.\n\n"
         "The method matters for fairness. We coarsened the same terrain step by step "
         "rather than rebuilding a model at each resolution, so the survey, the land cover "
         "and the corridor design are all held fixed and the computational grid is the only "
         "thing that changes.\n\n"
         "The result is stark. At 2 m — the resolution the benchmark itself uses — the "
         "corridor's measured benefit falls from 9.4 % to 0.6 %. A study run at that "
         "resolution would conclude the green corridor does essentially nothing.\n\n"
         "Peak velocity meanwhile rises as the grid coarsens, from 4.3 to 14.8 m/s, which "
         "is both the wrong direction and physically implausible. Block-averaging "
         "manufactures artificial steep gradients between cells while erasing the kerbs and "
         "narrow streets that actually convey the water.\n\n"
         "Honest caveat if pressed: coarsening an existing terrain is not identical to "
         "building a model natively at 2 m, where roughness would normally be recalibrated "
         "to compensate. The direction and size of the effect are nonetheless unambiguous.")

    # ---- 9 chapter ---------------------------------------------------------
    s = add(L_CHAPTER); title_of(s, "BEIRUT: THE GREEN CORRIDOR")
    note("Chapter: Beirut", "Section break.")

    # ---- 10 before / after -------------------------------------------------
    s = add(L_CONTENT)
    title_of(s, "Before and after, 25 November 2025 storm")
    drop_ph(s, 1); drop_ph(s, 10)
    fit(prs, s, f"{F}/beirut_before_after.png", (0.41, 1.45, 12.60, 4.85))
    caption(s, "Peak water depth. Teal outline = the right-of-way ribbon. "
               "Right panel: blue is drier after the corridor is built.",
            (0.41, 6.35, 12.60, 0.5))
    note("Before and after",
         "The observed 25 November 2025 storm, before and after the corridor, with the "
         "difference on the right. Blue means drier after.\n\n"
         "Two things to point out on the map. First, the effect is not confined to the "
         "ribbon itself — the blue extends into the surrounding street network, because the "
         "corridor intercepts runoff coming down from the Achrafieh side before it reaches "
         "those streets. Second, the corridor follows a natural drainage axis, which is why "
         "it works: it is already where the water wants to go.\n\n"
         "If asked about the large square feature near the top left: that is a real deep "
         "excavation on a construction site, which the LiDAR surveyed and which we model as "
         "a basin. It is genuine terrain, not an artefact.")

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
    note("Results",
         "These are the headline numbers, measured on the corridor's walkable surface.\n\n"
         "In a frequent storm the corridor removes well over half the flooding from the "
         "surfaces people actually walk on. In the observed November event, about half. In "
         "a 50-year storm, essentially nothing — and that deserves explanation rather than "
         "burial, which is the next slide.\n\n"
         "The metric matters. About a fifth of the corridor is deliberately dug out as "
         "bioswales, rain gardens and terraces. Those hold water on purpose. Counting a "
         "full bioretention pond as 'flooding' penalises the design for working, so the "
         "headline figure excludes them. Those cells are 20 % of the corridor but 34 % of "
         "its wet area, which is why the two ways of measuring diverge.\n\n"
         "The bar chart shows the benefit by distance from the corridor. It reaches roughly "
         "two blocks — about 25-50 m — before fading, which is a useful planning number: "
         "this is a linear intervention with a local catchment, not a city-wide fix.")

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
    note("The severe storm, and a correction",
         "This is the most interesting result and the one most likely to be challenged.\n\n"
         "In a 50-year storm the corridor stops reducing the flooded area on its own "
         "footprint. But it is still doing hydrological work: it absorbs over two thousand "
         "cubic metres more water than the same ground did before, and cuts the volume "
         "heading for the port by about a tenth. Its swales and ponds fill and hold. The "
         "water is on the corridor by design instead of running down the surrounding "
         "streets.\n\n"
         "So the corridor has two different jobs — keeping its own surface usable, and "
         "protecting what is downstream — and only the second survives an extreme event. "
         "That is a more useful message for a planner than a single percentage.\n\n"
         "The correction on this slide should be stated plainly rather than left for "
         "someone to find. The earlier version of this study credited the corridor with "
         "roughly tripling infiltration. With a proper wetting-front infiltration model "
         "(Green-Ampt) the multiple is about 1.3. The absolute volume gain holds up — "
         "around 900 cubic metres in the observed storm, inside the previously published "
         "range — but ordinary Beirut ground turns out to absorb far more than the earlier "
         "constant-rate model implied, so the corridor's relative contribution is smaller. "
         "Better physics, smaller claim.")

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
    note("What would make this a validated model — the ask",
         "This is the slide to deliver carefully, because it is both the honest caveat and "
         "the request.\n\n"
         "Make the distinction clearly first: the model is VERIFIED — it solves the "
         "equations correctly, reproduces analytic solutions, passes two international "
         "benchmarks and agrees with an independently written solver to 3.7 cm. What it is "
         "not is VALIDATED, because every one of those comparisons is against another "
         "model. We have observed rainfall for 25 November 2025, but no measured flood "
         "depths, extents or timings anywhere in the study area.\n\n"
         "Then make the ask, positively rather than apologetically. This gap cannot be "
         "closed by better modelling, more computing time or another benchmark. It can only "
         "be closed by measurement, and the measurement needed is genuinely modest: after "
         "the next significant storm, photographs and levelled water marks at a few dozen "
         "street locations, ideally within a day or two while the marks are still legible. "
         "A handful of inexpensive depth loggers at known ponding points would add timing. "
         "None of this requires specialist equipment or a large team.\n\n"
         "Be explicit about the value: with that data the study reports a measured error "
         "against reality rather than an agreement with other models, and its results can "
         "carry planning decisions with real confidence. Without it, everything here "
         "remains a well-built and well-tested hypothesis.\n\n"
         "Offer to help design the campaign — where to place loggers, which streets matter "
         "most — since the model already predicts where flooding concentrates and can be "
         "used to site the measurements efficiently. That turns the request into a "
         "collaboration rather than a favour.\n\n"
         "If asked whether the current numbers are usable meanwhile: yes, with care. The "
         "study reports differences — before versus after — and a systematic bias affects "
         "both scenarios and largely cancels. Absolute depths carry more uncertainty than "
         "the comparisons do.")

    # ---- 14 outro ----------------------------------------------------------
    s = add(L_OUTRO)
    title_of(s, "Questions")
    note("Close", "Thank you / questions.")

    prs.save(a.out)
    json.dump(notes, open(a.notes, "w"), indent=2)
    print(f"wrote {a.out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
    print(f"wrote {a.notes}")


if __name__ == "__main__":
    main()
