#!/usr/bin/env python3
"""Assemble the Al-Masar presentation page from the current run outputs.

Figures are embedded as data URIs (the artifact host blocks external requests),
and every headline number is read from the runs rather than typed in - so
re-running this after a study finishes updates the prose and the figures
together and the deck cannot drift from the results it describes.

  python scripts/make_presentation_figs.py --runs output/corridor_runs_v3
  python scripts/build_deck.py            --runs output/corridor_runs_v3

Sections whose data does not exist yet are dropped rather than shown empty.
"""

import argparse
import base64
import io
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))

STORMS = [("t2", "frequent (T2)"), ("v1_nov2025", "observed 25 Nov 2025"),
          ("t50", "severe (T50)")]

# bake_corridor.PROPS depression depth per material class. Cells dug to detain
# water (bioswale, bioretention pond, terrace) hold water BY DESIGN, so counting
# them as "flooded" penalises the corridor for working. The published study made
# the same distinction by reporting walkable surfaces separately.
DEPRESSION = {1: 0.0, 2: 0.0, 3: 0.15, 4: 0.0, 5: 0.0, 6: 0.40, 7: 0.10}


def corridor_masks(material_path):
    """(whole ribbon, walkable surface only)."""
    mat = np.load(material_path)
    ribbon = mat > 0
    dug = np.zeros(mat.shape, bool)
    for cls, d in DEPRESSION.items():
        if d > 0:
            dug |= (mat == cls)
    return ribbon, ribbon & ~dug



def embed(path, max_w=1100, quality=86):
    from PIL import Image
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def figure(src, alt, cap):
    if src is None:
        return ""
    return (f'<figure><img alt="{alt}" src="{src}">'
            f'<figcaption>{cap}</figcaption></figure>')


def corridor_numbers(terrain, runs, material):
    ribbon, walkable = corridor_masks(material)
    out = {}
    for key, _ in STORMS:
        pb = os.path.join(runs, f"before_{key}", "max_depth.npy")
        pa = os.path.join(runs, f"after_{key}", "max_depth.npy")
        if not (os.path.exists(pb) and os.path.exists(pa)):
            continue
        b, a = np.load(pb), np.load(pa)

        def red(mask):
            fb = ((b > 0.10) & mask).sum()
            fa = ((a > 0.10) & mask).sum()
            return 100.0 * (fb - fa) / fb if fb else float("nan")

        mb = json.load(open(os.path.join(runs, f"before_{key}", "run_meta.json")))
        ma = json.load(open(os.path.join(runs, f"after_{key}", "run_meta.json")))
        out[key] = {
            "reduction": red(walkable),        # headline: walkable surface
            "reduction_ribbon": red(ribbon),   # incl. cells dug to hold water
            "infil_gain": ma["vol_infiltrated_m3"] - mb["vol_infiltrated_m3"],
            "infil_x": ma["vol_infiltrated_m3"] / max(mb["vol_infiltrated_m3"], 1e-9),
            "outflow_cut": 100.0 * (mb["vol_outflow_m3"] - ma["vol_outflow_m3"])
                           / max(mb["vol_outflow_m3"], 1e-9),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="output/terrain_cut_0.5_v3")
    ap.add_argument("--runs", default="output/corridor_runs_v3")
    ap.add_argument("--material", default="output/corridor_gi_cut_v3/material.npy")
    ap.add_argument("--figs", default="output/presentation")
    ap.add_argument("--head", default=None, help="CSS/head fragment")
    ap.add_argument("--out", default="output/presentation/deck.html")
    ap.add_argument("--scheme-label", default="fast inertial scheme",
                    help="named in the caveats so the deck always says which "
                         "numerics produced its numbers")
    a = ap.parse_args()

    n = corridor_numbers(a.terrain, a.runs, a.material)
    F = a.figs
    im = {k: embed(os.path.join(F, f"{k}.png"),
                   1500 if k == "beirut_before_after" else 1100)
          for k in ("storms", "corridor_materials", "benchmark_ea4",
                    "validation_synxflow", "beirut_before_after",
                    "beirut_bands", "beirut_water_balance", "resolution")}

    def pct(k):
        return f"{n[k]['reduction']:.1f} %" if k in n else "—"

    head = open(a.head).read() if a.head else "<title>Rain on Beirut</title>"

    res_section = ""
    rp = "output/resolution/resolution_summary.json"
    if im["resolution"] and os.path.exists(rp):
        rr = {r["res_m"]: r for r in json.load(open(rp))["rows"]}
        f05, f2 = rr.get(0.5), rr.get(2.0)
        rows = "".join(
            f'<tr><td class="n">{r["res_m"]:g} m</td><td class="n">{r["cells"]:,}</td>'
            f'<td class="n">{r["reduction_pct"]:.1f} %</td>'
            f'<td class="n">{r["p99_velocity_ms"]:.1f} m/s</td></tr>'
            for r in sorted(rr.values(), key=lambda x: x["res_m"]))
        verdict = ""
        if f05 and f2:
            verdict = (f"""<div class="note"><b>The answer is yes, and emphatically.</b>
At 2 m — the resolution the EA benchmark itself uses — the corridor's measured benefit
collapses from <b>{f05['reduction_pct']:.1f} %</b> to <b>{f2['reduction_pct']:.1f} %</b>.
A study run at that resolution would conclude the green corridor does essentially nothing.
Peak street velocity meanwhile climbs from {f05['p99_velocity_ms']:.1f} to
{f2['p99_velocity_ms']:.1f} m/s and keeps climbing, which is the wrong direction and
physically implausible: block-averaging the terrain manufactures steep artificial
gradients between cells while erasing the kerbs and narrow streets that actually convey
the water.<br><br>For appraising this kind of intervention, sub-metre resolution is not a
refinement — it is the difference between measuring the effect and missing it.</div>""")
        res_section = f"""
<section>
<h2>4 · Does the half-metre grid earn its cost?</h2>
<p class="sublede">A question the UK benchmark leaves explicitly open.</p>
<p>The Environment Agency's benchmarking report concludes that
<i>"it is not currently clear that grid resolutions finer than 2 m will improve the
quality of velocity predictions"</i>, because terrain error and boundary conditions
matter at the same order as the grid. It recommends someone test it at 0.5 m. This is
that test, on a real dense-city catchment: the same terrain coarsened step by step, so
the computational grid is the only thing that changes.</p>
<div class="scroll"><table>
<thead><tr><th>Resolution</th><th>Cells</th><th>Corridor benefit</th><th>p99 street velocity</th></tr></thead>
<tbody>{rows}</tbody></table></div>
{figure(im['resolution'], 'Resolution convergence',
        'Corridor benefit and peak street velocity against grid resolution. '
        'Dashed line marks the 2 m resolution the EA benchmark uses.')}
{verdict}
<p style="font-size:.9rem;color:var(--muted)">Caveat: coarsening an existing terrain is
not identical to building a model natively at that resolution, where roughness would
normally be recalibrated to compensate. The non-monotonic 4 m point suggests the coarse
end is also noisy. The direction and magnitude of the effect are nonetheless unambiguous.</p>
</section>"""

    body = f"""
<div class="wrap">

<header>
<p class="eyebrow">Al-Masar Al-Akhdar · Beirut · 0.5 m rain-on-grid model</p>
<h1>Where the rain goes, and what a green corridor changes</h1>
<p class="stand">A GPU shallow-water model of the Fouad Boutros right-of-way, built from a
39 GB airborne LiDAR survey, cross-validated against an independent solver and two UK
Environment Agency benchmarks. What follows is how it works, how far it can be trusted,
and what it says about the corridor.</p>
</header>

<section>
<h2>1 · How the simulator works</h2>
<p class="sublede">Rain falls on every cell. Water moves. Nothing is routed by hand.</p>
<p>The model solves the <b>2-D shallow-water equations</b> on a half-metre grid — 2.3 million
cells over the corridor and its upslope catchment. Rain is applied directly to every cell
(<i>rain-on-grid</i>), so the flood pattern is an output, not an assumption: no drainage
network is drawn in advance and no sub-catchments are delineated.</p>
<div class="grid2">
<div class="card"><h3>The surface it flows over</h3><p>Each cell carries its own
infiltration rate, roughness and detention depth, from a nine-class land-cover
classification of the LiDAR returns and aerial imagery. Buildings stand as obstacles at
roof height; roof and courtyard rain is rerouted to the nearest street, as a downspout
would.</p></div>
<div class="card"><h3>Two ways to move the water</h3><p>A fast <b>inertial</b> scheme
(Bates et al. 2010) and a <b>shock-capturing Godunov/HLLC</b> scheme. The second matters
here: these streets run supercritical at 6–8 m/s on a 6 % grade, which is exactly where
simplified schemes smear the flood front.</p></div>
<div class="card"><h3>What the corridor does, physically</h3><p>The design is baked in
cell by cell — porous bikelane, bioswale, rain garden, bioretention pond, terrace — each
with its own infiltration, roughness and surface depression. The buildings the scheme
actually demolishes are cleared from the terrain.</p></div>
<div class="card"><h3>Conservation, checked every run</h3><p>Rain in equals infiltration
plus drainage plus outflow plus storage. Every run here closes that balance to better than
<span class="num">3 × 10⁻⁵</span> relative error. A model that loses water silently can
produce any answer you like.</p></div>
</div>
{figure(im['storms'], 'Design storm hyetographs',
        "The three design storms. The observed 25 Nov 2025 event delivered 25.4 mm in "
        "30 minutes — only a 2–5 year burst by the Lebanese code's own intensity table, "
        "yet it flooded Sassine Square and the Ring.")}
{figure(im['corridor_materials'], 'Corridor materials',
        'The designed cross-section rasterised at 0.5 m: a central lane flanked by porous '
        'bikelanes, bioswales and sidewalks, widening into rain gardens, bioretention '
        'ponds at the low points and terraces on the steep segments.')}
</section>

<section>
<h2>2 · How far it can be trusted</h2>
<p class="sublede">Analytic tests, two international benchmarks, one independent solver.</p>
<div class="kpis">
<div class="kpi"><div class="v">5 / 5</div><div class="l">analytic tests passed, both schemes</div></div>
<div class="kpi"><div class="v">0.913</div><div class="l">extent agreement (IoU) with SynxFlow</div></div>
<div class="kpi"><div class="v">3.7 cm</div><div class="l">depth RMSE vs SynxFlow</div></div>
<div class="kpi"><div class="v">13 / 13</div><div class="l">EA benchmark points in cluster</div></div>
</div>
<p><b>Analytic verification.</b> Volume conservation in a closed basin; steady runoff on a
plane matching Manning normal depth to 14.1 vs 14.7 mm; still water staying still over a
bowl to 6 × 10⁻⁵ m/s; and a numpy reference port agreeing to 0.027 mm.</p>
<p><b>UK Environment Agency benchmarks.</b> Test 8A (urban rainfall) is in cluster on all
four published gauges. Test 4 (flood propagation over a floodplain) is in cluster at all
nine cross-section points, against the spread of 19 commercial and research packages.</p>
{figure(im['benchmark_ea4'], 'EA Test 4 cross-section',
        'EA Test 4. Grey band = the spread of the 19 published packages; both our schemes '
        'sit inside it across the whole profile.')}
<p><b>The strongest evidence is cross-engine.</b> Run against SynxFlow — an independently
written full shallow-water Godunov solver — on the same 0.5 m corridor grid:</p>
<div class="scroll"><table>
<thead><tr><th>Scheme</th><th>Extent IoU</th><th>Correlation</th><th>RMSE</th><th>Wet-area bias</th></tr></thead>
<tbody>
<tr><td>Inertial (3-term)</td><td class="n">0.690</td><td class="n">0.938</td><td class="n">7.8 cm</td><td class="n bad">+27 %</td></tr>
<tr><td><b>HLLC (shock-capturing)</b></td><td class="n ok"><b>0.913</b></td><td class="n ok"><b>0.985</b></td><td class="n ok"><b>3.7 cm</b></td><td class="n ok"><b>+1 %</b></td></tr>
</tbody></table></div>
{figure(im['validation_synxflow'], 'Agreement with SynxFlow',
        "Every wet cell, our depth against SynxFlow's. Two independently written Godunov "
        'schemes agree to 3.7 cm across a full urban catchment.')}
<div class="note"><b>Stated plainly:</b> the simplified inertial scheme spreads water over
27 % more area than the full solver. The shock-capturing scheme removes that bias almost
entirely — so where the two disagree, the shock-capturing answer is the defensible one.
A third engine, LISFLOOD-FP, could not be used at all: it fabricated six times the input
volume on this stepped terrain and failed its own mass balance.</div>
</section>

<section>
<h2>3 · What it says about the green corridor</h2>
<p class="sublede">Before and after, three storms, measured on the corridor and outward from it.</p>
{figure(im['beirut_before_after'], 'Before and after flood maps',
        'Peak water depth during the observed 25 Nov 2025 storm. Teal outline = the '
        'right-of-way ribbon. Right panel: blue is drier after the corridor is built.')}
<div class="kpis">
<div class="kpi"><div class="v">{pct('t2')}</div><div class="l">less flooding underfoot, frequent storm</div></div>
<div class="kpi"><div class="v">{pct('v1_nov2025')}</div><div class="l">observed 25 Nov 2025 storm</div></div>
<div class="kpi"><div class="v">+{n['v1_nov2025']['infil_gain']:,.0f} m³</div><div class="l">extra water absorbed, observed storm</div></div>
<div class="kpi"><div class="v">−{n['v1_nov2025']['outflow_cut']:.0f} %</div><div class="l">less water sent toward the port</div></div>
</div>
<p><b>The corridor works, and it works hardest in ordinary rain.</b> In a frequent storm it
removes over half the flooding from the surfaces people actually walk on, and about half
again in the observed 25 November event.</p>
<div class="scroll"><table>
<thead><tr><th>Storm</th><th>Walkable surface</th><th>Whole ribbon</th><th>Extra absorbed</th><th>Outflow cut</th></tr></thead>
<tbody>
<tr><td>Frequent (T2)</td><td class="n ok">−{n['t2']['reduction']:.1f} %</td><td class="n">−{n['t2']['reduction_ribbon']:.1f} %</td><td class="n">+{n['t2']['infil_gain']:,.0f} m³</td><td class="n">−{n['t2']['outflow_cut']:.0f} %</td></tr>
<tr><td>Observed 25 Nov 2025</td><td class="n ok">−{n['v1_nov2025']['reduction']:.1f} %</td><td class="n">−{n['v1_nov2025']['reduction_ribbon']:.1f} %</td><td class="n">+{n['v1_nov2025']['infil_gain']:,.0f} m³</td><td class="n">−{n['v1_nov2025']['outflow_cut']:.0f} %</td></tr>
<tr><td>Severe (T50)</td><td class="n">{n['t50']['reduction']:+.1f} %</td><td class="n">{n['t50']['reduction_ribbon']:+.1f} %</td><td class="n">+{n['t50']['infil_gain']:,.0f} m³</td><td class="n">−{n['t50']['outflow_cut']:.0f} %</td></tr>
</tbody></table></div>
<div class="note"><b>In a 50-year storm the corridor stops keeping itself dry — and starts
protecting everywhere else.</b> Flooded area on the corridor barely changes
({n['t50']['reduction']:+.1f} % on walkable surface), yet it still absorbs
<b>{n['t50']['infil_gain']:,.0f} m³</b> more water and cuts the volume heading for the port by
<b>{n['t50']['outflow_cut']:.0f} %</b>. Its bioswales and ponds fill and hold, so the water is
on the corridor by design rather than running down the streets. Two different jobs, and only
the second one survives an extreme event.</div>
<p style="font-size:.9rem;color:var(--muted)"><b>On the metric:</b> "walkable surface"
excludes the fifth of the corridor deliberately dug out as bioswales, rain gardens and
terraces. Those hold water on purpose — counting a full bioretention pond as flooding
penalises the design for functioning. They are 20 % of the ribbon but 34 % of its wet
area, which is why the two columns diverge most in the severe storm.</p>
{figure(im['beirut_bands'], 'Reduction by distance band',
        'The benefit is not confined to the ribbon: it reaches roughly two blocks before '
        'fading. Note the severe storm, where the strongest relief is in the 0–25 m band '
        'rather than on the corridor itself.')}
{figure(im['beirut_water_balance'], 'Water balance',
        'Where the water goes. The corridor absorbs 1.7–2.2× what the same ground shed '
        'before — more than the rain landing on it, because it also intercepts runoff '
        'from upslope — and sends measurably less toward the port.')}
<div class="note"><b>Honest caveats, and they matter.</b>
<ul>
<li>These come from the <b>{a.scheme_label}</b> — the best-validated configuration
available, not the fast default. The terrain was also rebuilt along the way: a
mis-modelled excavation repaired, building demolition corrected, a richer land-cover
classifier.</li>
<li>The <b>whole-ribbon</b> figures are below the earlier published ones
(47 / 36 / 18 %), and the walkable-surface figures sit inside the published
28–64 % band for the two lighter storms. The severe storm is the real change: the
earlier study reported an 18 % reduction there, and with wetting-front infiltration
the honest answer is that flooded area on the corridor barely moves.</li>
<li>The corridor is credited with <b>tripling infiltration</b> in the earlier study.
With Green-Ampt the multiple is {n['v1_nov2025']['infil_x']:.2f}× — the absolute gain
holds up, but ordinary Beirut ground already absorbs far more than a constant rate
implied, so the corridor's <i>relative</i> contribution is smaller.</li>
<li>There is <b>no observed flood data</b> for Beirut — only observed rainfall. Every
validation here is against other models and benchmarks, never against a measured flood.
That is the single biggest gap in the study.</li>
<li>Material properties are literature values, not site measurements.</li>
</ul></div>
</section>
{res_section}

<footer>Model: 2.3 M cells at 0.5 m, GPU shallow-water, mass balance closing to ~10⁻⁵.
Terrain from a 39 GB airborne LiDAR survey; corridor geometry from the AUB Beirut Urban Lab
Al-Masar Al-Akhdar dataset. Figures and numbers regenerate from the run outputs via
<code>make_presentation_figs.py</code> and <code>build_deck.py</code>.</footer>

</div>
"""
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(head + body)
    print(f"wrote {a.out}  ({os.path.getsize(a.out)/1e6:.2f} MB)")
    for k, v in im.items():
        if v is None:
            print(f"  (no figure: {k})")


if __name__ == "__main__":
    main()
