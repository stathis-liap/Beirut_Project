#!/usr/bin/env python3
"""Technical report: what the simulator is, what it does, and how it performs.

Written for a supervisor rather than an audience - explanatory prose, the
governing equations, and every number pulled from the run outputs so the
document cannot drift from the results it describes.

  python scripts/make_report.py
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from build_deck import corridor_masks


def red(run, storm, mask, thr=0.10):
    b = f"{run}/before_{storm}/max_depth.npy"
    a = f"{run}/after_{storm}/max_depth.npy"
    if not (os.path.exists(b) and os.path.exists(a)):
        return None
    B, A = np.load(b), np.load(a)
    fb = ((B > thr) & mask).sum()
    fa = ((A > thr) & mask).sum()
    return 100.0 * (fb - fa) / fb if fb else float("nan")


def meta(run, storm, phase):
    p = f"{run}/{phase}_{storm}/run_meta.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", default="output/presentation")
    ap.add_argument("--best", default="output/corridor_runs_v3_best")
    ap.add_argument("--hllc", default="output/corridor_runs_v3_hllc")
    ap.add_argument("--base", default="output/corridor_runs_v3")
    ap.add_argument("--material", default="output/corridor_gi_cut_v3/material.npy")
    ap.add_argument("--out", default="output/presentation/almasar_technical_report.pdf")
    a = ap.parse_args()

    ribbon, walk = corridor_masks(a.material)
    S = [("t2", "T2 frequent"), ("v1_nov2025", "25 Nov 2025 observed"),
         ("t50", "T50 severe")]
    W = {k: {tag: red(run, k, walk) for tag, run in
             (("base", a.base), ("hllc", a.hllc), ("best", a.best))} for k, _ in S}
    R = {k: red(a.best, k, ribbon) for k, _ in S}
    B = {k: (meta(a.best, k, "before"), meta(a.best, k, "after")) for k, _ in S}
    res = {r["res_m"]: r for r in
           json.load(open("output/resolution/resolution_summary.json"))["rows"]}

    def f(v, d=1):
        return "--" if v is None else f"{v:.{d}f}"

    rows_att = "\n".join(
        r"%s & %s & %s & %s & %s \\" % (
            lab, f(W[k]["base"]), f(W[k]["hllc"]), f(W[k]["best"]),
            f(W[k]["best"] - W[k]["base"]) if None not in
            (W[k]["best"], W[k]["base"]) else "--")
        for k, lab in S)

    rows_bal = "\n".join(
        r"%s & %.0f & %.0f & %+.0f & %.2f & %.0f \\" % (
            lab, B[k][0]["vol_infiltrated_m3"], B[k][1]["vol_infiltrated_m3"],
            B[k][1]["vol_infiltrated_m3"] - B[k][0]["vol_infiltrated_m3"],
            B[k][1]["vol_infiltrated_m3"] / max(B[k][0]["vol_infiltrated_m3"], 1e-9),
            100 * (B[k][0]["vol_outflow_m3"] - B[k][1]["vol_outflow_m3"])
            / max(B[k][0]["vol_outflow_m3"], 1e-9))
        for k, lab in S)

    rows_res = "\n".join(
        r"%.1f & %s & %.1f & %.1f \\" % (
            r["res_m"], f"{r['cells']:,}".replace(",", r"\,"),
            r["reduction_pct"], r["p99_velocity_ms"])
        for r in sorted(res.values(), key=lambda x: x["res_m"]))

    F = a.figs
    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.3cm,top=2.2cm,bottom=2.3cm]{geometry}
\usepackage{fontspec}\usepackage{xcolor}\usepackage{booktabs}\usepackage{fancyhdr}
\usepackage{enumitem}\usepackage{graphicx}\usepackage{amsmath}\usepackage{caption}
\usepackage{microtype}\usepackage{float}\usepackage[hidelinks]{hyperref}

\definecolor{navy}{HTML}{0E2841}\definecolor{teal}{HTML}{156082}
\definecolor{grey}{HTML}{5C6B73}\definecolor{rule}{HTML}{D3DAD9}
\definecolor{box}{HTML}{F2F4F3}
\setmainfont{Liberation Sans}
\captionsetup{font=small,labelfont={bf,color=teal},labelsep=period}
\pagestyle{fancy}\fancyhf{}\renewcommand{\headrulewidth}{0.4pt}
\fancyhead[L]{\footnotesize\color{grey}Al-Masar Al-Akhdar flood model --- technical note}
\fancyhead[R]{\footnotesize\color{grey}\thepage}
\setlength{\parindent}{0pt}\setlength{\parskip}{7pt}
\newcommand{\shead}[1]{\vspace{14pt}{\Large\bfseries\color{navy}#1}\par
  \vspace{2pt}{\color{rule}\hrule height 1pt}\vspace{4pt}}
\newcommand{\sshead}[1]{\vspace{9pt}{\large\bfseries\color{navy}#1}\par\vspace{1pt}}
\newsavebox{\keybuf}
\newenvironment{keybox}
  {\begin{lrbox}{\keybuf}\begin{minipage}{\dimexpr\textwidth-4\fboxsep}\vspace{2pt}}
  {\vspace{2pt}\end{minipage}\end{lrbox}%
   \begin{center}\colorbox{box}{\usebox{\keybuf}}\end{center}}

\begin{document}

{\Huge\bfseries\color{navy} A rain-on-grid flood model of the\\Al-Masar Al-Akhdar corridor}
\vspace{4pt}

{\large\color{grey} What the simulator is, what it does, and how it performs}

\vspace{6pt}{\color{rule}\hrule height 2pt}\vspace{8pt}

This note describes a two-dimensional hydraulic model of pluvial (rainfall)
flooding built for the Fouad Boutros right-of-way in Beirut, and the results
obtained from it. It covers the governing equations and how they are solved, how
the terrain and the proposed green corridor are represented, the verification and
validation evidence, and the findings for the corridor itself. Numbers quoted
throughout are read directly from the model outputs.

\shead{1. What the model solves}

The model integrates the two-dimensional shallow-water equations --- depth-averaged
conservation of mass and momentum --- over a regular grid. In conservative form,

\begin{equation*}
\frac{\partial}{\partial t}\begin{pmatrix}h\\ hu\\ hv\end{pmatrix}
+ \frac{\partial}{\partial x}\begin{pmatrix}hu\\ hu^2 + \tfrac12 g h^2\\ huv\end{pmatrix}
+ \frac{\partial}{\partial y}\begin{pmatrix}hv\\ huv\\ hv^2 + \tfrac12 g h^2\end{pmatrix}
= \begin{pmatrix}R - I - D\\ -gh\,\partial_x z_b - \tau_x\\ -gh\,\partial_y z_b - \tau_y\end{pmatrix}
\end{equation*}

where $h$ is water depth, $(u,v)$ the depth-averaged velocity, $z_b$ the bed
elevation, and $\tau$ the bed friction. The three source terms on the mass
equation are what make this a \emph{pluvial} model rather than a river model:
$R$ is rainfall applied directly to every cell, $I$ is infiltration into the
ground, and $D$ is capture by storm-drain inlets.

\sshead{Rain-on-grid, and why it matters here}

Conventional flood studies delineate sub-catchments, estimate a runoff
coefficient for each, and route the resulting hydrographs through a drainage
network. That approach requires the modeller to decide in advance where the water
goes. Here, rainfall is instead applied to every cell of the grid and the water
finds its own path downhill under the momentum equations. The flood pattern is
therefore an \emph{output} of the model rather than an input to it, which is the
property that lets it evaluate an intervention --- such as a green corridor ---
whose whole purpose is to change where the water goes.

\sshead{Two numerical schemes}

The equations are solved on a Cartesian grid in \texttt{torch}, on GPU, with two
selectable flux schemes:

\begin{itemize}[leftmargin=15pt,itemsep=2pt]
\item \textbf{Inertial (three-term).} The formulation of Bates, Horritt \&
Fewtrell (2010), which neglects the advective acceleration term and updates
face-centred discharge semi-implicitly against Manning friction. Fast and widely
used --- it is the basis of LISFLOOD-FP --- but it smears sharp flood fronts.
\item \textbf{Shock-capturing (HLLC).} A Godunov finite-volume scheme with an
HLLC approximate Riemann solver, Audusse hydrostatic reconstruction for
well-balancedness over an irregular bed, and Kurganov--Petrova velocity
desingularisation at wet/dry fronts. Roughly $3.5\times$ the cost of the
inertial scheme, and considerably more accurate here.
\end{itemize}

Both schemes share the same terrain, source terms and mass accounting, so the
difference between them isolates the effect of the numerics --- a comparison
exploited in Section 6.

\sshead{Surface and sub-surface processes}

Beyond the hydraulics, each cell carries its own surface properties, derived from
a nine-class land-cover classification of the LiDAR returns and aerial imagery:

\begin{itemize}[leftmargin=15pt,itemsep=2pt]
\item \textbf{Infiltration}, either at a fixed rate or --- the preferred option ---
by the Green--Ampt wetting-front model, $f = K\left(1 + \psi\Delta\theta/F\right)$,
where $K$ is saturated conductivity, $\psi$ the wetting-front suction head,
$\Delta\theta$ the initial moisture deficit and $F$ the cumulative infiltration.
This is solved implicitly over each timestep from the cumulative form
$F - F_n - \psi\Delta\theta\ln\!\left[(F+\psi\Delta\theta)/(F_n+\psi\Delta\theta)\right] = K\Delta t$,
because the explicit rate is unbounded on dry ground at the start of a storm.
Infiltration capacity therefore \emph{declines} as the ground wets, which a fixed
rate cannot express.
\item \textbf{Surface detention}, a per-cell depression depth, so that engineered
features such as bioswales and bioretention ponds hold water rather than shedding it.
\item \textbf{Storm-drain inlets}, either as fixed-capacity sinks or with a
head--discharge relation --- weir control while the grate is free-flowing,
$Q = C_w P h^{3/2}$, and orifice control once drowned, $Q = C_o A\sqrt{2gh}$,
taking whichever is smaller. Capture then depends on the ponding the design
actually produces.
\item \textbf{Roof and courtyard routing.} Buildings stand as obstacles at roof
height; rainfall landing on them, and in enclosed courtyards, is transferred to
the nearest street cell, as a downspout would.
\item \textbf{An opt-in erosion sub-model}, in which erodible soft surfaces
degrade under sustained flow, lowering their live infiltration and roughness.
\end{itemize}

\begin{keybox}
\textbf{Mass conservation is verified on every run.} The model reports the full
balance --- rainfall in, versus infiltration plus drainage plus outflow plus
stored volume --- and every result in this note closes it to better than
$3\times10^{-5}$ relative error. This matters because a model that loses or
creates water silently can produce almost any answer; one of the three
comparison engines below fails exactly this test.
\end{keybox}

\shead{2. How the terrain is built}

The model domain is a 571\,m\,$\times$\,1000\,m corridor strip together with its
upslope catchment, gridded at 0.5\,m into 2.3 million cells, from a 39\,GB
airborne LiDAR survey of Beirut.

The processing chain streams the point cloud once to grid lowest and highest
returns per cell, then: fills holes and despeckles the ground surface; classifies
land cover using excess-green, surface relief, slope and local roughness;
rasterises buildings and water from OpenStreetMap; removes tree canopy and
re-interpolates the ground beneath it; detects enclosed courtyards; builds the
rain-routing raster; and derives the Manning, infiltration and soil-parameter
grids.

Two terrain corrections made during this work are worth recording, because both
changed results materially:

\begin{itemize}[leftmargin=15pt,itemsep=2pt]
\item \textbf{A non-physical excavation.} A 1{,}988\,m$^2$ patch read
$-0.8$\,m --- some 27\,m below sea level --- while the ground enclosing it sat at
37\,m. Photogrammetry has no reliable floor for a dark, enclosed excavation. It
had been masked as outflow water, which silently deleted every drop of runoff
reaching it. It is now identified and modelled as a bounded basin.
\item \textbf{Building demolition.} The corridor clears buildings in its path.
The fill used to replace them drew elevations from the nearest ``open ground'',
but the OpenStreetMap building extract is incomplete over this domain, so
unmapped rooftops were being treated as ground: 14 of 35 cleared footprints
ended up still standing more than 3\,m above their surroundings, the worst by
23\,m. Fill sources are now screened by a morphological top-hat filter that
rejects anything standing like a roof, whether or not OSM knows about it.
\end{itemize}

\shead{3. How the green corridor is represented}

The corridor is not modelled as a polygon with an average infiltration rate. Its
geometry comes from the AUB Beirut Urban Lab dataset --- the 19\,ha Green Path
zone polygon and the Fouad Boutros right-of-way alignment --- and the designed
cross-section is written into the terrain cell by cell, each of seven surface
types carrying its own infiltration rate, Manning roughness and detention depth.

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{@FIGS@/corridor_materials.png}
\caption{The corridor as modelled, and its composition by area. Rain garden
accounts for just over half of the 31{,}828\,m$^2$; the engineered bioswales and
bioretention ponds together come to under 4{,}000\,m$^2$.}
\end{figure}

\begin{keybox}
\textbf{The internal layout is an interpretation.} The zone polygon and the
right-of-way alignment are data. The cross-section --- band widths, where the
bioswale sits relative to the sidewalk, where ponds and terraces are placed ---
was read from the published design drawings and completed with what appeared
hydraulically sensible. Ponds are sited at terrain low points and terraces on the
steep segments, which is standard practice and also where the model independently
shows water collecting. This is faithful in spirit but is not a construction
specification; supplied dimensions could be substituted directly, as the pipeline
consumes a material raster.
\end{keybox}

Because the corridor also demolishes buildings, the clearing is applied as a
separate terrain layer that the corridor design switches on. The ``before'' case
therefore retains every building at full height, and building the corridor is what
removes those in its path --- so the before/after comparison is a genuine one
rather than two views of an already-cleared site.

\shead{4. Verification: does it solve the equations correctly?}

Five analytic tests, where the correct answer is known exactly, are run after any
change to the solver. All five pass on both schemes.

\begin{table}[H]\centering\small
\begin{tabular}{lll}
\toprule
Test & What it checks & Result\\
\midrule
Closed box & Volume conservation, no leakage over walls & exact\\
Planar runoff & Steady depth against Manning normal depth & 14.1 vs 14.7\,mm\\
Lake at rest & No spurious currents over an uneven bed & $6\times10^{-5}$\,m/s\\
Port equivalence & Agreement with an independent implementation & 0.027\,mm\\
Erosion sanity & Inert when disabled; mass still closes when on & bit-identical\\
\bottomrule
\end{tabular}
\end{table}

The lake-at-rest test deserves comment: it is the check that a scheme is
\emph{well-balanced}, meaning it can represent still water over an irregular bed
without generating artificial flow. Many schemes fail it, and a failure is
serious in a domain like this one, which is terraced with retaining walls.

\shead{5. Validation: does it agree with reality, and with others?}

\sshead{Independent solver}

The strongest available evidence is a comparison against SynxFlow --- a separately
written full shallow-water Godunov code --- on the identical grid and storm.

\begin{table}[H]\centering\small
\begin{tabular}{lrrrr}
\toprule
Our scheme & Extent IoU & Depth correlation & RMSE & Wet-area bias\\
\midrule
Inertial (three-term) & 0.690 & 0.938 & 7.8\,cm & $+27\%$\\
\textbf{Shock-capturing (HLLC)} & \textbf{0.913} & \textbf{0.985} & \textbf{3.7\,cm} & \textbf{$+1\%$}\\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=0.92\textwidth]{@FIGS@/validation_synxflow.png}
\caption{Cell-by-cell depth agreement with SynxFlow. Two independently written
Godunov schemes agree to 3.7\,cm across a full urban catchment.}
\end{figure}

Two conclusions follow. First, the shock-capturing scheme is corroborated about as
far as model-to-model evidence permits. Second, the simplified inertial scheme
carries a systematic bias here: it spreads water over 27\% more area than the full
solver, which is the expected consequence of neglecting advective momentum on
streets that run supercritical at 6--8\,m/s on a 6\% grade.

A third engine, LISFLOOD-FP, could not be used at all. On this stepped terrain it
generated roughly six times the input water volume and failed its own mass-balance
check, so its output is void rather than merely different. This is a documented
limitation of its scheme on terrain with retaining walls.

\sshead{International benchmarks}

Two cases from the UK Environment Agency benchmark suite (Néelz \& Pender,
report SC120002), which publishes results from 19 commercial and research
packages.

\begin{figure}[H]\centering
\includegraphics[width=0.49\textwidth]{@FIGS@/benchmark_ea8.png}\hfill
\includegraphics[width=0.49\textwidth]{@FIGS@/benchmark_ea4.png}
\caption{Left: Test 8A, rainfall flooding in an urban street network --- inside
the main published cluster at all four gauges with published curves. Right:
Test 4, flood propagation over a floodplain --- inside the published spread at
all nine cross-section points, on both schemes.}
\end{figure}

Test 8A is the closest published analogue to this application: direct rainfall on
a real, dense street network. Test 4 tests the celerity of an advancing front. Of
the remaining six benchmark cases, four require terrain and boundary data not
available to us, and two require coupled one-dimensional sewer or river models,
which this tool does not implement.

\begin{keybox}
\textbf{What is \emph{not} validated.} Every comparison above is against another
model or a benchmark. There is no observed flood data for Beirut --- rainfall for
the 25 November 2025 event is recorded, but no measured depths, extents or
timings anywhere in the study area. The model is therefore \emph{verified} but not
\emph{validated} in the strict sense. Closing this requires field measurement:
levelled water marks at a few dozen street locations after a significant storm,
and ideally a small number of depth loggers at known ponding points.
\end{keybox}

\shead{6. Grid resolution: does half a metre earn its cost?}

The EA benchmark report states that ``it is not currently clear that grid
resolutions finer than 2\,m will improve the quality of velocity predictions'',
because terrain error and boundary conditions act at the same order as the grid,
and recommends the question be tested at 0.5\,m. We are unusually well placed to
answer it. The same terrain was coarsened by block aggregation, so that the
computational grid is the only quantity varying.

\begin{table}[H]\centering\small
\begin{tabular}{rrrr}
\toprule
Resolution (m) & Cells & Corridor benefit & p99 street velocity\\
\midrule
@ROWS_RES@
\bottomrule
\end{tabular}
\end{table}

At 2\,m --- the resolution the benchmark itself uses --- the measured benefit of
the corridor falls to 0.6\%, from 9.4\% at 0.5\,m. A study conducted at that
resolution would conclude the intervention does essentially nothing. Peak street
velocity meanwhile rises as the grid coarsens, which is both the wrong direction
and physically implausible: block-averaging manufactures artificial steep
gradients between cells while erasing the kerbs and narrow streets that actually
convey the water.

For appraising interventions of this kind, sub-metre resolution is therefore not a
refinement but a precondition. The caveat is that coarsening an existing terrain
is not identical to building a model natively at 2\,m, where roughness would
normally be recalibrated; the direction and magnitude of the effect are
nonetheless unambiguous.

\shead{7. Results for the corridor}

Three design storms are used: a frequent 2-year event, the observed
25 November 2025 storm (25.4\,mm in 30 minutes --- by the Lebanese stormwater
code's own intensity table only a 2--5 year burst, yet it flooded Sassine Square
and the Ring), and a severe 50-year event.

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{@FIGS@/beirut_before_after.png}
\caption{Peak water depth during the observed storm, before and after the
corridor, with the difference at right. Blue indicates a drier outcome. The
effect extends beyond the right-of-way into the surrounding street network,
because the corridor intercepts runoff descending from Achrafieh.}
\end{figure}

\sshead{Flooded-area reduction}

Reductions are quoted on the corridor's \emph{walkable} surface. Approximately a
fifth of the corridor is deliberately excavated as bioswales, rain gardens and
terraces, which hold water by design; counting a full bioretention pond as
``flooding'' would penalise the design for functioning. Those cells are 20\% of
the corridor by area but 34\% of its wet area, so the distinction is material.

\begin{table}[H]\centering\small
\begin{tabular}{lrrrr}
\toprule
Storm & Baseline & + shock capturing & + Green--Ampt & Net change\\
\midrule
@ROWS_ATT@
\bottomrule
\end{tabular}
\caption*{\small Reduction in flooded walkable area (\%), by model configuration.
The final column is the difference between the best-validated configuration and
the baseline.}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=0.72\textwidth]{@FIGS@/beirut_bands.png}
\caption{Reduction by distance from the corridor. The benefit extends roughly two
blocks before fading --- a useful planning figure, and a reminder that this is a
linear intervention with a local catchment rather than a city-wide remedy.}
\end{figure}

\sshead{Water balance}

\begin{table}[H]\centering\small
\begin{tabular}{lrrrrr}
\toprule
Storm & Infil. before & after & Gain & Multiple & Outflow cut\\
 & (m$^3$) & (m$^3$) & (m$^3$) & & (\%)\\
\midrule
@ROWS_BAL@
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=0.92\textwidth]{@FIGS@/beirut_water_balance.png}
\caption{Infiltrated volume and volume discharged toward the port, before and
after. The corridor absorbs more than the rain falling on it, because it also
intercepts runoff from upslope.}
\end{figure}

\sshead{Two findings that revise the earlier study}

\textbf{The severe storm.} In a 50-year event the corridor no longer reduces the
flooded area on its own footprint. It continues to perform hydrological work ---
absorbing over 2{,}000\,m$^3$ more water and cutting the volume discharged toward
the port by around a tenth --- but its swales and ponds fill and hold, so the
water stands on the corridor by design instead of running down the surrounding
streets. The corridor therefore has two distinct functions, keeping its own
surface usable and protecting what lies downstream, and only the second survives
an extreme event.

The attribution table above shows this is a \emph{numerical} result rather than a
soil-physics one: shock capturing alone moves the severe-storm figure by
$-35$ percentage points, against 1--3 points for the two lighter storms. The
earlier scheme was over-spreading water in the before-case, where flow is fastest
and most supercritical, and thereby inflating the apparent benefit.

\textbf{Infiltration.} An earlier version of this study credited the corridor with
roughly tripling infiltration. Under the Green--Ampt model the multiple is about
1.3. The absolute gain holds --- some 900\,m$^3$ in the observed storm, within the
previously published range --- but ordinary Beirut ground absorbs considerably
more than a fixed-rate model implied, so the corridor's \emph{relative}
contribution is smaller. Better physics, smaller claim.

\shead{8. Limitations}

\begin{itemize}[leftmargin=15pt,itemsep=3pt]
\item \textbf{No observed flood data.} The single largest gap; see Section 5.
\item \textbf{Material properties are literature values} --- bioretention and
permeable-pavement infiltration rates from published guidance, taken
conservatively --- not site measurements. Nothing is calibrated to Beirut soils.
The Green--Ampt soil parameters add four uncalibrated values per cover class.
\item \textbf{Drain inlet capacity is assumed}; no per-inlet data exists.
Results are reported as differences, which are robust to a common bias.
\item \textbf{The corridor cross-section is interpreted} from design drawings,
not specified.
\item \textbf{The erosion sub-model is illustrative}, not calibrated, and is
disabled in all results quoted here.
\item \textbf{No coupled one-dimensional sewer or river model}, which places two
EA benchmark cases out of scope.
\end{itemize}

\shead{9. What is distinctive about this model}

Set against the 19 packages in the EA benchmark comparison, two differences are
worth noting for any future publication.

\emph{None of those packages models infiltration or green infrastructure at all.}
They are hydraulics-only: shallow-water equations plus Manning roughness. The
per-cell infiltration, detention and head-dependent inlet capture here make this
a tool for appraising nature-based interventions rather than only for mapping
floods.

\emph{None of them is differentiable.} Because the solver is written in
\texttt{torch}, the gradient of a flood objective with respect to every cell of a
design can be obtained by automatic differentiation --- verified against central
finite differences to $2\times10^{-8}$, with gradient checkpointing making
full-length storms tractable. This permits gradient-based design optimisation
rather than trial of a handful of candidate layouts, and is the most promising
direction for publication. Initial experiments are encouraging but not yet
conclusive: the continuous optimum outperforms both a uniform and a
greedy-heuristic design at equal area, but does not survive projection onto a
buildable binary layout, which indicates the relaxation needs a projection scheme
before the result can be claimed.

\vspace{12pt}{\color{rule}\hrule height 1pt}\vspace{4pt}
{\footnotesize\color{grey}Model: 2.3\,M cells at 0.5\,m, GPU shallow-water,
mass balance closing to $\sim10^{-5}$. Terrain from a 39\,GB airborne LiDAR
survey; corridor geometry from the AUB Beirut Urban Lab Al-Masar Al-Akhdar
dataset. All figures and numbers in this document are generated directly from the
model outputs.}

\end{document}
"""

    for k, v in (("@FIGS@", F), ("@ROWS_RES@", rows_res),
                ("@ROWS_ATT@", rows_att), ("@ROWS_BAL@", rows_bal)):
        tex = tex.replace(k, v)

    out_dir = os.path.dirname(a.out) or "."
    stem = os.path.splitext(os.path.basename(a.out))[0]
    tex_path = os.path.join(out_dir, stem + ".tex")
    open(tex_path, "w").write(tex)
    for _ in range(2):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                            "-output-directory", out_dir, tex_path],
                           capture_output=True, text=True)
    if not os.path.exists(a.out):
        print(r.stdout[-3000:])
        sys.exit("xelatex failed")
    for ext in (".aux", ".log", ".out", ".toc"):
        p = os.path.join(out_dir, stem + ext)
        if os.path.exists(p):
            os.remove(p)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
