#!/usr/bin/env python3
"""Speaker transcript PDF for the Al-Masar deck, from slides.json + the run outputs.

Built from the same slides.json that make_pptx.py emits, so the transcript can
never describe a slide the deck no longer has. Numbers in the reference tables
are read from the runs for the same reason.

  python scripts/make_pptx.py            # writes slides.json
  python scripts/make_transcript.py      # writes the PDF
"""

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from build_deck import corridor_masks, STORMS


def tex_escape(s):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    # keep the typographic characters the notes actually use
    return (s.replace("—", "---").replace("–", "--").replace("×", r"$\times$")
             .replace("⁻", "-").replace("²", r"$^2$").replace("³", r"$^3$")
             .replace("“", "``").replace("”", "''").replace("’", "'")
             .replace("−", "-").replace("·", r"$\cdot$"))


def corridor_table(runs, material):
    ribbon, walkable = corridor_masks(material)
    rows = []
    for key, lab in STORMS:
        pb, pa = f"{runs}/before_{key}", f"{runs}/after_{key}"
        if not os.path.exists(f"{pa}/max_depth.npy"):
            continue
        b, a = np.load(f"{pb}/max_depth.npy"), np.load(f"{pa}/max_depth.npy")
        def red(mask):
            fb = ((b > 0.10) & mask).sum(); fa = ((a > 0.10) & mask).sum()
            return 100.0 * (fb - fa) / fb if fb else float("nan")
        mb = json.load(open(f"{pb}/run_meta.json"))
        ma = json.load(open(f"{pa}/run_meta.json"))
        rows.append((lab, red(walkable), red(ribbon),
                     ma["vol_infiltrated_m3"] - mb["vol_infiltrated_m3"],
                     100 * (mb["vol_outflow_m3"] - ma["vol_outflow_m3"])
                     / max(mb["vol_outflow_m3"], 1e-9)))
    return rows


HEAD = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.4cm,top=2.2cm,bottom=2.4cm]{geometry}
\usepackage{fontspec}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{fancyhdr}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}

\definecolor{navy}{HTML}{0E2841}
\definecolor{teal}{HTML}{156082}
\definecolor{grey}{HTML}{5C6B73}
\definecolor{rule}{HTML}{D3DAD9}

% Nunito (the template's face) is not installed here; Liberation Sans is the
% closest available humanist sans and keeps the notes visually adjacent to the
% deck without pretending to match it.
\setmainfont{Liberation Sans}
\newfontfamily\dispfont{Liberation Sans}

\pagestyle{fancy}\fancyhf{}
\renewcommand{\headrulewidth}{0.4pt}
\fancyhead[L]{\footnotesize\color{grey}Al-Masar Al-Akhdar --- speaker notes}
\fancyhead[R]{\footnotesize\color{grey}\thepage}
\setlength{\parindent}{0pt}
\setlength{\parskip}{7pt}

\newcommand{\slidehdr}[2]{%
  \vspace{16pt}%
  {\color{teal}\dispfont\footnotesize SLIDE #1}\par\vspace{1pt}%
  {\dispfont\Large\bfseries\color{navy}#2}\par\vspace{4pt}%
  {\color{rule}\hrule height 1pt}\vspace{2pt}}

\begin{document}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="output/presentation/slides.json")
    ap.add_argument("--runs", default="output/corridor_runs_v3_best")
    ap.add_argument("--material", default="output/corridor_gi_cut_v3/material.npy")
    ap.add_argument("--out", default="output/presentation/almasar_speaker_notes.pdf")
    a = ap.parse_args()

    slides = json.load(open(a.notes))
    rows = corridor_table(a.runs, a.material)

    body = [HEAD]
    body.append(r"""
{\dispfont\Huge\bfseries\color{navy} Where the rain goes}\\[2pt]
{\large\color{grey} Speaker notes --- Al-Masar Al-Akhdar green corridor, Beirut}

\vspace{4pt}{\color{rule}\hrule height 2pt}\vspace{10pt}

Built to be scanned, not read aloud. Each slide gives one \textbf{Say} line --- the
single message --- then the figures to have ready, then the questions that slide
reliably attracts. The \emph{Key numbers} page at the end carries every number quoted
on the slides.
""")

    for i, sl in enumerate(slides, 1):
        body.append(r"\slidehdr{%d}{%s}" % (i, tex_escape(sl["title"])))
        if sl.get("say"):
            body.append(r"{\color{teal}\bfseries Say:} %s" % tex_escape(sl["say"]))
        if sl.get("numbers"):
            body.append(r"\vspace{-3pt}\begin{itemize}[leftmargin=14pt,itemsep=1pt,"
                        r"topsep=2pt,parsep=0pt]")
            for nline in sl["numbers"]:
                body.append(r"\item %s" % tex_escape(nline))
            body.append(r"\end{itemize}")
        for q, ans in sl.get("asked", []):
            body.append(r"\vspace{-2pt}{\color{grey}\itshape %s} \\ %s"
                        % (tex_escape(q), tex_escape(ans)))

    # ---- reference tables --------------------------------------------------
    body.append(r"\clearpage")
    body.append(r"\slidehdr{REF}{Key numbers}")
    body.append(r"""
\textbf{Corridor results} --- flooded-area reduction, best-validated configuration
(shock-capturing scheme with Green--Ampt infiltration). ``Walkable'' excludes the
fifth of the corridor dug out as swales and ponds, which hold water by design.

\vspace{4pt}
\begin{tabular}{lrrrr}
\toprule
Storm & Walkable & Whole ribbon & Extra absorbed & Outflow cut\\
\midrule""")
    for lab, w, r_, inf, out in rows:
        body.append(r"%s & %.1f\%% & %.1f\%% & %+,.0f m$^3$ & %.0f\%%\\"
                    .replace(",", "") % (tex_escape(lab), w, r_, inf, out))
    body.append(r"""\bottomrule
\end{tabular}

\vspace{10pt}
\textbf{Validation} --- against SynxFlow, an independently written full shallow-water
solver, on the identical 0.5\,m grid and storm.

\vspace{4pt}
\begin{tabular}{lrrrr}
\toprule
Scheme & Extent IoU & Correlation & RMSE & Wet-area bias\\
\midrule
Fast inertial (3-term) & 0.690 & 0.938 & 7.8 cm & +27\%\\
\textbf{Shock-capturing (HLLC)} & \textbf{0.913} & \textbf{0.985} & \textbf{3.7 cm} & \textbf{+1\%}\\
\bottomrule
\end{tabular}

\vspace{10pt}
\textbf{Grid resolution} --- same terrain, coarsened; only the computational grid varies.

\vspace{4pt}
\begin{tabular}{lrrr}
\toprule
Resolution & Cells & Corridor benefit & p99 street velocity\\
\midrule""")
    rp = "output/resolution/resolution_summary.json"
    if os.path.exists(rp):
        for r in sorted(json.load(open(rp))["rows"], key=lambda x: x["res_m"]):
            body.append(r"%.1f m & %s & %.1f\%% & %.1f m/s\\" %
                        (r["res_m"], f"{r['cells']:,}".replace(",", "\\,"),
                         r["reduction_pct"], r["p99_velocity_ms"]))
    body.append(r"""\bottomrule
\end{tabular}

\vspace{14pt}
{\color{rule}\hrule height 1pt}\vspace{6pt}
\textbf{If you are asked something not covered here}, the safe answer is that the model
is verified against analytic solutions and agrees closely with an independent solver,
but has never been tested against a measured flood in Beirut, because no such
measurement exists. Say that rather than improvise.
\end{document}""")

    tex = "\n\n".join(body)
    out_dir = os.path.dirname(a.out) or "."
    stem = os.path.splitext(os.path.basename(a.out))[0]
    tex_path = os.path.join(out_dir, stem + ".tex")
    open(tex_path, "w").write(tex)
    for _ in range(2):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                            "-output-directory", out_dir, tex_path],
                           capture_output=True, text=True)
    if not os.path.exists(a.out):
        print(r.stdout[-2500:])
        sys.exit("xelatex failed")
    for ext in (".aux", ".log", ".out"):
        p = os.path.join(out_dir, stem + ext)
        if os.path.exists(p):
            os.remove(p)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
