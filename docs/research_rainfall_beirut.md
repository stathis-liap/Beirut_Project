# Research: Beirut rainfall statistics & documented floods (2026-07-12)

Feeds PLAN.md Phase C (forcing) and D6 (ground truth). Compiled from web
research; all URLs verified at compile time.

## 1. Design-storm IDF table for Beirut

**Primary source:** *Stormwater Network Code in Lebanon* (Sayed Ahmad &
Swaydan, Nov 2022, EU HAWKAMAA programme) — "Table 2: IDF curves in
Beirut", the only published tabulated Beirut IDF found. Local copy:
`docs/stormwater_code_lebanon.pdf`. Source:
https://www.pseau.org/outils/ouvrages/hawkamaa_eu_stormwater_network_code_in_lebanon_2023.pdf

Intensities in **mm/h**:

| Duration | 2 yr | 5 yr | 10 yr | 25 yr* | 50 yr | 100 yr |
|---|---|---|---|---|---|---|
| 10 min | 74.1 | 98.7 | 118.4 | 141.9* | 159.3 | 181.0 |
| 20 min | 53.3 | 70.0 | 84.9 | 102.4* | 115.4 | 131.2 |
| 30 min | 41.9 | 55.0 | 66.8 | 81.0* | 91.6 | 102.2 |
| 60 min | 25.9 | 33.3 | 41.7 | 51.0* | 57.9 | 64.5 |
| 90 min | 20.0 | 25.4 | 31.7 | 38.2* | 43.0 | 48.8 |
| 120 min | 17.8 | 23.0 | 27.4 | 32.0* | 35.3 | 40.3 |

\* 25-yr interpolated on the Gumbel reduced variate between published
10-yr and 50-yr — state this in the report. The code does not document
the derivation (gauge, record length), hence the cross-check below.

**Cross-check (peer-reviewed, 2026):** Dargham & Andraos, *Frontiers in
Water* — GPM-IMERG 1998–2025 disaggregated at Beirut Airport, Gumbel/MLE.
Quantiles computed from their published Gumbel parameters (mm/h):
10 min: 66.2 / 107.1 / 127.7 / 143.0 for 2/10/25/50-yr;
1 h: 20.1 / 32.5 / 38.7 / 43.3; 2 h: 12.6 / 20.4 / 24.4 / 27.3.
Agrees with the code table within ~10–25 % (satellite lower, expected —
IMERG smooths point extremes). Use the code table as design basis
(conservative), cite Frontiers as independent confirmation.
https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2026.1727182/full

Reality anchor: 12 Dec 2023 airport storm delivered **47 mm in 20 min
(= 141 mm/h)** — above the table's 100-yr 20-min value; the table is
adequate-to-slightly-low for rare events.

Official Météo Liban gauge-based IDF is not openly published — say so in
the report.

## 2. Annual pattern

~715–850 mm/yr (commonly ~730), ~62 rain days; Dec–Feb wettest (Jan
~154–191 mm), Jun–Aug ≈ 0; 80–90 % falls Oct–Apr. Storm character:
Mediterranean cyclonic winters with short intense convective bursts
(observed: 47 mm/20 min Dec 2023; 22.2 mm/15 min ≈ 89 mm/h Nov 2025).
⇒ simulate 30–60 min design bursts, not long-duration storms, for this
small steep coastal catchment.

## 3. Documented flood events (validation cases)

| Date | Where / what | Numbers | Sources |
|---|---|---|---|
| **25 Nov 2025** | **Achrafieh (Sassine Sq) + Ring bridge flooded**; minister: drains "clogged within minutes" by garbage | **25.4 mm/30 min; 22.2 mm/15 min (≈89 mm/h)** — only a ~2–5-yr burst per the IDF table | L'Orient Today (today.lorientlejour.com/article/1486218), NNA |
| 22–23 Dec 2023 | Karantina + northern entrance flooded; **Beirut River overflowed** (bounds our domain east) | — | Kataeb, Spectee |
| 12 Dec 2023 | Airport area, tunnel, roads; drains overwhelmed + waste accumulation | 47 mm/20 min | Arab News (arabnews.com/node/2424711) |
| 11 Nov 2024 | First winter storm: Beirut streets, airport tunnel, Sanayeh | — | Anadolu |
| 9–10 Dec 2019 | City paralyzed; "streets turned into rivers"; ~1.5 m in Jnah/Ouzai; "50-year-old infrastructure" | — | Arab News, Middle East Eye, FloodList |
| 6–9 Jan 2019 | Storm Norma: major underpasses ponded | — | ReliefWeb |

No news item names the Vendôme / St-Nicolas / Massad stairs specifically;
nearest documented anchors are Sassine, the Ring, Karantina/port, and the
river. State that in the report; ask BUL to annotate local hotspots.
ThinkHazard rates Beirut urban flood "low" from global data — useful
contrast between global screening and documented reality.

**Primary validation case: replay 25 Nov 2025** (hyetograph 25.4 mm/30 min
with a 22.2 mm/15 min peak, drains clogged) → should reproduce ponding at
Sassine-adjacent low points and the Ring.

## 4. Drainage assumptions (no measured data exists)

- Lebanese code: minimum design storm **10-yr**; Rational method < 80 ha;
  Kirpich tc (stormwater code, local PDF).
- Beirut largely **combined sewers**, sludge-hardened, overflow when it
  rains heavily (New Arab explainer; UNDP SOER 2020).
- Infrastructure ~50+ yr old; annual pre-winter drain-clearing campaigns;
  chronic inlet blockage by litter.
- Beirut River channelized 1968, functions as combined conveyance.
- **Simulation assumptions to state:** functioning network intercepts
  ~2–10-yr intensity (code 10-yr; aged network realistically ≤ 2–5-yr);
  baseline = 50–100 % inlet blockage (reproduces Nov 2025 behavior). No
  published per-inlet capacity (L/s) exists — treat as scenario range.

## 5. Climate uplift

No official Lebanese factor. Mediterranean: historical 100-yr daily
extremes underestimate 21st-c magnitudes by ~20–30 % (Zittis et al. 2021,
W&CE); sub-hourly scaling ≈ Clausius-Clapeyron 7 %/°C, super-CC possible
for convective events (Lenderink 2021 GRL; Nat. Geosci. 2025).
**Use: +10–15 % on short-duration intensities for 2050; +20–30 %
end-century/high-emissions; cite Zittis et al.**

Full link list: see agent transcript / PLAN history. Key: HAWKAMAA code
PDF (local copy in docs/), Frontiers in Water 2026 IDF paper, L'Orient
Today 2025-11-25, Arab News 2023-12-12 & 2019-12-10, UNDP SOER 2020,
Zittis et al. 2021 (PMC8686183).
