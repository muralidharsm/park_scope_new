# PARK-SCOPE 🛞

**Parking Risk & Congestion Optimization for Smart Patrol Enforcement**
Bengaluru Traffic Police × Flipkart — Gridlock Hackathon 2026 · Theme 1 (Poor Visibility on Parking-Induced Congestion)

PARK-SCOPE turns the Bengaluru Traffic Police's own parking-violation ticketing
data into a daily **patrol deployment plan**. It answers the question BTP says it
cannot answer today: *given limited manpower, where and when should patrols be
deployed to suppress the most traffic-choking parking violations?*

---

## Why it matters (from the data)

Analysed across **298,445** parking-violation records (Nov 2023 – Apr 2024),
169 monitored junctions, 54 police stations:

| Concentration | Share of all violations |
|---|---|
| Top **1%** of ~111m zones (≈78 zones) | **33%** |
| Top **5%** of zones | **64%** |
| Top **10%** of zones | **77%** |
| Top 8 named junctions | **45%** of junction-tagged tickets |

Enforcement today is even and reactive; the violations are not. That gap is the
entire opportunity.

---

## What it does — three layers

1. **Hotspot Intelligence** — spatiotemporal heatmap of where/when violations
   peak, filterable by day of week and shift window.
2. **Congestion Impact Score** — ranks zones by *impact*, not raw count, by
   fusing the parking data with the **Astram event dataset** (congestion,
   construction, water-logging, accidents, processions, road-closures near each
   zone) plus junction criticality and strategic proximity (metro / market /
   hospital / school / bus stop). Weights are transparent and shown in-app.
3. **Patrol Route Optimizer** — a two-stage engine. First a weighted
   maximum-coverage *selection* picks the highest impact-weighted zones for the
   shift; then a *routing* stage treats junctions as graph nodes and drive-times
   as edges (haversine ÷ avg patrol speed) and solves a Travelling-Salesperson
   beat with nearest-neighbour + 2-opt. Output: a single drivable patrol route
   with stop sequence, per-leg and cumulative drive time, total distance, and the
   **% of impact-weighted violation load** it covers — drawn live on the map.

A forward-in-time validation (gradient boosting vs. a naive baseline) confirms
the spatial-temporal patterns are learnable, not noise.

### Command-console upgrades

- **Patrol router** — the optimizer's chosen stops are sequenced into an actual
  drive order with a TSP heuristic (nearest-neighbour + 2-opt) and drawn as a
  route on the map. Edge weight is straight-line distance (a labelled proxy;
  drop in a drive-time matrix for production).
- **Explainable AI** — the Forecast tab exposes the gradient-boosting model's
  `feature_importances_`, showing *what* it leans on (location dominates).
- **Live e-challan feed** — a sidebar panel streams simulated incoming
  violations every few seconds (via `st.fragment`), mimicking the Astram live
  environment without blocking the app.
- **Deep-dive tooltips** — hovering any zone reveals its peak window and
  violation mix (e.g. "Wrong Parking 84% · No Parking 10%").
- **Dark Inferno map** — `carto-darkmatter` base with an Inferno colour scale so
  hotspots punch through the dark console.

### Command-center features

- **Dark operational map** — Inferno colour scale on a `carto-darkmatter`
  basemap so critical clusters punch through; toggle between impact bubbles and
  a violation-density heat layer.
- **Explainable AI (XAI)** — the GBM's `feature_importances_` are surfaced as a
  chart on the Forecast tab, showing the model leans hardest on *location*, with
  day-of-week and seasonality secondary.
- **Live e-challan feed** — an auto-refreshing panel (`st.fragment(run_every=3)`)
  simulates an ASTraM-style stream of incoming violations.
- **Deep-dive tooltips** — hovering any zone reveals its peak window and
  violation-type mix (e.g. *Wrong Parking 84% · No Parking 10%*).

---

## Architecture

```
 raw CSVs (data/)
        │   src/prepare_data.py  ──────────────────────────────────────┐
        ▼                                                               │
 clean + geocode + IST time + ~111m grid                                │
        ├─► src/impact.py     Congestion Impact Score (fuses event data)│
        ├─► src/forecast.py   hour×weekday rate table + GBM validation  │
        └─► small artifacts (artifacts/*.csv, meta.json)  ◄─────────────┘
        ▼
 app.py (Streamlit)  ──►  Hotspot Map · Impact Leaderboard ·
                          Patrol Optimizer (src/optimizer.py) · Forecast
```

Heavy work is precomputed into tiny artifacts so the dashboard loads instantly
and the shipped code stays well under the 50MB limit (raw CSVs are not bundled).

---

## Run it locally

```bash
# 1. clone
git clone <your-repo-url> && cd park-scope

# 2. environment
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. data — place the two HackerEarth CSVs in data/ (see data/README_DATA.md)

# 4. build artifacts (reads data/, writes artifacts/)
python -m src.prepare_data

# 5. launch the dashboard
streamlit run app.py
```

A hosted live version can be deployed free on **Streamlit Community Cloud** or
**Hugging Face Spaces** — push this repo and point the platform at `app.py`.

---

## Rule compliance

Every signal in PARK-SCOPE is derived **only** from the two datasets HackerEarth
provides. "Strategic proximity" is *not* an external map layer — it is a keyword
flag (`metro / market / hospital / school / bus / station / circle`) read from
BTP's own `junction_name` labels in the supplied data. No external dataset is
read anywhere in the pipeline.

## Honest limitation

This is **enforcement-activity** data: it shows where tickets were *issued*, not
every violation that *occurred*, so it carries enforcement selection bias.
PARK-SCOPE is therefore framed as a tool to make existing enforcement effort far
more efficient. The pipeline is built to ingest ANPR / camera / sensor feeds as a
direct upgrade to ground truth.

## Repo layout

```
park-scope/
├── app.py                 # Streamlit command console
├── requirements.txt
├── README.md
├── data/                  # put raw CSVs here (git-ignored)
├── artifacts/             # generated small outputs the app reads
└── src/
    ├── config.py          # all weights/paths/assumptions (auditable)
    ├── prepare_data.py     # pipeline entrypoint
    ├── impact.py          # Congestion Impact Score
    ├── forecast.py        # rate table + GBM validation
    └── optimizer.py       # patrol allocation
```
