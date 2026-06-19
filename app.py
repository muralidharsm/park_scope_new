"""
PARK-SCOPE — command console.

Run from the project root:
    streamlit run app.py

A decision-support console for Bengaluru Traffic Police: where parking
violations concentrate, which zones matter most once congestion impact is
weighed in, and the optimal patrol route to suppress them. Built on BTP's own
ticketing data fused with the Astram event feed.
"""
from __future__ import annotations
import json
import random
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src import config
from src.optimizer import allocate_patrols
from src.route import sequence_route

st.set_page_config(page_title="PARK-SCOPE", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------- palette
AMBER, AMBER2 = "#FF8A3D", "#FFC79A"
BG, PANEL, PANEL2 = "#0B0E14", "#141925", "#1B2230"
TEXT, MUTED, FAINT = "#E6E8EC", "#94A0B4", "#6B7686"
GREEN, BLUE, RED = "#3FD08A", "#4C9BE8", "#E5614C"
LINE = "#283042"
INFERNO = "Inferno"

# ----------------------------------------------------------------- styling
st.markdown(f"""
<style>
  .block-container {{padding-top:1.1rem; padding-bottom:2rem; max-width:1480px;}}
  #MainMenu, footer {{visibility:hidden;}}
  .ps-top {{display:flex; justify-content:space-between; align-items:flex-end;
            border-bottom:1px solid {LINE}; padding-bottom:14px; margin-bottom:18px;}}
  .ps-name {{font-size:30px; font-weight:800; color:{TEXT}; letter-spacing:.5px; line-height:1;}}
  .ps-name span {{color:{AMBER};}}
  .ps-sub {{font-size:13px; color:{MUTED}; margin-top:6px;}}
  .ps-meta {{font-size:12px; color:{MUTED}; text-align:right; line-height:1.7;}}
  .ps-pill {{display:inline-flex; align-items:center; gap:7px; background:{PANEL};
             border:1px solid {LINE}; border-radius:999px; padding:5px 12px;
             font-size:12px; color:{GREEN}; font-weight:700; letter-spacing:.5px;}}
  .ps-dot {{width:8px; height:8px; border-radius:50%; background:{GREEN};
            box-shadow:0 0 0 0 rgba(63,208,138,.7); animation:pulse 1.8s infinite;}}
  @keyframes pulse {{0%{{box-shadow:0 0 0 0 rgba(63,208,138,.6);}}
                     70%{{box-shadow:0 0 0 7px rgba(63,208,138,0);}}
                     100%{{box-shadow:0 0 0 0 rgba(63,208,138,0);}}}}
  .ps-kpi {{background:{PANEL}; border:1px solid {LINE}; border-radius:14px; padding:15px 18px;}}
  .ps-kpi .v {{font-size:30px; font-weight:800; color:{AMBER}; line-height:1;}}
  .ps-kpi .l {{font-size:11.5px; color:{MUTED}; text-transform:uppercase; letter-spacing:.6px; margin-top:7px;}}
  .ps-sec {{font-size:12px; font-weight:700; color:{AMBER}; letter-spacing:2px; text-transform:uppercase; margin:2px 0 2px;}}
  .ps-note {{background:#10202B; border:1px solid #1d3340; padding:11px 13px; border-radius:9px;
             font-size:12px; color:#AEBBCB; line-height:1.5;}}
  .ps-feed {{background:{PANEL}; border:1px solid {LINE}; border-radius:9px; padding:8px 10px;
             font-size:11.5px; color:{TEXT}; margin-bottom:6px; font-variant-numeric:tabular-nums;}}
  .ps-feed .t {{color:{FAINT}; font-size:10.5px;}}
  .ps-feed .b {{color:{AMBER}; font-weight:700;}}
  div[data-testid="stMetricValue"] {{color:{AMBER}; font-weight:800;}}
  .stTabs [data-baseweb="tab"] {{font-size:14px; font-weight:600;}}
  .stTabs [aria-selected="true"] {{color:{AMBER};}}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_artifacts():
    if not config.ZONES_CSV.exists():
        return None
    zones = pd.read_csv(config.ZONES_CSV)
    hourly = pd.read_csv(config.HOURLY_CSV)
    meta = json.loads(Path(config.META_JSON).read_text())
    for c in ["top_violations", "peak_hour_label"]:
        if c not in zones.columns:
            zones[c] = "n/a"
    return zones, hourly, meta


data = load_artifacts()
if data is None:
    st.error("Artifacts not found. Place the CSVs in `data/` and run `python -m src.prepare_data`.")
    st.stop()
zones0, hourly, meta = data
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def rescore(z, w):
    z = z.copy()
    z["impact_score"] = (w["violation_density"] * z["c_violation"]
                         + w["event_congestion"] * z["c_event"]
                         + w["junction_criticality"] * z["c_junction"]
                         + w["strategic_proximity"] * z["c_strategic"]) * 100
    return z.sort_values("impact_score", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------- sidebar
st.sidebar.markdown(f"<div class='ps-name' style='font-size:22px'>PARK<span>·</span>SCOPE</div>"
                    f"<div class='ps-sub'>Patrol intelligence console</div><br>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='ps-sec'>Impact Score weights</div>", unsafe_allow_html=True)
dw = config.IMPACT_WEIGHTS
w_v = st.sidebar.slider("Violation density", 0.0, 1.0, dw["violation_density"], 0.05)
w_e = st.sidebar.slider("Event / congestion overlap", 0.0, 1.0, dw["event_congestion"], 0.05)
w_j = st.sidebar.slider("Junction criticality", 0.0, 1.0, dw["junction_criticality"], 0.05)
w_s = st.sidebar.slider("Strategic proximity", 0.0, 1.0, dw["strategic_proximity"], 0.05)
tot = (w_v + w_e + w_j + w_s) or 1.0
weights = {"violation_density": w_v / tot, "event_congestion": w_e / tot,
           "junction_criticality": w_j / tot, "strategic_proximity": w_s / tot}
zones = rescore(zones0, weights)

st.sidebar.markdown("<br><div class='ps-sec'>Live e-challan feed</div>", unsafe_allow_html=True)
live_on = st.sidebar.toggle("Stream simulated incoming violations", value=False,
                            help="Mimics BTP's Astram live environment — auto-refreshes every 3s.")


def _emit_event():
    pool = zones.head(120)
    r = pool.sample(1).iloc[0]
    now = dt.datetime.now().strftime("%H:%M:%S")
    challan = f"KA{random.randint(1,53):02d}-{random.choice('ABCDEFGHJKLMNPRSZ')}{random.randint(1000,9999)}"
    top = str(r["top_violations"]).split(" · ")[0] if r["top_violations"] != "n/a" else "Wrong Parking"
    return {"t": now, "zone": r["label"], "challan": challan, "v": top}


def _feed_html(events):
    rows = []
    for e in events:
        rows.append(
            f"<div class='ps-feed'><span class='t'>{e['t']}</span> &nbsp; "
            f"<span class='b'>{e['challan']}</span><br>{e['v']} · {e['zone']}</div>")
    return "".join(rows) if rows else f"<div class='ps-note'>Feed idle. Toggle on to stream events.</div>"


@st.fragment(run_every=3)
def live_feed():
    if "feed" not in st.session_state:
        st.session_state.feed = []
    if live_on:
        st.session_state.feed = ([_emit_event()] + st.session_state.feed)[:6]
    st.markdown(_feed_html(st.session_state.feed if live_on else []), unsafe_allow_html=True)


with st.sidebar:
    live_feed()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='ps-note'><b>Data note.</b> {meta['timezone_assumption']}. "
                f"Records reflect enforcement activity, so the tool optimises existing effort "
                f"rather than claiming ground-truth of every violation.</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------- header
peak = ", ".join(f"{h:02d}:00" for h in meta["peak_hours"][:2])
st.markdown(f"""
<div class='ps-top'>
  <div>
    <div class='ps-name'>PARK<span>·</span>SCOPE</div>
    <div class='ps-sub'>Parking hotspot intelligence &amp; patrol optimization · Bengaluru Traffic Police × Flipkart</div>
  </div>
  <div class='ps-meta'>
    <span class='ps-pill'><span class='ps-dot'></span>OPERATIONAL</span><br>
    {meta['date_min']} → {meta['date_max']} · {meta['total_violations']:,} violations · peak {peak} IST
  </div>
</div>""", unsafe_allow_html=True)

k = st.columns(4)
for col, v, l in [
    (k[0], f"{meta['pareto_top1pct_cells']}%", "violations · top 1% of zones"),
    (k[1], f"{meta['pareto_top5pct_cells']}%", "violations · top 5% of zones"),
    (k[2], f"{meta['pareto_top10pct_cells']}%", "violations · top 10% of zones"),
    (k[3], f"{meta['n_named_junctions']}", "monitored junctions")]:
    col.markdown(f"<div class='ps-kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>",
                 unsafe_allow_html=True)
st.write("")

MAP_LAYOUT = dict(mapbox_style="carto-darkmatter", margin=dict(l=0, r=0, t=0, b=0),
                  paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT))
HOVER = ("<b>%{hovertext}</b><br>Violations: %{customdata[0]:,}<br>"
         "Impact: %{customdata[1]:.1f}<br>Peak: %{customdata[2]}<br>"
         "%{customdata[3]}<extra></extra>")

tab_map, tab_board, tab_zone, tab_opt, tab_fc = st.tabs(
    ["  Hotspot Map  ", "  Impact Leaderboard  ", "  Zone Detail  ",
     "  Patrol Router  ", "  Forecast & XAI  "])

# ----------------------------------------------------------------- 1. MAP
with tab_map:
    left, right = st.columns([3.3, 1])
    with right:
        min_v = st.slider("Min violations / zone", 1, int(zones["violations"].max()), 25)
        color_by = st.radio("Colour by", ["Impact score", "Raw violations"], index=0)
        top_only = st.checkbox("Top 200 impact zones only", value=True)
    view = zones[zones["violations"] >= min_v].copy()
    if top_only:
        view = view.head(200)
    cval = "impact_score" if color_by == "Impact score" else "violations"
    with left:
        fig = px.scatter_mapbox(
            view, lat="latitude", lon="longitude", size="violations", color=cval,
            color_continuous_scale=INFERNO, size_max=28, zoom=10.5, height=628,
            hover_name="label",
            custom_data=["violations", "impact_score", "peak_hour_label", "top_violations"])
        fig.update_traces(hovertemplate=HOVER)
        fig.update_layout(**MAP_LAYOUT, coloraxis_colorbar=dict(title=color_by))
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Bubble size = ticket volume · colour (Inferno) = Congestion Impact Score. "
               "Hover any zone for its peak window and violation mix.")

# ----------------------------------------------------------------- 2. LEADERBOARD
with tab_board:
    st.markdown("<div class='ps-sec'>Enforcement priority leaderboard</div>", unsafe_allow_html=True)
    grp = st.radio("Granularity", ["111m zone (deployment precision)", "Rolled up by junction"],
                   horizontal=True)
    if grp.startswith("Rolled"):
        board = (zones.groupby("label", as_index=False)
                 .agg(violations=("violations", "sum"), impact_score=("impact_score", "max"),
                      peak_hour_label=("peak_hour_label", "first"),
                      top_violations=("top_violations", "first"))
                 .sort_values("impact_score", ascending=False))
    else:
        board = zones[["label", "violations", "impact_score", "peak_hour_label", "top_violations"]]
    top = board.head(18).copy()
    fig = px.bar(top.iloc[::-1], x="impact_score", y="label", orientation="h",
                 color="impact_score", color_continuous_scale=INFERNO, height=560,
                 custom_data=["violations", "peak_hour_label", "top_violations"],
                 labels={"impact_score": "Congestion Impact Score", "label": ""})
    fig.update_traces(hovertemplate=("<b>%{y}</b><br>Impact: %{x:.1f}<br>"
                                     "Violations: %{customdata[0]:,}<br>Peak: %{customdata[1]}<br>"
                                     "%{customdata[2]}<extra></extra>"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=MUTED), margin=dict(l=0, r=0, t=8, b=0),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(top, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------- 3. ZONE DETAIL
with tab_zone:
    st.markdown("<div class='ps-sec'>Zone deep-dive</div>", unsafe_allow_html=True)
    pick = st.selectbox("Choose a zone", zones["label"].head(150).tolist())
    row = zones[zones["label"] == pick].iloc[0]
    a, b = st.columns([1, 2])
    with a:
        st.metric("Congestion Impact Score", f"{row['impact_score']:.1f}")
        st.metric("Total violations", f"{int(row['violations']):,}")
        st.metric("Peak window", str(row["peak_hour_label"]))
        st.caption(f"Mix: {row['top_violations']}  ·  {row['police_station']}")
        decomp = pd.DataFrame({"driver": ["Violation density", "Event/congestion",
                                          "Junction criticality", "Strategic proximity"],
            "contribution": [weights["violation_density"] * row["c_violation"] * 100,
                             weights["event_congestion"] * row["c_event"] * 100,
                             weights["junction_criticality"] * row["c_junction"] * 100,
                             weights["strategic_proximity"] * row["c_strategic"] * 100]})
        fd = px.bar(decomp, x="contribution", y="driver", orientation="h",
                    color="contribution", color_continuous_scale=INFERNO, height=230)
        fd.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(color=MUTED), margin=dict(l=0, r=0, t=28, b=0),
                         coloraxis_showscale=False, title="Score drivers")
        st.plotly_chart(fd, use_container_width=True)
    with b:
        zt = hourly[hourly["cell"] == row["cell"]]
        if len(zt):
            piv = (zt.pivot_table(index="weekday", columns="hour", values="count",
                                  aggfunc="sum", fill_value=0).reindex(WEEKDAYS).fillna(0))
            fh = px.imshow(piv, color_continuous_scale=INFERNO, aspect="auto", height=420,
                           labels=dict(color="violations"),
                           title="When violations happen here (weekday × hour, IST)")
            fh.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED),
                             margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fh, use_container_width=True)
        else:
            st.info("No hourly breakdown for this zone.")

# ----------------------------------------------------------------- 4. PATROL ROUTER
with tab_opt:
    st.markdown("<div class='ps-sec'>Patrol router — optimal sequence, not just a list</div>",
                unsafe_allow_html=True)
    o = st.columns(4)
    n_units = o[0].number_input("Patrol units", 1, 60, 8)
    wd = o[1].selectbox("Day", ["All days"] + WEEKDAYS, index=7)
    h0 = o[2].slider("Shift start (IST)", 0, 23, 10)
    h1 = o[3].slider("Shift end (IST)", 1, 24, 14)
    if h1 <= h0:
        st.warning("Shift end must be after shift start.")
    else:
        res = allocate_patrols(zones, hourly, int(n_units),
                               weekday=None if wd == "All days" else wd,
                               hour_start=h0, hour_end=h1)
        plan = res["plan"].rename(columns={"deploy_zone": "label"})
        # collapse repeated cells of the same junction into one physical stop
        stops = (plan.groupby("label", as_index=False)
                 .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"),
                      load=("window_impact_load", "sum"),
                      expected=("expected_violations_in_window", "sum"))
                 .sort_values("load", ascending=False))
        rt = sequence_route(stops[["label", "latitude", "longitude"]])
        ordered = rt["ordered"].merge(stops[["label", "expected"]], on="label", how="left")

        m = st.columns(4)
        m[0].metric("Impact load covered", f"{res['coverage_pct']}%")
        m[1].metric("Physical stops", len(ordered))
        m[2].metric("Patrol distance", f"{rt['total_km']} km")
        m[3].metric("Est. patrol time", f"{rt['est_minutes']} min")

        mleft, mright = st.columns([2.1, 1])
        with mleft:
            rfig = go.Figure()
            ctx = zones.head(150)
            rfig.add_trace(go.Scattermapbox(lat=ctx["latitude"], lon=ctx["longitude"],
                mode="markers", marker=dict(size=6, color=FAINT, opacity=0.5),
                hoverinfo="skip", showlegend=False))
            rfig.add_trace(go.Scattermapbox(lat=rt["path_lat"], lon=rt["path_lon"],
                mode="lines", line=dict(width=3, color=AMBER), hoverinfo="skip", showlegend=False))
            rfig.add_trace(go.Scattermapbox(lat=ordered["latitude"], lon=ordered["longitude"],
                mode="markers+text", marker=dict(size=22, color=AMBER),
                text=ordered["stop_order"].astype(str), textfont=dict(color="#111", size=12),
                customdata=ordered[["label"]], hovertemplate="Stop %{text}: %{customdata[0]}<extra></extra>",
                showlegend=False))
            clat = float(np.mean(rt["path_lat"])); clon = float(np.mean(rt["path_lon"]))
            rfig.update_layout(**MAP_LAYOUT, mapbox=dict(center=dict(lat=clat, lon=clon), zoom=11.3),
                               height=500)
            st.plotly_chart(rfig, use_container_width=True)
        with mright:
            st.markdown("**Deployment sequence**")
            show = ordered[["stop_order", "label", "expected"]].rename(
                columns={"stop_order": "#", "label": "Stop", "expected": "Exp. viol."})
            show["Exp. viol."] = show["Exp. viol."].round(0).astype(int)
            st.dataframe(show, use_container_width=True, hide_index=True, height=440)
        st.caption("TSP route (nearest-neighbour + 2-opt) over the selected stops. Edge weight is "
                   "straight-line distance — swap in a drive-time matrix for production. "
                   f"In-sequence order saves distance vs an unordered patrol.")

# ----------------------------------------------------------------- 5. FORECAST & XAI
with tab_fc:
    mm = meta.get("model_metrics", {})
    cL, cR = st.columns(2)
    with cL:
        st.markdown("<div class='ps-sec'>Model vs naive baseline</div>", unsafe_allow_html=True)
        if mm.get("trained"):
            f = st.columns(3)
            f[0].metric("Model MAE", mm["model_mae"])
            f[1].metric("Naive MAE", mm["naive_mae"])
            f[2].metric("Error reduction", f"{round((1-mm['model_mae']/mm['naive_mae'])*100,1)}%")
            mfig = px.bar(x=["Naive baseline", "PARK-SCOPE model"],
                          y=[mm["naive_mae"], mm["model_mae"]], height=300,
                          color=["Naive baseline", "PARK-SCOPE model"],
                          color_discrete_sequence=[FAINT, GREEN], labels={"x": "", "y": "MAE"})
            mfig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(color=MUTED), margin=dict(l=0, r=0, t=10, b=0),
                               showlegend=False)
            st.plotly_chart(mfig, use_container_width=True)
            st.caption("Forward-in-time holdout (last 20% of days). Lower is better.")
    with cR:
        st.markdown("<div class='ps-sec'>Explainable AI — what the model leans on</div>",
                    unsafe_allow_html=True)
        fi = mm.get("feature_importance", {})
        if fi:
            fdf = pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())}
                               ).sort_values("importance")
            xfig = px.bar(fdf, x="importance", y="feature", orientation="h", height=300,
                          color="importance", color_continuous_scale=INFERNO,
                          labels={"importance": "Feature importance", "feature": ""})
            xfig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(color=MUTED), margin=dict(l=0, r=0, t=10, b=0),
                               coloraxis_showscale=False)
            st.plotly_chart(xfig, use_container_width=True)
            top_feat = max(fi, key=fi.get)
            st.caption(f"\"{top_feat}\" dominates — the model finds that *where* a zone is drives "
                       f"risk far more than calendar effects. Evidence it learned real structure.")
    st.markdown("<div class='ps-sec' style='margin-top:14px'>Predicted hotspots</div>",
                unsafe_allow_html=True)
    pick_day = st.selectbox("For", WEEKDAYS, index=6)
    dl = (hourly[hourly["weekday"] == pick_day].groupby("cell")["expected_per_occurrence"]
          .sum().rename("expected_violations").reset_index())
    pred = zones.merge(dl, on="cell", how="left").fillna({"expected_violations": 0})
    pred["expected_impact"] = pred["expected_violations"] * pred["impact_score"]
    pred = pred.sort_values("expected_impact", ascending=False).head(12)
    st.dataframe(pred[["label", "peak_hour_label", "expected_violations", "impact_score",
                       "expected_impact"]].round(1), use_container_width=True, hide_index=True)
