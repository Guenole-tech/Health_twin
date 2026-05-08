# -*- coding: utf-8 -*-
"""
TWINOS · Dashboard (UI Premium v3)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.io import load_json

# ── CONFIG ─────────────────────────────────────
st.set_page_config(
    page_title="TWINOS · Jumeau Numérique",
    page_icon="🧬",
    layout="wide",
)

# ── FONTS ───────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ── STYLE GLOBAL ───────────────────────────────
st.markdown("""
<style>

/* ── BASE ── */
*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: #050A0F;
    color: #C8D8E8;
    font-family: 'DM Mono', monospace;
}

[data-testid="stMain"] { background: transparent; }
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { display: none; }
footer { visibility: hidden; }

/* ── SCANLINE OVERLAY ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 200, 150, 0.012) 2px,
        rgba(0, 200, 150, 0.012) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── TYPOGRAPHY ── */
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    color: #E8F4F0 !important;
}

/* ── HEADER BANNER ── */
.twinos-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 28px;
    margin-bottom: 28px;
    background: linear-gradient(135deg, rgba(0,240,180,0.06) 0%, rgba(0,100,200,0.06) 100%);
    border: 1px solid rgba(0, 240, 180, 0.15);
    border-radius: 4px;
    position: relative;
    overflow: hidden;
}

.twinos-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00F0B4, #0080FF, #00F0B4);
    background-size: 200% 100%;
    animation: scan 4s linear infinite;
}

@keyframes scan {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

.twinos-logo {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: #E8F4F0;
    letter-spacing: -0.03em;
}

.twinos-logo span {
    color: #00F0B4;
}

.twinos-badge {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #00F0B4;
    background: rgba(0, 240, 180, 0.1);
    border: 1px solid rgba(0, 240, 180, 0.25);
    border-radius: 3px;
    padding: 4px 10px;
    letter-spacing: 0.08em;
}

.twinos-ts {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: rgba(200, 216, 232, 0.4);
    letter-spacing: 0.05em;
}

/* ── METRIC CARDS ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 28px;
}

.mc {
    background: rgba(255,255,255,0.032);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 4px;
    padding: 18px 16px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.mc:hover { border-color: rgba(0,240,180,0.25); }

.mc::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
    opacity: 0.5;
}

.mc-icon {
    font-size: 14px;
    margin-bottom: 10px;
    display: block;
    opacity: 0.7;
}

.mc-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.1em;
    color: rgba(200,216,232,0.45);
    text-transform: uppercase;
    margin-bottom: 6px;
    display: block;
}

.mc-value {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: #E8F4F0;
    display: block;
    line-height: 1;
    letter-spacing: -0.02em;
}

.mc-unit {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: rgba(200,216,232,0.35);
    margin-left: 3px;
}

/* ── SENSOR GRID ── */
.sensor-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 28px;
}

.sensor-card {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.055);
    border-left: 2px solid rgba(0,240,180,0.35);
    border-radius: 0 4px 4px 0;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.sensor-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.1em;
    color: rgba(200,216,232,0.4);
    text-transform: uppercase;
}

.sensor-value {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 600;
    color: #00F0B4;
    letter-spacing: -0.01em;
}

/* ── SECTION TITLES ── */
.section-title {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.15em;
    color: rgba(200,216,232,0.35);
    text-transform: uppercase;
    padding: 0 0 12px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-title::before {
    content: '';
    display: inline-block;
    width: 16px;
    height: 2px;
    background: #00F0B4;
    border-radius: 1px;
}

/* ── ALERT PILLS ── */
.alert-red {
    background: rgba(255, 60, 60, 0.07);
    border: 1px solid rgba(255, 60, 60, 0.2);
    border-left: 3px solid #FF3C3C;
    border-radius: 0 4px 4px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #FF9090;
}

.alert-yellow {
    background: rgba(255, 190, 0, 0.07);
    border: 1px solid rgba(255, 190, 0, 0.2);
    border-left: 3px solid #FFBE00;
    border-radius: 0 4px 4px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #FFD570;
}

.alert-green {
    background: rgba(0, 240, 180, 0.06);
    border: 1px solid rgba(0, 240, 180, 0.15);
    border-left: 3px solid #00F0B4;
    border-radius: 0 4px 4px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #70FFDB;
}

/* ── DIVIDER ── */
.tw-divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 24px 0;
}

/* ── STREAMLIT OVERRIDES ── */
[data-testid="stMetric"] { background: transparent !important; }
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 28px !important;
    color: #E8F4F0 !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 4px !important;
}

/* ── BUTTON ── */
[data-testid="stButton"] button {
    background: rgba(0,240,180,0.08) !important;
    border: 1px solid rgba(0,240,180,0.25) !important;
    color: #00F0B4 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.05em !important;
    border-radius: 3px !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}

[data-testid="stButton"] button:hover {
    background: rgba(0,240,180,0.15) !important;
    border-color: rgba(0,240,180,0.5) !important;
}

/* ── VITALITY RING ── */
.vring-wrap {
    text-align: center;
    padding: 12px;
}

.vring-val {
    font-family: 'Syne', sans-serif;
    font-size: 42px;
    font-weight: 800;
    color: #00F0B4;
    letter-spacing: -0.03em;
    display: block;
}

.vring-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    color: rgba(200,216,232,0.35);
    text-transform: uppercase;
}

/* ── PLOTLY CONTAINER ── */
.stPlotlyChart {
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 4px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)


# ── DATA ───────────────────────────────────────
@st.cache_data(ttl=30)
def load_data():
    snap = load_json("data/dashboard_data.json")
    try:
        anom = pd.read_csv("data/anomalies.csv", parse_dates=["ts"])
    except FileNotFoundError:
        anom = pd.DataFrame()
    return snap, anom

snap, anomalies = load_data()

if not snap:
    st.error("Aucune donnée trouvée. Lance : python main.py")
    st.stop()


# ── COULEURS CHARTS ────────────────────────────
CHART_COLORS = ["#00F0B4", "#0080FF", "#FF6B35", "#B060FF", "#FFD700"]
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8AA4B4", family="DM Mono, monospace", size=11),
    margin=dict(l=0, r=0, t=0, b=0),
    coloraxis_showscale=False,
)

PLOTLY_AXES = dict(
    gridcolor="rgba(255,255,255,0.05)",
    linecolor="rgba(255,255,255,0.08)",
    tickcolor="rgba(255,255,255,0.08)",
)


# ── HEADER ─────────────────────────────────────
vitality = snap.get('vitality_score', 0)
ts = snap.get('ts', '—')

st.markdown(f"""
<div class="twinos-header">
    <div>
        <div class="twinos-logo">TWIN<span>OS</span></div>
        <div class="twinos-ts" style="margin-top:5px">Dernière sync — {ts}</div>
    </div>
    <div style="text-align:center">
        <div class="vring-val">{vitality:.0f}</div>
        <div class="vring-label">Vitalité / 100</div>
    </div>
    <div class="twinos-badge">⬤ SYSTÈME ACTIF</div>
</div>
""", unsafe_allow_html=True)


# ── VITAUX ─────────────────────────────────────
st.markdown('<div class="section-title">Systèmes vitaux</div>', unsafe_allow_html=True)

hr    = snap.get('hr', 0)
hrv   = snap.get('hrv', 0)
sleep = snap.get('sleep_score', 0)
stress= snap.get('stress_index', 0)
steps = snap.get('steps_today', 0)

st.markdown(f"""
<div class="metric-row">
  <div class="mc" style="--accent:#FF6B6B">
    <span class="mc-icon">♥</span>
    <span class="mc-label">Fréquence card.</span>
    <span class="mc-value">{hr:.0f}<span class="mc-unit">bpm</span></span>
  </div>
  <div class="mc" style="--accent:#FF9EAA">
    <span class="mc-icon">◈</span>
    <span class="mc-label">HRV</span>
    <span class="mc-value">{hrv:.0f}<span class="mc-unit">ms</span></span>
  </div>
  <div class="mc" style="--accent:#7B9EFF">
    <span class="mc-icon">◐</span>
    <span class="mc-label">Sommeil</span>
    <span class="mc-value">{sleep:.0f}<span class="mc-unit">/100</span></span>
  </div>
  <div class="mc" style="--accent:#FFB347">
    <span class="mc-icon">⚡</span>
    <span class="mc-label">Stress</span>
    <span class="mc-value">{stress:.2f}</span>
  </div>
  <div class="mc" style="--accent:#00F0B4">
    <span class="mc-icon">▶</span>
    <span class="mc-label">Pas aujourd'hui</span>
    <span class="mc-value">{steps:,}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── CAPTEURS ───────────────────────────────────
st.markdown('<div class="section-title">Capteurs sensoriels</div>', unsafe_allow_html=True)

sound = snap.get('sound_db', 0)
voc   = snap.get('voc_index', 0)
temp  = snap.get('skin_temp', 0)
gsr   = snap.get('gsr', 0)

st.markdown(f"""
<div class="sensor-grid">
  <div class="sensor-card">
    <span class="sensor-label">Son ambiant</span>
    <span class="sensor-value">{sound:.1f} <span style="font-size:12px;color:rgba(200,216,232,0.3)">dB</span></span>
  </div>
  <div class="sensor-card">
    <span class="sensor-label">VOC Index</span>
    <span class="sensor-value">{voc:.3f}</span>
  </div>
  <div class="sensor-card">
    <span class="sensor-label">Temp. cutanée</span>
    <span class="sensor-value">{temp:.1f} <span style="font-size:12px;color:rgba(200,216,232,0.3)">°C</span></span>
  </div>
  <div class="sensor-card">
    <span class="sensor-label">GSR</span>
    <span class="sensor-value">{gsr:.2f} <span style="font-size:12px;color:rgba(200,216,232,0.3)">μS</span></span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="tw-divider"></div>', unsafe_allow_html=True)


# ── RISQUES + ALERTES ──────────────────────────
col_risk, col_alert = st.columns([3, 2], gap="large")

with col_risk:
    st.markdown('<div class="section-title">Risques anticipés · 6 mois</div>', unsafe_allow_html=True)

    risks = snap.get("risks", {})

    if risks:
        risk_df = pd.DataFrame({
            "Risque": [k.replace("_", " ").title() for k in risks],
            "Probabilité": list(risks.values()),
        }).sort_values("Probabilité")

        # Color gradient based on value
        colors = [
            "#00F0B4" if v < 0.3 else "#FFB347" if v < 0.6 else "#FF6B6B"
            for v in risk_df["Probabilité"]
        ]

        fig = go.Figure(go.Bar(
            x=risk_df["Probabilité"],
            y=risk_df["Risque"],
            orientation="h",
            marker=dict(
                color=risk_df["Probabilité"],
                colorscale=[[0, "#00F0B4"], [0.5, "#FFB347"], [1, "#FF6B6B"]],
                line=dict(color="rgba(0,0,0,0)", width=0),
            ),
            text=risk_df["Probabilité"].apply(lambda x: f"{x:.0%}"),
            textfont=dict(family="DM Mono, monospace", size=11, color="#C8D8E8"),
            textposition="outside",
        ))

        fig.update_layout(height=300, bargap=0.35, **PLOTLY_LAYOUT)
        fig.update_xaxes(
            range=[0, 1.1],
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.08)",
            tickcolor="rgba(255,255,255,0.08)",
            tickformat=".0%",
            tickfont=dict(family="DM Mono", size=10),
        )
        fig.update_yaxes(
            gridcolor="rgba(0,0,0,0)",
            linecolor="rgba(255,255,255,0.08)",
            tickcolor="rgba(255,255,255,0.08)",
            tickfont=dict(family="DM Mono", size=11),
        )

        st.plotly_chart(fig, use_container_width=True)

with col_alert:
    st.markdown('<div class="section-title">Alertes système</div>', unsafe_allow_html=True)

    for alert in snap.get("alerts", []):
        if "🔴" in alert:
            st.markdown(f'<div class="alert-red">{alert}</div>', unsafe_allow_html=True)
        elif "🟡" in alert:
            st.markdown(f'<div class="alert-yellow">{alert}</div>', unsafe_allow_html=True)
        elif "✅" in alert:
            st.markdown(f'<div class="alert-green">{alert}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-green">{alert}</div>', unsafe_allow_html=True)


st.markdown('<div class="tw-divider"></div>', unsafe_allow_html=True)


# ── ANOMALIES ──────────────────────────────────
n_anom = snap.get('anomalies_today', 0)
st.markdown(f'<div class="section-title">Anomalies détectées · <span style="color:#FF6B6B">{n_anom} aujourd\'hui</span></div>', unsafe_allow_html=True)

if not anomalies.empty:

    cols = [c for c in ["ts","anomaly_score","hr","gsr","sound_db","voc_index","stress_index"]
            if c in anomalies.columns]

    display = anomalies[cols].head(15).copy()

    if "anomaly_score" in display.columns:
        display["anomaly_score"] = display["anomaly_score"].round(3)

    st.dataframe(
        display.sort_values("anomaly_score", ascending=False),
        use_container_width=True,
        height=280,
    )

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=anomalies["ts"],
        y=anomalies["anomaly_score"],
        mode="markers",
        marker=dict(
            size=7,
            color=anomalies["anomaly_score"],
            colorscale=[[0, "#00F0B4"], [0.5, "#FFB347"], [1, "#FF4444"]],
            opacity=0.85,
            line=dict(width=0),
        ),
    ))

    # Threshold line
    fig2.add_hline(
        y=0.75,
        line_dash="dot",
        line_color="rgba(255,107,107,0.35)",
        annotation_text="seuil critique",
        annotation_font=dict(family="DM Mono", size=10, color="rgba(255,107,107,0.5)"),
    )

    fig2.update_layout(
        height=220,
        **PLOTLY_LAYOUT,
        showlegend=False,
    )

    st.plotly_chart(fig2, use_container_width=True)

else:
    st.markdown("""
    <div style="padding:32px;text-align:center;border:1px solid rgba(255,255,255,0.06);border-radius:4px;font-family:'DM Mono',monospace;font-size:12px;color:rgba(200,216,232,0.3)">
        — Aucune anomalie détectée — Lance <code>python main.py</code>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="tw-divider"></div>', unsafe_allow_html=True)


# ── FOOTER ─────────────────────────────────────
col_foot, col_btn = st.columns([4, 1])

with col_foot:
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:0.08em;color:rgba(200,216,232,0.2);padding:8px 0">
        TWINOS v0.1-alpha · système expérimental · refresh 30s
    </div>
    """, unsafe_allow_html=True)

with col_btn:
    if st.button("⟳ Relancer pipeline"):
        import subprocess
        with st.spinner("Exécution..."):
            subprocess.run(["python", "main.py"], check=True)
        st.cache_data.clear()
        st.rerun()
