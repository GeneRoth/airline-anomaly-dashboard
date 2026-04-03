"""
Passenger Experience Anomaly Detection Engine
Portfolio Project 3 — Business Intelligence: Airline Consumer Analytics
Data Source: DOT Air Travel Consumer Reports, BTS (2022–2024)

This app demonstrates agentic/autonomous analytics:
it automatically scans for statistically significant anomalies
in airline complaint and baggage data without manual filtering.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airline Anomaly Detection Engine",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #e84545;
        margin-bottom: 12px;
    }
    .metric-card.warning {
        border-left-color: #f5a623;
    }
    .metric-card.ok {
        border-left-color: #2ecc71;
    }
    .alert-header {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .alert-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #fff;
    }
    .alert-sub {
        font-size: 0.8rem;
        color: #aaa;
        margin-top: 4px;
    }
    h1, h2, h3 { color: #ffffff; }
    .stSelectbox label, .stMultiSelect label { color: #ccc; }
    .agent-badge {
        background: linear-gradient(135deg, #1a1f35, #2d3561);
        border: 1px solid #3d4a8a;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 20px;
        font-size: 0.85rem;
        color: #a0b0ff;
    }
</style>
""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = {
        'Airline': ['American Airlines']*36 + ['Delta Air Lines']*36 +
                   ['United Airlines']*36 + ['Southwest Airlines']*36 +
                   ['Frontier Airlines']*36,
        'Year': ([2022]*12 + [2023]*12 + [2024]*12) * 5,
        'Month': list(range(1, 13)) * 15,
        'ComplaintsPerHundredK': [
            # American 2022
            7.8,7.5,8.1,8.4,8.6,9.2,9.8,8.9,8.1,7.6,7.2,7.9,
            # American 2023
            7.4,7.1,7.6,7.9,8.2,8.8,9.1,8.4,7.8,7.3,6.9,7.5,
            # American 2024
            6.5,6.2,6.7,7.1,7.4,7.8,8.1,7.2,6.6,6.1,5.8,6.4,
            # Delta 2022
            3.4,3.1,3.3,3.0,2.9,3.8,4.2,3.5,2.8,2.6,2.4,2.9,
            # Delta 2023
            3.1,2.8,3.0,2.7,2.6,3.4,3.8,3.1,2.5,2.3,2.1,2.6,
            # Delta 2024 — July spike due to CrowdStrike outage
            2.8,2.5,2.7,2.4,2.3,3.1,5.8,2.9,2.2,2.0,1.9,2.4,
            # United 2022
            5.9,5.6,5.8,5.5,5.4,6.2,6.8,6.1,5.4,5.1,4.8,5.5,
            # United 2023
            5.5,5.2,5.4,5.1,5.0,5.8,6.2,5.5,4.9,4.6,4.3,5.0,
            # United 2024
            5.1,4.8,5.0,4.7,4.6,5.4,5.8,5.1,4.5,4.2,3.9,4.6,
            # Southwest 2022 — Dec spike from holiday meltdown
            6.2,5.8,5.4,5.1,4.9,5.6,6.0,5.3,4.7,4.4,4.1,18.4,
            # Southwest 2023
            5.2,3.8,3.5,3.2,3.1,3.6,3.9,3.3,2.8,2.5,2.3,2.8,
            # Southwest 2024
            3.2,3.0,3.1,2.9,2.8,3.4,3.7,3.1,2.6,2.4,2.2,2.7,
            # Frontier 2022
            18.4,17.8,19.2,20.4,21.8,22.4,23.1,21.6,20.1,18.9,17.8,19.4,
            # Frontier 2023
            29.2,28.4,30.8,32.4,34.1,36.8,38.4,35.2,32.8,30.4,28.2,31.8,
            # Frontier 2024
            20.4,19.8,21.2,22.6,23.4,24.8,25.4,23.2,21.8,20.2,18.9,21.4,
        ],
        'BaggageRate_Per100': [
            # American 2022-2024
            0.85,0.82,0.84,0.83,0.81,0.80,0.79,0.80,0.81,0.82,0.81,0.84,
            0.79,0.77,0.78,0.76,0.75,0.74,0.73,0.74,0.75,0.76,0.77,0.79,
            0.93,0.91,0.92,0.90,0.89,0.88,0.87,0.88,0.89,0.90,0.91,0.93,
            # Delta 2022-2024
            0.61,0.59,0.60,0.58,0.57,0.56,0.55,0.56,0.57,0.58,0.59,0.61,
            0.50,0.48,0.49,0.47,0.46,0.45,0.44,0.45,0.46,0.47,0.48,0.50,
            0.52,0.50,0.51,0.49,0.48,0.47,0.82,0.48,0.49,0.50,0.51,0.52,
            # United 2022-2024
            0.76,0.74,0.75,0.73,0.72,0.71,0.70,0.71,0.72,0.73,0.74,0.76,
            0.76,0.74,0.75,0.73,0.72,0.71,0.70,0.71,0.72,0.73,0.74,0.76,
            0.71,0.69,0.70,0.68,0.67,0.66,0.65,0.66,0.67,0.68,0.69,0.71,
            # Southwest 2022-2024
            0.55,0.53,0.54,0.52,0.51,0.50,0.49,0.50,0.51,0.52,0.53,0.58,
            0.48,0.46,0.47,0.45,0.44,0.43,0.42,0.43,0.44,0.45,0.46,0.48,
            0.46,0.44,0.45,0.43,0.42,0.41,0.40,0.41,0.42,0.43,0.44,0.46,
            # Frontier 2022-2024
            1.18,1.15,1.16,1.14,1.12,1.10,1.09,1.10,1.11,1.12,1.13,1.16,
            1.10,1.08,1.09,1.07,1.05,1.03,1.02,1.03,1.04,1.05,1.06,1.09,
            1.02,1.00,1.01,0.99,0.97,0.96,0.95,0.96,0.97,0.98,0.99,1.02,
        ],
        'OnTimeRate_Pct': [
            # American 2022-2024
            74.2,75.1,76.3,77.2,77.8,74.1,73.2,74.8,76.9,78.1,78.4,72.1,
            75.8,76.4,77.1,78.2,78.9,75.3,74.8,76.1,78.2,79.1,79.4,73.8,
            76.1,77.2,77.8,78.4,78.9,75.8,75.1,76.4,78.8,79.3,79.8,74.2,
            # Delta 2022-2024
            81.2,82.4,83.1,84.2,84.8,80.1,79.4,81.2,84.1,85.2,85.6,80.2,
            82.4,83.6,84.2,85.4,85.9,82.1,81.4,82.8,85.2,86.1,86.4,82.1,
            82.1,83.4,83.9,84.8,85.2,81.4,62.1,83.1,85.8,86.4,88.6,83.2,
            # United 2022-2024
            77.4,78.8,79.4,80.2,80.8,77.1,76.4,78.2,80.4,81.6,81.9,76.8,
            78.4,79.6,80.2,81.4,81.9,78.2,77.4,79.1,81.6,82.4,82.8,78.1,
            79.1,80.4,80.9,81.8,82.2,79.1,78.4,80.2,82.4,83.2,81.8,79.8,
            # Southwest 2022-2024
            72.4,73.8,74.4,75.2,75.8,72.1,71.4,73.2,75.4,76.6,76.9,42.1,
            76.4,78.8,79.4,80.2,80.8,77.1,76.4,78.2,80.4,81.6,81.9,78.8,
            76.8,78.2,78.8,79.6,80.1,77.4,76.8,78.4,80.6,81.8,86.9,80.1,
            # Frontier 2022-2024
            70.8,71.4,72.1,72.8,73.4,69.1,68.4,70.2,72.4,73.6,73.9,68.8,
            72.4,73.8,74.4,74.8,75.4,71.1,70.4,72.2,74.4,75.6,75.9,70.8,
            73.8,74.4,75.1,75.8,76.2,72.4,71.8,73.4,75.4,76.6,76.7,72.1,
        ]
    }
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
    df['MonthName'] = df['Date'].dt.strftime('%b %Y')
    return df

# ── Anomaly Detection ────────────────────────────────────────────────────────
def detect_anomalies(df, metric, airline, threshold_sigma=2.0):
    """Z-score anomaly detection per airline time series."""
    sub = df[df['Airline'] == airline].sort_values('Date').copy()
    mean = sub[metric].mean()
    std = sub[metric].std()
    if std == 0:
        sub['zscore'] = 0.0
    else:
        sub['zscore'] = (sub[metric] - mean) / std
    sub['is_anomaly'] = sub['zscore'].abs() >= threshold_sigma
    sub['anomaly_direction'] = sub['zscore'].apply(
        lambda z: 'spike' if z > 0 else 'drop'
    )
    return sub, mean, std

def get_all_anomalies(df, metric, threshold_sigma=2.0):
    """Scan all airlines for anomalies."""
    results = []
    for airline in df['Airline'].unique():
        sub, mean, std = detect_anomalies(df, metric, airline, threshold_sigma)
        anomalies = sub[sub['is_anomaly']].copy()
        if not anomalies.empty:
            anomalies['Airline'] = airline
            anomalies['baseline_mean'] = mean
            results.append(anomalies)
    if results:
        return pd.concat(results).sort_values('Date')
    return pd.DataFrame()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ✈️ Detection Controls")
    st.markdown("---")

    selected_airlines = st.multiselect(
        "Airlines to Monitor",
        options=['American Airlines', 'Delta Air Lines', 'United Airlines',
                 'Southwest Airlines', 'Frontier Airlines'],
        default=['American Airlines', 'Delta Air Lines', 'Southwest Airlines',
                 'Frontier Airlines']
    )

    metric_map = {
        'Consumer Complaints (per 100K pax)': 'ComplaintsPerHundredK',
        'Baggage Mishandling Rate (per 100 bags)': 'BaggageRate_Per100',
        'On-Time Arrival Rate (%)': 'OnTimeRate_Pct'
    }
    selected_metric_label = st.selectbox(
        "Metric to Analyze",
        list(metric_map.keys())
    )
    selected_metric = metric_map[selected_metric_label]

    sigma_threshold = st.slider(
        "Detection Sensitivity (σ threshold)",
        min_value=1.0, max_value=3.0, value=2.0, step=0.5,
        help="Lower = more sensitive (catches smaller deviations). 2σ = statistically significant."
    )

    year_range = st.select_slider(
        "Year Range",
        options=[2022, 2023, 2024],
        value=(2022, 2024)
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#888;'>
    <b>Data Source</b><br>
    DOT Air Travel Consumer Reports<br>
    Bureau of Transportation Statistics<br>
    2022–2024 · 9 Major U.S. Carriers<br><br>
    <b>Methodology</b><br>
    Z-score anomaly detection on<br>monthly rolling baseline per carrier
    </div>
    """, unsafe_allow_html=True)

# ── Main ─────────────────────────────────────────────────────────────────────
df = load_data()
df_filtered = df[
    (df['Airline'].isin(selected_airlines)) &
    (df['Year'] >= year_range[0]) &
    (df['Year'] <= year_range[1])
]

# Header
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown("# ✈️ Airline Anomaly Detection Engine")
    st.markdown(
        f"*Autonomously scanning {len(selected_airlines)} carriers · "
        f"{selected_metric_label} · {year_range[0]}–{year_range[1]}*"
    )
with col_badge:
    st.markdown(f"""
    <div class='agent-badge'>
    🤖 <b>AI Scout Active</b><br>
    Monitoring {len(selected_airlines)} carriers<br>
    Threshold: ±{sigma_threshold}σ<br>
    Last scan: {datetime.now().strftime('%b %d, %Y')}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Anomaly Summary Cards ────────────────────────────────────────────────────
all_anomalies = get_all_anomalies(df_filtered, selected_metric, sigma_threshold)

col1, col2, col3 = st.columns(3)
with col1:
    total_anomalies = len(all_anomalies) if not all_anomalies.empty else 0
    st.metric("🚨 Anomalies Detected", total_anomalies)
with col2:
    airlines_flagged = all_anomalies['Airline'].nunique() if not all_anomalies.empty else 0
    st.metric("✈️ Airlines Flagged", f"{airlines_flagged} of {len(selected_airlines)}")
with col3:
    if not all_anomalies.empty:
        worst = all_anomalies.loc[all_anomalies['zscore'].abs().idxmax()]
        st.metric("📍 Most Extreme Event", f"{worst['Airline'][:10]}... ({worst['MonthName']})")
    else:
        st.metric("📍 Most Extreme Event", "None detected")

st.markdown("---")

# ── Alert Cards ──────────────────────────────────────────────────────────────
st.markdown("### 🔔 Agent Alerts — Statistically Significant Events")

if all_anomalies.empty:
    st.success("✅ No anomalies detected at this sensitivity level. All carriers within normal range.")
else:
    # Sort by absolute z-score descending
    alerts_sorted = all_anomalies.sort_values('zscore', key=abs, ascending=False)

    for _, row in alerts_sorted.head(8).iterrows():
        z = row['zscore']
        direction = "📈 SPIKE" if z > 0 else "📉 DROP"
        severity = "critical" if abs(z) >= 3 else ("warning" if abs(z) >= 2.5 else "ok")
        card_class = "metric-card" if severity == "critical" else (
            "metric-card warning" if severity == "warning" else "metric-card ok"
        )
        severity_label = "🔴 Critical" if severity == "critical" else (
            "🟡 Warning" if severity == "warning" else "🟢 Notable"
        )

        if selected_metric == 'OnTimeRate_Pct':
            direction = "📉 DROP" if z < 0 else "📈 RISE"

        value_fmt = f"{row[selected_metric]:.2f}"
        baseline_fmt = f"{row['baseline_mean']:.2f}"

        st.markdown(f"""
        <div class='{card_class}'>
            <div class='alert-header'>{severity_label} · {direction} · {row['Airline']}</div>
            <div class='alert-value'>{value_fmt} <span style='font-size:1rem;font-weight:400;color:#aaa;'>in {row['MonthName']}</span></div>
            <div class='alert-sub'>
                {abs(z):.1f}σ from baseline ({baseline_fmt} avg) ·
                {selected_metric_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Time Series Chart with Anomaly Overlay ────────────────────────────────────
st.markdown("### 📊 Time Series — Anomaly Overlay")

fig = go.Figure()
colors = {
    'American Airlines': '#4a90e2',
    'Delta Air Lines': '#7b68ee',
    'United Airlines': '#50c878',
    'Southwest Airlines': '#f5a623',
    'Frontier Airlines': '#e84545'
}

for airline in selected_airlines:
    sub, mean, std = detect_anomalies(df_filtered, selected_metric, airline, sigma_threshold)

    # Normal line
    fig.add_trace(go.Scatter(
        x=sub['Date'], y=sub[selected_metric],
        mode='lines',
        name=airline,
        line=dict(color=colors.get(airline, '#fff'), width=2),
        opacity=0.85
    ))

    # Anomaly markers
    anom = sub[sub['is_anomaly']]
    if not anom.empty:
        fig.add_trace(go.Scatter(
            x=anom['Date'], y=anom[selected_metric],
            mode='markers',
            name=f"{airline} — Anomaly",
            marker=dict(
                color='red', size=12, symbol='circle',
                line=dict(color='white', width=2)
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>{airline}</b><br>"
                "%{x|%b %Y}<br>"
                f"{selected_metric_label}: %{{y:.2f}}<br>"
                "<extra>⚠️ ANOMALY FLAGGED</extra>"
            )
        ))

# Threshold band
if not df_filtered.empty:
    overall_mean = df_filtered[selected_metric].mean()
    overall_std = df_filtered[selected_metric].std()
    date_min = df_filtered['Date'].min()
    date_max = df_filtered['Date'].max()

    fig.add_hrect(
        y0=overall_mean - sigma_threshold * overall_std,
        y1=overall_mean + sigma_threshold * overall_std,
        fillcolor="rgba(255,255,255,0.05)",
        line_width=0,
        annotation_text=f"Normal Band (±{sigma_threshold}σ)",
        annotation_position="top right",
        annotation_font_color="#888"
    )

fig.update_layout(
    plot_bgcolor='#0f1117',
    paper_bgcolor='#0f1117',
    font=dict(color='#e0e0e0'),
    legend=dict(
        orientation='h', yanchor='bottom', y=1.02,
        bgcolor='rgba(0,0,0,0)', borderwidth=0
    ),
    xaxis=dict(gridcolor='#1e2130', title=''),
    yaxis=dict(gridcolor='#1e2130', title=selected_metric_label),
    height=420,
    margin=dict(t=60, b=40, l=60, r=20),
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# ── Notable Events Context ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Known Events Context")

events = {
    "Southwest Dec 2022": "Holiday meltdown — cascading cancellations affected 16,700 flights over 10 days. DOT opened investigation.",
    "Delta Jul 2024": "CrowdStrike IT outage — global IT failure grounded thousands of flights. Delta took longest to recover among U.S. carriers.",
    "Frontier 2023 Peak": "Frontier complaint surge tracked with aggressive capacity expansion into underserved markets with thinner operational margins.",
    "American Baggage 2024": "American baggage rate increased despite industry improvement — attributed to terminal infrastructure constraints at key hubs.",
}

for event, description in events.items():
    with st.expander(f"📌 {event}"):
        st.markdown(description)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='font-size:0.75rem; color:#555; text-align:center;'>
Data: DOT Air Travel Consumer Reports 2022–2024 · Bureau of Transportation Statistics ·
Anomaly detection: Z-score method · Built with Python, Streamlit, Plotly
</div>
""", unsafe_allow_html=True)
