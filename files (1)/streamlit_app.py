"""
Fraud Transaction Detection — Streamlit GUI
--------------------------------------------
Run with:  streamlit run app.py
Requires:  pip install streamlit pandas numpy plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from agent import FraudDetectionAgent

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud Detection Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — Dark industrial theme
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f8fafc;
    color: #1e293b;
}
.main { background-color: #f8fafc; }
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}
section[data-testid="stSidebar"] * { color: #1e293b !important; }
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="metric-container"] * { color: #1e293b !important; }
h1, h2, h3 { font-family: 'Inter', sans-serif !important; font-weight: 700; color: #0f172a !important; }
p, span, label, div { color: #1e293b; }
.stButton > button {
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    padding: 10px 28px;
    font-size: 15px;
}
[data-testid="stDownloadButton"] > button {
    background: #2563eb !important;
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
[data-testid="stFileUploader"] {
    border: 2px dashed #cbd5e1;
    border-radius: 12px;
    background: #ffffff;
    padding: 10px;
}
[data-testid="stFileUploader"] * { color: #1e293b !important; }
details { background: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 10px !important; }
details * { color: #1e293b !important; }
table, th, td { color: #1e293b !important; }
input[type="number"], .stNumberInput input { color: #1e293b !important; background: #ffffff !important; }
.stSlider * { color: #1e293b !important; }
[data-testid="stAlert"] { color: #1e293b !important; }
code { color: #dc2626; background: #fef2f2; padding: 2px 6px; border-radius: 4px; }
.stSelectbox * { color: #1e293b !important; }
.stTextInput input { color: #1e293b !important; background: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ Detection Settings")
    st.divider()

    high_amount = st.number_input(
        "High-Amount Threshold ($)",
        min_value=100, max_value=1_000_000, value=5000, step=500,
        help="Transactions above this amount are flagged."
    )
    z_threshold = st.slider(
        "Z-Score Sensitivity",
        min_value=1.5, max_value=5.0, value=3.0, step=0.1,
        help="Lower = more sensitive. 3.0 is standard."
    )
    velocity_limit = st.number_input(
        "Max Transactions / Hour (per card)",
        min_value=1, max_value=50, value=5,
        help="Cards exceeding this are flagged for velocity fraud."
    )
    round_threshold = st.number_input(
        "Round-Number Flag Minimum ($)",
        min_value=100, max_value=10_000, value=1000, step=100,
        help="Round amounts (e.g., $2000) above this trigger suspicion."
    )

    st.divider()
    st.markdown("**ℹ️ Detection Rules**")
    st.caption("""
- High amount threshold  
- Statistical Z-score anomaly  
- IQR outlier detection  
- Round-number pattern  
- Rapid velocity (per card)  
- Unusual hour (1–4 AM)  
- Duplicate transaction  
    """)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown("""
<div style='padding: 30px 0 10px 0;'>
    <h1 style='font-size:2.8rem; margin:0; color:#f1f5f9;'>
        🛡️ Fraud Transaction<br>Detection Agent
    </h1>
    <p style='color:#64748b; font-size:1rem; margin-top:8px;'>
        Upload a CSV of transactions — the agent will flag suspicious activity using multi-layer detection.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# Sample CSV Download
# ─────────────────────────────────────────────

def generate_sample_csv() -> bytes:
    np.random.seed(42)
    n = 60
    timestamps = list(pd.date_range("2024-01-01 08:00", periods=n, freq="47min").astype(str))
    cards = list(np.random.choice(["4111-XXXX-1234", "5500-XXXX-4321", "3714-XXXX-9876"], n))

    # Inject rapid velocity cluster (last 5 rows)
    for i in range(5):
        timestamps[n - 5 + i] = "2024-01-03 02:15:00"
        cards[n - 5 + i] = "4111-XXXX-1234"

    data = {
        "transaction_id": [f"TXN-{i+1:04d}" for i in range(n)],
        "amount": np.concatenate([
            np.random.normal(250, 80, 50).clip(10),
            [8500, 12000, 500, 500, 999],
            np.random.normal(200, 50, 5).clip(10),
        ]).round(2),
        "timestamp": timestamps,
        "merchant": list(np.random.choice(["Amazon", "Walmart", "Shell", "Uber", "Unknown Vendor", "ATM"], n)),
        "card_number": cards,
        "category": list(np.random.choice(["retail", "travel", "fuel", "food", "withdrawal"], n)),
        "location": list(np.random.choice(["New York", "London", "Dubai", "Lagos", "Unknown"], n)),
    }

    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode()

col_dl, col_fmt = st.columns([1, 2])
with col_dl:
    st.download_button(
        "⬇️ Download Sample CSV",
        data=generate_sample_csv(),
        file_name="sample_transactions.csv",
        mime="text/csv",
    )
with col_fmt:
    with st.expander("📋 Expected CSV columns"):
        st.markdown("""
| Column | Required | Notes |
|---|---|---|
| `transaction_id` | No | Auto-generated if missing |
| `amount` | **Yes** | Numeric values |
| `timestamp` | No | Any parseable date/time |
| `card_number` | No | For velocity/duplicate checks |
| `merchant` | No | Displayed in results |
| `category` | No | Displayed in results |
| `location` | No | Displayed in results |

Column names are case-insensitive and common variants are auto-detected.
        """)

# ─────────────────────────────────────────────
# File Upload
# ─────────────────────────────────────────────

st.markdown("### 📂 Upload Transaction CSV")
uploaded_file = st.file_uploader(
    "Drag & drop or click to browse",
    type=["csv"],
    label_visibility="collapsed"
)

# ─────────────────────────────────────────────
# Run Detection
# ─────────────────────────────────────────────

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.success(f"Loaded **{len(df_raw):,}** rows × **{len(df_raw.columns)}** columns.")

    with st.expander("🔍 Preview raw data"):
        st.dataframe(df_raw.head(10), use_container_width=True)

    st.markdown("---")

    if st.button("🚀 Run Fraud Detection"):
        config = {
            "high_amount_threshold": high_amount,
            "z_score_threshold": z_threshold,
            "velocity_window": velocity_limit,
            "round_amount_threshold": round_threshold,
        }

        with st.spinner("Analyzing transactions…"):
            agent = FraudDetectionAgent(config=config)
            report = agent.analyze(df_raw)
            results_df = agent.results_to_dataframe(report)

        # ── KPI Metrics ───────────────────────────
        st.markdown("### 📊 Detection Summary")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Total Transactions", f"{report.total_transactions:,}")
        with m2:
            st.metric("Flagged as Fraud", f"{report.flagged_count:,}",
                      delta=f"{report.fraud_rate*100:.1f}% fraud rate",
                      delta_color="inverse")
        with m3:
            clean = report.total_transactions - report.flagged_count
            st.metric("Clean Transactions", f"{clean:,}")
        with m4:
            st.metric("Mean Amount", f"${report.stats['mean_amount']:,.2f}")

        st.info(report.summary)

        # ── Charts ────────────────────────────────
        st.markdown("### 📈 Visualizations")
        chart_col1, chart_col2 = st.columns(2)

        # Pie chart — fraud vs clean
        with chart_col1:
            pie_df = pd.DataFrame({
                "Status": ["Clean", "Flagged"],
                "Count": [clean, report.flagged_count],
            })
            fig_pie = px.pie(
                pie_df, names="Status", values="Count",
                color="Status",
                color_discrete_map={"Clean": "#22c55e", "Flagged": "#ef4444"},
                title="Fraud vs Clean",
                hole=0.5,
            )
            fig_pie.update_layout(
                paper_bgcolor="#111827", plot_bgcolor="#111827",
                font_color="#e2e8f0", title_font_size=16,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Risk level bar
        with chart_col2:
            risk_counts = results_df["Risk Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            color_map = {
                "LOW": "#22c55e", "MEDIUM": "#eab308",
                "HIGH": "#f97316", "CRITICAL": "#ef4444"
            }
            risk_counts["Risk Level"] = pd.Categorical(risk_counts["Risk Level"], categories=order, ordered=True)
            risk_counts = risk_counts.sort_values("Risk Level")
            fig_bar = px.bar(
                risk_counts, x="Risk Level", y="Count",
                color="Risk Level",
                color_discrete_map=color_map,
                title="Transactions by Risk Level",
            )
            fig_bar.update_layout(
                paper_bgcolor="#111827", plot_bgcolor="#111827",
                font_color="#e2e8f0", showlegend=False, title_font_size=16,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Risk score distribution
        scores = [r.risk_score for r in report.results]
        fig_hist = px.histogram(
            x=scores, nbins=30,
            labels={"x": "Risk Score"},
            title="Risk Score Distribution",
            color_discrete_sequence=["#3b82f6"],
        )
        fig_hist.add_vline(x=0.40, line_dash="dash", line_color="#ef4444",
                           annotation_text="Fraud threshold (0.40)")
        fig_hist.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            font_color="#e2e8f0", title_font_size=16,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Results Table ─────────────────────────
        st.markdown("### 🗂️ Transaction Results")

        # Filter controls
        fc1, fc2 = st.columns([1, 2])
        with fc1:
            show_filter = st.selectbox(
                "Show",
                ["All", "Fraud Only", "Clean Only", "HIGH + CRITICAL"]
            )
        with fc2:
            search = st.text_input("🔎 Search by Transaction ID or explanation", "")

        display_df = results_df.copy()
        if show_filter == "Fraud Only":
            display_df = display_df[display_df["Fraud?"] == "🚨 YES"]
        elif show_filter == "Clean Only":
            display_df = display_df[display_df["Fraud?"] == "✅ NO"]
        elif show_filter == "HIGH + CRITICAL":
            display_df = display_df[display_df["Risk Level"].isin(["HIGH", "CRITICAL"])]

        if search:
            mask = (
                display_df["Transaction ID"].str.contains(search, case=False, na=False) |
                display_df["Explanation"].str.contains(search, case=False, na=False)
            )
            display_df = display_df[mask]

        st.dataframe(
            display_df,
            use_container_width=True,
            height=420,
            column_config={
                "Risk Score": st.column_config.ProgressColumn(
                    "Risk Score",
                    help="0 = safe, 1 = certain fraud",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
                "Fraud?": st.column_config.TextColumn("Fraud?", width="small"),
                "Risk Level": st.column_config.TextColumn("Risk Level", width="small"),
            }
        )

        # ── Detail Expander per fraud ─────────────
        fraudulent = [r for r in report.results if r.is_fraud]
        if fraudulent:
            st.markdown("### 🔎 Fraud Case Details")
            for r in fraudulent[:20]:   # cap at 20 to avoid UI flooding
                with st.expander(f"🚨 {r.transaction_id} — Score: {r.risk_score:.2f} ({r.risk_level})"):
                    st.markdown(f"**Explanation:** {r.explanation}")
                    if r.triggered_rules:
                        st.markdown("**Triggered Rules:**")
                        for rule in r.triggered_rules:
                            st.markdown(f"- {rule}")

        # ── Export ────────────────────────────────
        st.markdown("### 💾 Export Results")
        csv_out = results_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Full Results CSV",
            data=csv_out,
            file_name="fraud_detection_results.csv",
            mime="text/csv",
        )

else:
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px; color:#334155;'>
        <div style='font-size:4rem;'>📤</div>
        <h3 style='color:#475569;'>Upload a CSV to get started</h3>
        <p>Download the sample file above to test the agent instantly.</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.divider()
st.caption("Fraud Detection Agent · Built with Streamlit + Python · Rule-based & Statistical Anomaly Detection")