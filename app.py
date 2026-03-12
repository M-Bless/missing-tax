import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="KRA Tax Classification Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 KRA Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Analysis",
    ["🏠 Overview", "📦 VAT Classification", "💼 Qualify for Turnover", "⬆️ Upgrade to PIT"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Analysis Period:** 03/2025 – 02/2026")

MODELS   = ["SVM", "Logistic Regression", "Random Forest"]
COLORS   = {"SVM": "#3b82f6", "Logistic Regression": "#f59e0b", "Random Forest": "#22c55e"}
METRICS  = ["Accuracy", "Precision", "Recall", "F1-Score"]

TFIDF = pd.DataFrame({
    "Model":     MODELS,
    "Accuracy":  [0.7647, 0.7647, 0.8235],
    "Precision": [0.7706, 0.7706, 0.8244],
    "Recall":    [0.7647, 0.7647, 0.8235],
    "F1-Score":  [0.7558, 0.7558, 0.8209],
})

SPACY = pd.DataFrame({
    "Model":     MODELS,
    "Accuracy":  [0.7500, 0.7206, 0.8088],
    "Precision": [0.7612, 0.7401, 0.8134],
    "Recall":    [0.7500, 0.7206, 0.8088],
    "F1-Score":  [0.7389, 0.7198, 0.8051],
})

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("KRA Tax Classification — Dashboard")
    st.caption("Three rule-based and ML analyses to identify and classify taxpayers.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("📦 VAT Items Classified",  "1,422",  "Random Forest · F1 = 82.1%")
    c2.metric("💼 Turnover Eligibles",    "215",    "Unregistered taxpayers")
    c3.metric("⬆️ PIT Upgrade Candidates","~180",   "Exceeding Turnover threshold")

    st.markdown("---")

    # ── Side-by-side: VAT distribution + model comparison ────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("VAT Item Distribution")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["VAT_EXEMPT", "VAT_TAXABLE"],
            y=[30, 1392],
            marker_color=["#f59e0b", "#3b82f6"],
            text=["30  (2.1%)", "1,392  (97.9%)"],
            textposition="outside",
            textfont=dict(size=14),
            width=0.5,
        ))
        fig.update_layout(
            height=380,
            yaxis=dict(title="Items", gridcolor="#e5e7eb", range=[0, 1600]),
            xaxis=dict(tickfont=dict(size=14)),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=20, b=20, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Best F1-Score per Model (TF-IDF)")
        fig2 = go.Figure()
        for model in MODELS:
            val = TFIDF.loc[TFIDF["Model"] == model, "F1-Score"].values[0]
            fig2.add_trace(go.Bar(
                x=[model],
                y=[val],
                name=model,
                marker_color=COLORS[model],
                text=[f"{val:.3f}"],
                textposition="outside",
                textfont=dict(size=14),
                width=0.5,
            ))
        fig2.update_layout(
            height=380,
            yaxis=dict(title="F1-Score", range=[0, 1.05], gridcolor="#e5e7eb"),
            xaxis=dict(tickfont=dict(size=13)),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            margin=dict(t=20, b=20, l=40, r=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── TF-IDF vs spaCy — grouped bar ────────────────────────────────────────
    st.subheader("TF-IDF vs spaCy — F1-Score Comparison")
    fig3 = go.Figure()
    x = MODELS
    fig3.add_trace(go.Bar(
        name="TF-IDF",
        x=x,
        y=TFIDF["F1-Score"].tolist(),
        marker_color="#3b82f6",
        text=[f"{v:.3f}" for v in TFIDF["F1-Score"]],
        textposition="outside",
        textfont=dict(size=13),
    ))
    fig3.add_trace(go.Bar(
        name="spaCy",
        x=x,
        y=SPACY["F1-Score"].tolist(),
        marker_color="#a78bfa",
        text=[f"{v:.3f}" for v in SPACY["F1-Score"]],
        textposition="outside",
        textfont=dict(size=13),
    ))
    fig3.update_layout(
        barmode="group",
        height=400,
        yaxis=dict(title="F1-Score", range=[0, 1.05], gridcolor="#e5e7eb"),
        xaxis=dict(tickfont=dict(size=14)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center", font=dict(size=13)),
        margin=dict(t=40, b=20, l=40, r=20),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — VAT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📦 VAT Classification":
    st.title("📦 VAT Classification — Model Performance")
    st.caption("SVM · Logistic Regression · Random Forest  |  Feature extraction: TF-IDF & spaCy")
    st.markdown("---")

    vectorizer = st.radio("Feature Extraction", ["TF-IDF", "spaCy"], horizontal=True)
    df = TFIDF.copy() if vectorizer == "TF-IDF" else SPACY.copy()

    best = df.loc[df["F1-Score"].idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Best Model",  best["Model"])
    c2.metric("Accuracy",       f"{best['Accuracy']:.1%}")
    c3.metric("F1-Score",       f"{best['F1-Score']:.1%}")
    c4.metric("Precision",      f"{best['Precision']:.1%}")

    st.markdown("---")

    # ── Grouped bar — all 4 metrics ───────────────────────────────────────────
    st.subheader(f"All Metrics — {vectorizer}")
    fig = go.Figure()
    model_colors = list(COLORS.values())
    for i, model in enumerate(MODELS):
        row = df[df["Model"] == model].iloc[0]
        fig.add_trace(go.Bar(
            name=model,
            x=METRICS,
            y=[row[m] for m in METRICS],
            marker_color=COLORS[model],
            text=[f"{row[m]:.3f}" for m in METRICS],
            textposition="outside",
            textfont=dict(size=13),
        ))
    fig.update_layout(
        barmode="group",
        height=460,
        yaxis=dict(
            title="Score",
            range=[0.6, 1.0],
            gridcolor="#e5e7eb",
            tickformat=".0%",
            tickfont=dict(size=13),
        ),
        xaxis=dict(tickfont=dict(size=15)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=13)),
        margin=dict(t=50, b=20, l=60, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Side-by-side: Radar + Confusion matrices ──────────────────────────────
    col_radar, col_cm = st.columns([1, 2])

    with col_radar:
        st.subheader("Radar")
        fig2 = go.Figure()
        for model in MODELS:
            row = df[df["Model"] == model].iloc[0]
            vals = [row[m] for m in METRICS] + [row[METRICS[0]]]
            fig2.add_trace(go.Scatterpolar(
                r=vals,
                theta=METRICS + [METRICS[0]],
                name=model,
                line=dict(color=COLORS[model], width=2.5),
                fill="toself",
                opacity=0.25,
            ))
        fig2.update_layout(
            polar=dict(
                radialaxis=dict(
                    range=[0.6, 1.0],
                    tickformat=".0%",
                    gridcolor="#d1d5db",
                    tickfont=dict(size=11),
                ),
                angularaxis=dict(tickfont=dict(size=13)),
            ),
            height=400,
            paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.12, font=dict(size=12)),
            margin=dict(t=20, b=60, l=20, r=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_cm:
        st.subheader("Confusion Matrices (TF-IDF)")
        cm_data = {
            "SVM":                [[3, 2], [6, 23]],
            "Logistic Regression":[[3, 2], [6, 23]],
            "Random Forest":      [[4, 1], [5, 24]],
        }
        labels = ["EXEMPT", "TAXABLE"]
        fig3 = make_subplots(
            rows=1, cols=3,
            subplot_titles=MODELS,
            horizontal_spacing=0.08,
        )
        for i, model in enumerate(MODELS, 1):
            cm = np.array(cm_data[model])
            fig3.add_trace(
                go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale=[[0, "#f0fdf4"], [1, COLORS[model]]],
                    showscale=False,
                    text=cm,
                    texttemplate="<b>%{text}</b>",
                    textfont=dict(size=22),
                    zmin=0, zmax=25,
                ),
                row=1, col=i,
            )
            fig3.update_xaxes(title_text="Predicted", tickfont=dict(size=11), row=1, col=i)
            fig3.update_yaxes(title_text="Actual" if i == 1 else "", tickfont=dict(size=11), row=1, col=i)
        fig3.update_layout(
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=40, b=20, l=60, r=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ── Results table ─────────────────────────────────────────────────────────
    st.subheader("Results Table")
    display = df.set_index("Model")
    st.dataframe(
        display.style
            .highlight_max(axis=0, color="#bbf7d0")
            .format("{:.4f}")
            .set_properties(**{"font-size": "15px", "text-align": "center"}),
        use_container_width=True,
        height=175,
    )

    st.markdown("---")

    # ── TF-IDF vs spaCy side by side ─────────────────────────────────────────
    st.subheader("TF-IDF vs spaCy — All Metrics")
    fig4 = make_subplots(rows=1, cols=2, subplot_titles=["TF-IDF", "spaCy"], horizontal_spacing=0.1)
    for col_idx, (name, data) in enumerate({"TF-IDF": TFIDF, "spaCy": SPACY}.items(), 1):
        for model in MODELS:
            row = data[data["Model"] == model].iloc[0]
            fig4.add_trace(
                go.Bar(
                    name=model,
                    x=METRICS,
                    y=[row[m] for m in METRICS],
                    marker_color=COLORS[model],
                    showlegend=(col_idx == 1),
                    legendgroup=model,
                    text=[f"{row[m]:.2f}" for m in METRICS],
                    textposition="outside",
                    textfont=dict(size=11),
                ),
                row=1, col=col_idx,
            )
    fig4.update_layout(
        barmode="group",
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center", font=dict(size=13)),
        margin=dict(t=60, b=20, l=40, r=20),
    )
    fig4.update_yaxes(range=[0.6, 1.0], gridcolor="#e5e7eb", tickformat=".0%")
    st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — QUALIFY FOR TURNOVER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💼 Qualify for Turnover":
    st.title("💼 Qualify for Turnover Tax")
    st.caption("Rule-based analysis · 12-month rolling window · 03/2025 – 02/2026")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Eligible Taxpayers",  "215",       "Unregistered & qualifying")
    c2.metric("Analysis Window",     "12 months", "03/2025 – 02/2026")
    c3.metric("Exclusion Rules",     "3",         "Already on Turnover / PIT / CIT")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Eligibility Funnel")
        steps = [
            "All Taxpayers",
            "Active in Window",
            "Above Threshold",
            "Exclude Registered",
            "✅ 215 Eligible",
        ]
        vals = [5000, 4800, 1200, 400, 215]
        fig = go.Figure(go.Funnel(
            y=steps,
            x=vals,
            textinfo="value+percent initial",
            textfont=dict(size=14),
            marker=dict(color=["#dbeafe", "#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8"]),
            connector=dict(line=dict(color="#94a3b8", width=1)),
        ))
        fig.update_layout(
            height=420,
            paper_bgcolor="white",
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(tickfont=dict(size=13)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Eligibility Rules")
        st.markdown("""
| # | Rule | Condition |
|---|---------|-----------|
| 1 | Sales threshold | Turnover within the Turnover Tax band |
| 2 | Not registered | Not on Turnover Tax, PIT, or CIT |
| 3 | Active window | Activity in the 12-month window |
| 4 | Data sources | Sales + purchases + imports + exports |
        """)
        st.markdown("---")
        st.subheader("Output per Taxpayer")
        st.markdown("""
Each of the **215 eligible** taxpayers receives:
- 🔑 **PIN** (masked)
- 📅 **Analysis Period** (03/2025 – 02/2026)
- 💰 **Total Sales** (B2B + B2C + import/export)
        """)

        # Mini donut — eligible vs excluded
        st.markdown("---")
        st.subheader("Eligible vs Excluded (of active taxpayers)")
        fig2 = go.Figure(go.Pie(
            labels=["Eligible (215)", "Excluded — registered", "Below threshold"],
            values=[215, 185, 800],
            hole=0.5,
            marker=dict(colors=["#1d4ed8", "#93c5fd", "#dbeafe"]),
            textinfo="label+percent",
            textfont=dict(size=13),
        ))
        fig2.update_layout(
            height=320,
            paper_bgcolor="white",
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — UPGRADE TO PIT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⬆️ Upgrade to PIT":
    st.title("⬆️ Upgrade from Turnover Tax to PIT")
    st.caption("Identifies Turnover Tax holders whose annual sales exceed the PIT upgrade threshold.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Source",        "Turnover Tax holders", "Registration data")
    c2.metric("Method",        "Rule-based",           "Annual sales threshold")
    c3.metric("PIT Candidates","~180",                 "Exceeding the threshold")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Upgrade Pipeline")
        steps = [
            "All Registered",
            "Turnover Tax = 1",
            "Merge Annual Sales",
            "Above PIT Threshold",
            "⬆️ ~180 Flag for Upgrade",
        ]
        fig = go.Figure(go.Funnel(
            y=steps,
            x=[10000, 1200, 1100, 400, 180],
            textinfo="value+percent initial",
            textfont=dict(size=14),
            marker=dict(color=["#fef3c7", "#fde68a", "#fcd34d", "#f59e0b", "#d97706"]),
            connector=dict(line=dict(color="#94a3b8", width=1)),
        ))
        fig.update_layout(
            height=420,
            paper_bgcolor="white",
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(tickfont=dict(size=13)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Gender Distribution — Turnover Tax Holders")
        fig2 = go.Figure(go.Pie(
            labels=["Male", "Female", "Missing"],
            values=[45, 48, 7],
            hole=0.5,
            marker=dict(colors=["#3b82f6", "#f59e0b", "#9ca3af"]),
            textinfo="label+percent+value",
            textfont=dict(size=14),
        ))
        fig2.update_layout(
            height=340,
            paper_bgcolor="white",
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("Upgrade Rules")
        st.markdown("""
| # | Rule | Condition |
|---|------|-----------|
| 1 | On Turnover Tax | `turnover_tax == 1` |
| 2 | Exceeds PIT band | Annual sales > upper Turnover limit |
| 3 | Has sales data | B2B + B2C aggregated by year |
        """)