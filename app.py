"""
app.py
------
Streamlit Web UI for Restaurant Review Sentiment Analysis
Run with:  streamlit run app.py
"""

import os
import sys
import joblib
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ─── Path Setup ───────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR    = os.path.join(BASE_DIR, "outputs")
sys.path.insert(0, BASE_DIR)

from src.preprocessing import clean_text

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer — Restaurant Reviews",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}

/* ── Background ── */
.stApp {
    background: #0b0f1a;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f1422 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] strong {
    color: #e2e8f0 !important;
}

/* ── Hero ── */
.hero-wrap {
    padding: 3rem 0 2rem;
    text-align: center;
}
.hero-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.15;
    color: #f1f5f9;
    margin: 0 0 0.75rem;
}
.hero-title span {
    background: linear-gradient(90deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: #64748b;
    font-size: 1rem;
    font-weight: 400;
}

/* ── Glass Card ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.25rem;
}

/* ── Section Label ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.6rem;
}
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #cbd5e1;
    margin-bottom: 1.1rem;
}

/* ── Sentiment Badges ── */
.badge {
    display: inline-block;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.35rem 1.2rem;
    border-radius: 100px;
}
.badge-positive {
    background: rgba(52, 211, 153, 0.15);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}
.badge-neutral {
    background: rgba(129, 140, 248, 0.15);
    color: #818cf8;
    border: 1px solid rgba(129, 140, 248, 0.3);
}
.badge-negative {
    background: rgba(248, 113, 113, 0.15);
    color: #f87171;
    border: 1px solid rgba(248, 113, 113, 0.3);
}

/* ── Confidence Number ── */
.conf-score {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1;
    color: #f1f5f9;
    margin: 0.6rem 0 0.2rem;
}
.conf-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
}

/* ── Inputs ── */
textarea {
    background: rgba(255,255,255,0.04) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    caret-color: #818cf8;
}
textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.18) !important;
}

/* ── Select box ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: #ffffff;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.04em;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2rem;
    width: 100%;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35);
}
.stButton > button:hover {
    opacity: 0.92;
    transform: translateY(-1px);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px 8px 0 0;
    color: #64748b;
    font-weight: 500;
    font-size: 0.88rem;
    letter-spacing: 0.03em;
    padding: 0.6rem 1.2rem;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.12) !important;
    color: #818cf8 !important;
    border-bottom: 2px solid #6366f1 !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(255,255,255,0.07) !important;
    margin: 1rem 0 !important;
}

/* ── Status note ── */
.status-note {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 0.5rem;
}
.status-high { color: #34d399; font-weight: 600; }
.status-mid  { color: #f59e0b; font-weight: 600; }
.status-low  { color: #f87171; font-weight: 600; }

/* ── Metric grid ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
}
.metric-cell {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-cell .m-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
}
.metric-cell .m-lbl {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-top: 0.15rem;
}

/* ── Table ── */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Artifacts ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model_path = os.path.join(MODELS_DIR, "logistic_regression.joblib")
    vec_path   = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        return None, None
    return joblib.load(model_path), joblib.load(vec_path)


model, vectorizer = load_artifacts()

# ─── Sentiment Config ─────────────────────────────────────────────────────────
SENTIMENT = {
    "Positive": {"badge": "badge-positive", "color": "#34d399", "accent": "rgba(52,211,153,0.15)"},
    "Neutral":  {"badge": "badge-neutral",  "color": "#818cf8", "accent": "rgba(129,140,248,0.15)"},
    "Negative": {"badge": "badge-negative", "color": "#f87171", "accent": "rgba(248,113,113,0.15)"},
}

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Sentiment Analyzer")
    st.markdown("NLP pipeline trained on **10,000 Yelp reviews**")
    st.divider()

    if model is None:
        st.error("Model not found. Run `py main.py` first.")
    else:
        st.markdown("**Status**")
        st.markdown('<span style="color:#34d399; font-weight:600;">Model loaded</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**Models Trained**")
    st.markdown("- Naive Bayes  \n- Logistic Regression  \n- SVM (LinearSVC)")
    st.divider()
    st.markdown(
        """
        <div class="metric-grid">
            <div class="metric-cell"><div class="m-val">10k</div><div class="m-lbl">Reviews</div></div>
            <div class="metric-cell"><div class="m-val">0.82</div><div class="m-lbl">Best F1</div></div>
            <div class="metric-cell"><div class="m-val">10k</div><div class="m-lbl">Features</div></div>
            <div class="metric-cell"><div class="m-val">3</div><div class="m-lbl">Classes</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-label">NLP &nbsp;&bull;&nbsp; Machine Learning &nbsp;&bull;&nbsp; Yelp Dataset</div>
    <h1 class="hero-title">Restaurant Review<br><span>Sentiment Analysis</span></h1>
    <p class="hero-sub">Classify any review as Positive, Neutral, or Negative using Logistic Regression + TF-IDF</p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_overview = st.tabs(["Single Review", "Batch Analysis", "Model Overview"])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Single Review
# ══════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Input</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Paste or type a restaurant review</div>", unsafe_allow_html=True)

    examples = {
        "Select an example —": "",
        "Positive review": "The food was absolutely amazing! The pasta was perfectly cooked and the service was incredibly friendly. Highly recommend!",
        "Neutral review": "The place was okay. Food came out in reasonable time and tasted fine. Nothing special but nothing bad either.",
        "Negative review": "Terrible experience. Waited over an hour for our food and when it finally arrived, it was cold and tasteless. Never coming back.",
    }
    choice = st.selectbox("Example", list(examples.keys()), label_visibility="collapsed")
    prefill = examples[choice]

    review_input = st.text_area(
        "Review text",
        value=prefill,
        height=130,
        placeholder="e.g. The lamb chops were perfectly seasoned and the ambience was cozy...",
        label_visibility="collapsed",
    )
    analyze_btn = st.button("Analyze Sentiment", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze_btn:
        if not review_input.strip():
            st.warning("Please enter a review to analyze.")
        elif model is None:
            st.error("Model not loaded. Run `py main.py` first.")
        else:
            cleaned = clean_text(review_input)
            vec     = vectorizer.transform([cleaned])
            pred    = model.predict(vec)[0]
            proba   = model.predict_proba(vec)[0]
            cls     = model.classes_.tolist()
            prob_d  = dict(zip(cls, proba))
            conf    = prob_d[pred]
            cfg     = SENTIMENT[pred]

            ordered = ["Positive", "Neutral", "Negative"]

            col_l, col_r = st.columns([1, 1.8], gap="large")

            with col_l:
                # Prediction card
                st.markdown(
                    f"""
                    <div class="card" style="border-color:{cfg['color']}22; background:{cfg['accent']};">
                        <div class="conf-label">Predicted Sentiment</div>
                        <div style="margin:0.6rem 0;">
                            <span class="badge {cfg['badge']}">{pred}</span>
                        </div>
                        <div class="conf-score">{conf:.0%}</div>
                        <div class="conf-label">Confidence</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Confidence status note
                if conf >= 0.75:
                    note_cls, note = "status-high", "High confidence"
                elif conf >= 0.50:
                    note_cls, note = "status-mid", "Moderate confidence"
                else:
                    note_cls, note = "status-low", "Low confidence — borderline"
                st.markdown(
                    f'<p class="status-note"><span class="{note_cls}">{note}</span></p>',
                    unsafe_allow_html=True,
                )

            with col_r:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-label'>Probability Distribution</div>", unsafe_allow_html=True)

                bar_colors = [SENTIMENT[c]["color"] for c in ordered]
                bar_vals   = [prob_d.get(c, 0) for c in ordered]

                fig = go.Figure(go.Bar(
                    x=bar_vals,
                    y=ordered,
                    orientation="h",
                    marker=dict(
                        color=[SENTIMENT[c]["color"] for c in ordered],
                        opacity=0.85,
                        line=dict(width=0),
                    ),
                    text=[f"{v:.1%}" for v in bar_vals],
                    textposition="outside",
                    textfont=dict(color="#94a3b8", size=12, family="Inter"),
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        range=[0, 1.2],
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.05)",
                        zeroline=False,
                        tickformat=".0%",
                        color="#475569",
                        tickfont=dict(family="Inter", size=11),
                    ),
                    yaxis=dict(
                        showgrid=False,
                        color="#94a3b8",
                        tickfont=dict(family="Inter", size=13, color="#94a3b8"),
                    ),
                    margin=dict(l=0, r=30, t=10, b=10),
                    height=190,
                    showlegend=False,
                    bargap=0.35,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("View preprocessed tokens"):
                st.code(cleaned or "(empty after cleaning)", language=None)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Batch
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Input</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Enter multiple reviews, one per line</div>", unsafe_allow_html=True)

    batch_input = st.text_area(
        "batch",
        height=190,
        placeholder="Great food and friendly staff.\nThe wait was too long and food was cold.\nDecent place, nothing extraordinary.",
        label_visibility="collapsed",
    )
    batch_btn = st.button("Run Batch Analysis", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if batch_btn:
        if not batch_input.strip():
            st.warning("Please enter at least one review.")
        elif model is None:
            st.error("Model not loaded. Run `py main.py` first.")
        else:
            lines        = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
            cleaned_list = [clean_text(l) for l in lines]
            vecs         = vectorizer.transform(cleaned_list)
            preds        = model.predict(vecs)
            probas       = model.predict_proba(vecs)

            rows = []
            for review, pred, prob in zip(lines, preds, probas):
                rows.append({
                    "Review":     review[:90] + ("..." if len(review) > 90 else ""),
                    "Sentiment":  pred,
                    "Confidence": f"{max(prob):.1%}",
                })
            df = pd.DataFrame(rows)

            st.divider()

            col_pie, col_tbl = st.columns([1, 2.2], gap="large")

            with col_pie:
                counts      = df["Sentiment"].value_counts()
                pie_colors  = [SENTIMENT.get(s, {}).get("color", "#888") for s in counts.index]
                pie_fig = go.Figure(go.Pie(
                    labels=counts.index,
                    values=counts.values,
                    hole=0.65,
                    marker=dict(colors=pie_colors, line=dict(color="#0b0f1a", width=3)),
                    textinfo="label+percent",
                    textfont=dict(color="#94a3b8", size=12, family="Inter"),
                    insidetextfont=dict(color="#94a3b8"),
                ))
                pie_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=240,
                    annotations=[dict(
                        text=f"<b>{len(lines)}</b><br><span style='font-size:11px'>reviews</span>",
                        x=0.5, y=0.5, font_size=18, font_color="#e2e8f0",
                        showarrow=False,
                    )],
                )
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-label'>Distribution</div>", unsafe_allow_html=True)
                st.plotly_chart(pie_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_tbl:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-label'>Results</div>", unsafe_allow_html=True)

                def style_row(val):
                    colors = {"Positive": "#0d2a1f", "Neutral": "#13153c", "Negative": "#2a0d0d"}
                    text   = {"Positive": "#34d399", "Neutral": "#818cf8", "Negative": "#f87171"}
                    return f"background-color:{colors.get(val,'#111')}; color:{text.get(val,'#ddd')}; font-weight:600;"

                styled = df.style.map(style_row, subset=["Sentiment"])
                st.dataframe(styled, use_container_width=True, hide_index=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, "results.csv", "text/csv", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Model Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("<div class='section-label'>Pipeline Outputs</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Charts generated from training on 10,000 Yelp reviews</div>", unsafe_allow_html=True)

    chart_files = [
        ("sentiment_distribution.png",                "Sentiment Distribution"),
        ("model_comparison.png",                       "Model Performance Comparison"),
        ("wordclouds.png",                             "Word Clouds by Sentiment Class"),
        ("per_class_f1.png",                           "Per-Class F1 Score"),
        ("confusion_matrix_logistic_regression.png",  "Confusion Matrix — Logistic Regression"),
        ("confusion_matrix_naive_bayes.png",           "Confusion Matrix — Naive Bayes"),
        ("confusion_matrix_svm.png",                   "Confusion Matrix — SVM"),
    ]

    available = [(f, t) for f, t in chart_files if os.path.exists(os.path.join(OUT_DIR, f))]

    if not available:
        st.info("No charts found. Run `py main.py` first to generate them.")
    else:
        # Show first two side-by-side, rest full width
        if len(available) >= 2:
            c1, c2 = st.columns(2, gap="medium")
            for col, (fname, title) in zip([c1, c2], available[:2]):
                with col:
                    st.markdown(f"<div class='card'><div class='section-label'>{title}</div>", unsafe_allow_html=True)
                    st.image(os.path.join(OUT_DIR, fname), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            remaining = available[2:]
        else:
            remaining = available

        for fname, title in remaining:
            st.markdown(f"<div class='card'><div class='section-label'>{title}</div>", unsafe_allow_html=True)
            st.image(os.path.join(OUT_DIR, fname), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2.5rem 0 1rem; color:#334155; font-size:0.78rem; letter-spacing:0.08em; text-transform:uppercase;">
    Sentiment Analysis &nbsp;&bull;&nbsp; Logistic Regression + TF-IDF &nbsp;&bull;&nbsp; Yelp Restaurant Reviews
</div>
""", unsafe_allow_html=True)
