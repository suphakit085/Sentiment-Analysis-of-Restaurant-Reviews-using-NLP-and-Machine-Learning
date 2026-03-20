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
    page_title="Sentiment Analyzer - Restaurant Reviews",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Fixed accent colors (work on both light & dark) ─────────────────────────
ACCENT = "#06b6d4"
C_POS  = "#059669"
C_NEU  = "#6b7280"
C_NEG  = "#dc2626"

# ─── Custom CSS (theme-agnostic: works on BOTH light & dark) ─────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
}}

/* ── Hero ── */
.hero {{
    padding: 2.5rem 0 2rem;
    text-align: center;
}}
.hero-tag {{
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {ACCENT};
    background: rgba(6,182,212,0.1);
    padding: 0.25rem 0.85rem;
    border-radius: 4px;
    margin-bottom: 1rem;
    border: 1px solid rgba(6,182,212,0.25);
}}
.hero h1 {{
    font-size: 2.4rem;
    font-weight: 700;
    color: inherit;
    margin: 0 0 0.5rem;
    letter-spacing: -0.02em;
    line-height: 1.2;
}}
.hero p {{
    color: {C_NEU};
    font-size: 0.92rem;
    margin: 0;
}}

/* ── Card — uses semi-transparent bg that adapts to any background ── */
.card {{
    background: rgba(128,128,128,0.06);
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}}

/* ── Section head ── */
.sh {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {C_NEU};
    margin-bottom: 0.5rem;
}}
.sdesc {{
    font-size: 0.92rem;
    font-weight: 500;
    color: inherit;
    opacity: 0.7;
    margin-bottom: 1rem;
}}

/* ── Badges ── */
.badge {{
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.25rem 0.9rem;
    border-radius: 4px;
}}
.badge-positive {{
    background: rgba(5,150,105,0.12);
    color: {C_POS};
    border: 1px solid rgba(5,150,105,0.3);
}}
.badge-neutral {{
    background: rgba(107,114,128,0.1);
    color: {C_NEU};
    border: 1px solid rgba(107,114,128,0.25);
}}
.badge-negative {{
    background: rgba(220,38,38,0.1);
    color: {C_NEG};
    border: 1px solid rgba(220,38,38,0.25);
}}

/* ── Big number ── */
.big-num {{
    font-size: 2.8rem;
    font-weight: 700;
    color: inherit;
    line-height: 1;
    margin: 0.5rem 0 0.15rem;
    letter-spacing: -0.03em;
}}
.big-label {{
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {C_NEU};
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.5rem;
    border-bottom: 1px solid rgba(128,128,128,0.2);
    padding-bottom: 0;
}}
.stTabs [data-baseweb="tab"] {{
    color: {C_NEU};
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0.6rem 1.2rem;
    border: none;
    border-radius: 0;
}}
.stTabs [aria-selected="true"] {{
    color: inherit !important;
    border-bottom: 2px solid {ACCENT} !important;
}}

/* ── Button ── */
.stButton > button {{
    background: {ACCENT};
    color: #ffffff;
    font-weight: 600;
    font-size: 0.85rem;
    border: none;
    border-radius: 6px;
    padding: 0.55rem 1.5rem;
    width: 100%;
    transition: filter 0.15s;
}}
.stButton > button:hover {{
    filter: brightness(1.15);
}}

/* ── Download button ── */
.stDownloadButton > button {{
    background: transparent;
    color: {ACCENT};
    font-weight: 600;
    font-size: 0.82rem;
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 6px;
    width: 100%;
}}
.stDownloadButton > button:hover {{
    border-color: {ACCENT};
    background: rgba(6,182,212,0.06);
}}

/* ── Divider ── */
hr {{
    border-color: rgba(128,128,128,0.2) !important;
}}

/* ── Status ── */
.s-note {{
    font-size: 0.78rem;
    color: {C_NEU};
    margin-top: 0.3rem;
}}
.s-hi {{ color: {C_POS}; font-weight: 600; }}
.s-md {{ color: #ca8a04; font-weight: 600; }}
.s-lo {{ color: {C_NEG}; font-weight: 600; }}

/* ── Stat grid (sidebar) ── */
.sgrid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem;
}}
.scell {{
    background: rgba(128,128,128,0.06);
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 6px;
    padding: 0.65rem 0.8rem;
    text-align: center;
}}
.scell .sv {{
    font-size: 1.2rem;
    font-weight: 700;
    color: inherit;
}}
.scell .sl {{
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {C_NEU};
    margin-top: 0.1rem;
}}

/* ── Status dot ── */
.status-dot {{
    display: inline-block;
    width: 7px;
    height: 7px;
    background: {C_POS};
    border-radius: 50%;
    margin-right: 0.35rem;
    vertical-align: middle;
}}

/* ── Table ── */
.stDataFrame {{
    border-radius: 8px;
    overflow: hidden;
}}
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
    "Positive": {"badge": "badge-positive", "color": C_POS, "accent": "rgba(5,150,105,0.08)"},
    "Neutral":  {"badge": "badge-neutral",  "color": C_NEU, "accent": "rgba(107,114,128,0.06)"},
    "Negative": {"badge": "badge-negative", "color": C_NEG, "accent": "rgba(220,38,38,0.06)"},
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
        st.markdown(
            f'<span class="status-dot"></span>'
            f'<span style="color:{C_POS}; font-weight:600; font-size:0.85rem;">Active</span>',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown("**Models Trained**")
    st.markdown("- Naive Bayes  \n- Logistic Regression  \n- SVM (LinearSVC)")
    st.divider()
    st.markdown(
        """
        <div class="sgrid">
            <div class="scell"><div class="sv">10k</div><div class="sl">Reviews</div></div>
            <div class="scell"><div class="sv">0.82</div><div class="sl">Best F1</div></div>
            <div class="scell"><div class="sv">10k</div><div class="sl">Features</div></div>
            <div class="scell"><div class="sv">3</div><div class="sl">Classes</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">NLP / Machine Learning / Yelp Dataset</div>
    <h1>Restaurant Review<br>Sentiment Analysis</h1>
    <p>Classify any review as Positive, Neutral, or Negative using Logistic Regression + TF-IDF</p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_overview = st.tabs(["Single Review", "Batch Analysis", "Model Overview"])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Single Review
# ══════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='sh'>Input</div>", unsafe_allow_html=True)
    st.markdown("<div class='sdesc'>Paste or type a restaurant review</div>", unsafe_allow_html=True)

    examples = {
        "Select an example": "",
        "Positive review": "The food was absolutely amazing! The pasta was perfectly cooked and the service was incredibly friendly. Highly recommend!",
        "Neutral review": "The place was okay. Food came out in reasonable time and tasted fine. Nothing special but nothing bad either.",
        "Negative review": "Terrible experience. Waited over an hour for our food and when it finally arrived, it was cold and tasteless. Never coming back.",
    }
    choice = st.selectbox("Example", list(examples.keys()), label_visibility="collapsed")
    prefill = examples[choice]

    review_input = st.text_area(
        "Review text",
        value=prefill,
        height=120,
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
                st.markdown(
                    f"""
                    <div class="card" style="border-color:{cfg['color']}40;">
                        <div class="big-label">Predicted Sentiment</div>
                        <div style="margin:0.5rem 0;">
                            <span class="badge {cfg['badge']}">{pred}</span>
                        </div>
                        <div class="big-num">{conf:.0%}</div>
                        <div class="big-label">Confidence</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if conf >= 0.75:
                    note_cls, note = "s-hi", "High confidence"
                elif conf >= 0.50:
                    note_cls, note = "s-md", "Moderate confidence"
                else:
                    note_cls, note = "s-lo", "Low confidence"
                st.markdown(
                    f'<p class="s-note"><span class="{note_cls}">{note}</span></p>',
                    unsafe_allow_html=True,
                )

            with col_r:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='sh'>Probability Distribution</div>", unsafe_allow_html=True)

                bar_vals   = [prob_d.get(c, 0) for c in ordered]
                bar_colors = [SENTIMENT[c]["color"] for c in ordered]

                fig = go.Figure(go.Bar(
                    x=bar_vals,
                    y=ordered,
                    orientation="h",
                    marker=dict(
                        color=bar_colors,
                        opacity=0.85,
                        line=dict(width=0),
                    ),
                    text=[f"{v:.1%}" for v in bar_vals],
                    textposition="outside",
                    textfont=dict(color=C_NEU, size=12, family="Inter"),
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        range=[0, 1.2],
                        showgrid=True,
                        gridcolor="rgba(128,128,128,0.12)",
                        zeroline=False,
                        tickformat=".0%",
                        color=C_NEU,
                        tickfont=dict(family="Inter", size=11),
                    ),
                    yaxis=dict(
                        showgrid=False,
                        color=C_NEU,
                        tickfont=dict(family="Inter", size=13, color=C_NEU),
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
    st.markdown("<div class='sh'>Input</div>", unsafe_allow_html=True)
    st.markdown("<div class='sdesc'>Enter multiple reviews, one per line</div>", unsafe_allow_html=True)

    batch_input = st.text_area(
        "batch",
        height=180,
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
                    marker=dict(colors=pie_colors, line=dict(width=2, color="rgba(128,128,128,0.15)")),
                    textinfo="label+percent",
                    textfont=dict(color=C_NEU, size=12, family="Inter"),
                    insidetextfont=dict(color=C_NEU),
                ))
                pie_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=240,
                    annotations=[dict(
                        text=f"<b>{len(lines)}</b><br><span style='font-size:11px'>reviews</span>",
                        x=0.5, y=0.5, font_size=18, font_color=C_NEU,
                        showarrow=False,
                    )],
                )
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='sh'>Distribution</div>", unsafe_allow_html=True)
                st.plotly_chart(pie_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_tbl:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='sh'>Results</div>", unsafe_allow_html=True)

                def style_row(val):
                    bg   = {"Positive": "rgba(5,150,105,0.08)", "Neutral": "rgba(107,114,128,0.06)", "Negative": "rgba(220,38,38,0.06)"}
                    text = {"Positive": C_POS, "Neutral": C_NEU, "Negative": C_NEG}
                    return f"background-color:{bg.get(val, 'transparent')}; color:{text.get(val, 'inherit')}; font-weight:600;"

                styled = df.style.map(style_row, subset=["Sentiment"])
                st.dataframe(styled, use_container_width=True, hide_index=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, "results.csv", "text/csv", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Model Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("<div class='sh'>Pipeline Outputs</div>", unsafe_allow_html=True)
    st.markdown("<div class='sdesc'>Charts generated from training on 10,000 Yelp reviews</div>", unsafe_allow_html=True)

    chart_files = [
        ("sentiment_distribution.png",                "Sentiment Distribution"),
        ("model_comparison.png",                       "Model Comparison"),
        ("wordclouds.png",                             "Word Clouds"),
        ("per_class_f1.png",                           "Per-Class F1 Score"),
        ("confusion_matrix_logistic_regression.png",  "CM - Logistic Regression"),
        ("confusion_matrix_naive_bayes.png",           "CM - Naive Bayes"),
        ("confusion_matrix_svm.png",                   "CM - SVM"),
    ]

    available = [(f, t) for f, t in chart_files if os.path.exists(os.path.join(OUT_DIR, f))]

    if not available:
        st.info("No charts found. Run `py main.py` first to generate them.")
    else:
        if len(available) >= 2:
            c1, c2 = st.columns(2, gap="medium")
            for col, (fname, title) in zip([c1, c2], available[:2]):
                with col:
                    st.markdown(f"<div class='card'><div class='sh'>{title}</div>", unsafe_allow_html=True)
                    st.image(os.path.join(OUT_DIR, fname), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            remaining = available[2:]
        else:
            remaining = available

        for fname, title in remaining:
            st.markdown(f"<div class='card'><div class='sh'>{title}</div>", unsafe_allow_html=True)
            st.image(os.path.join(OUT_DIR, fname), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; padding:2rem 0 1rem; color:{C_NEU}; font-size:0.72rem;
            letter-spacing:0.08em; text-transform:uppercase; border-top:1px solid rgba(128,128,128,0.2); margin-top:1.5rem;">
    Sentiment Analysis / Logistic Regression + TF-IDF / Yelp Restaurant Reviews
</div>
""", unsafe_allow_html=True)
