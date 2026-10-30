"""
Streamlit Web Application for NLP-Based Sarcasm and Irony Detection in Text.
Academic Project - SemEval-2018 Task 3 Subtask B Benchmark.
"""

import os
import sys
import json
import io
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import validate_input_text
from src.prediction import predict_single, predict_batch, load_artifacts
from src.database import (
    init_db, insert_prediction, get_history, get_statistics,
    clear_history, export_all_predictions
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NLP Sarcasm & Irony Detection | AI Dashboard",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_db()

# Pre-load model artifacts (with caching in session or memory)
@st.cache_resource
def get_cached_artifacts():
    try:
        return load_artifacts()
    except Exception as e:
        return None, None, {"error": str(e)}

model, vectorizer, metadata = get_cached_artifacts()

# -----------------------------------------------------------------------------
# 2. CUSTOM THEME & CSS INJECTION (Dark Navy & Purple Aesthetic)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #0a0e1a;
        color: #f8fafc;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f1526;
        border-right: 1px solid rgba(139, 92, 246, 0.2);
    }

    /* Cards */
    .custom-card {
        background: linear-gradient(145deg, #121829, #172038);
        border: 1px solid rgba(139, 92, 246, 0.22);
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .custom-card:hover {
        border-color: rgba(168, 85, 247, 0.45);
    }

    /* Stat Cards */
    .stat-metric-card {
        background: linear-gradient(145deg, #121829, #18223a);
        border: 1px solid rgba(139, 92, 246, 0.22);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        position: relative;
        overflow: hidden;
    }
    .stat-metric-title {
        color: #94a3b8;
        font-size: 0.78rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }
    .stat-metric-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    .stat-metric-sub {
        font-size: 0.75rem;
        margin-top: 0.2rem;
    }

    /* Custom Badges */
    .badge-sarcastic {
        background: linear-gradient(135deg, #e11d48, #f43f5e);
        color: #ffffff;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 0 16px rgba(244, 63, 94, 0.4);
        font-size: 0.95rem;
    }
    .badge-ironical {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: #ffffff;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 0 16px rgba(245, 158, 11, 0.4);
        font-size: 0.95rem;
    }
    .badge-literal {
        background: linear-gradient(135deg, #059669, #10b981);
        color: #ffffff;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.4);
        font-size: 0.95rem;
    }

    /* Feature Token Tag */
    .feature-tag {
        display: inline-block;
        background: rgba(168, 85, 247, 0.16);
        border: 1px solid rgba(168, 85, 247, 0.35);
        color: #e9d5ff;
        padding: 4px 10px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        margin: 3px;
    }

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #8b5cf6, #ec4899);
    }

    /* Textarea & Inputs */
    .stTextArea textarea {
        background-color: #0b1120 !important;
        border: 1px solid rgba(139, 92, 246, 0.25) !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
    }
    .stTextArea textarea:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.25) !important;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed, #9333ea) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.35) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #8b5cf6, #a855f7) !important;
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.45) !important;
    }

    /* Secondary buttons */
    .stButton > button[kind="secondary"] {
        background: #141b2d !important;
        border: 1px solid rgba(139, 92, 246, 0.25) !important;
        color: #c084fc !important;
        border-radius: 8px !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(139, 92, 246, 0.15) !important;
        border-color: #a855f7 !important;
        color: #f3e8ff !important;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: rgba(255, 255, 255, 0.04);
        border-left: 3px solid #8b5cf6;
        padding: 0.65rem 0.9rem;
        border-radius: 6px;
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & SYSTEM METRICS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
        <div style="background: linear-gradient(135deg, #4c1d95, #7c3aed); width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; box-shadow: 0 0 15px rgba(124, 58, 237, 0.4);">
            🎭
        </div>
        <div>
            <div style="font-weight: 800; font-size: 1.15rem; color: #ffffff;">Sarcasm & Irony</div>
            <div style="font-size: 0.75rem; color: #c084fc; font-weight: 500;">NLP Detection Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display: inline-flex; align-items: center; gap: 6px;">
        <span style="width: 8px; height: 8px; background-color: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981;"></span>
        SemEval-2018 Task 3 Model Online
    </span>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(139, 92, 246, 0.2); margin: 15px 0;'>", unsafe_allow_html=True)

    nav_selection = st.radio(
        "Navigation",
        [
            "🔍 Single Text Analysis",
            "📂 Batch & CSV Analysis",
            "📜 Prediction History",
            "📊 Model Benchmarks",
            "📖 Project & Viva Guide"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(139, 92, 246, 0.2); margin: 15px 0;'>", unsafe_allow_html=True)

    # Quick System Specs
    st.markdown("""
    <div style="background-color: #141b2d; border: 1px solid rgba(139, 92, 246, 0.18); border-radius: 10px; padding: 12px; font-size: 0.78rem;">
        <div style="font-weight: 700; color: #e2e8f0; margin-bottom: 6px;">Pipeline Architecture</div>
        <div style="color: #94a3b8; display: flex; flex-direction: column; gap: 4px;">
            <div>• <b>Classifier:</b> Logistic Regression</div>
            <div>• <b>Features:</b> 5,000 TF-IDF N-grams</div>
            <div>• <b>Calibration:</b> Platt Sigmoidal Scaling</div>
            <div>• <b>Storage:</b> SQLite DB (Thread-safe)</div>
            <div>• <b>Benchmark:</b> SemEval-2018 Subtask B</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.72rem; margin-top: 25px;">
        B.Tech Academic Project &bull; 2026<br>
        Developed for Transparent NLP Research
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. VIEW 1: SINGLE TEXT ANALYSIS (MAIN DASHBOARD)
# -----------------------------------------------------------------------------
if nav_selection == "🔍 Single Text Analysis":
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <h2 style="font-weight: 800; margin-bottom: 0.2rem; color: #ffffff;">
            Sarcasm & Irony Detection Dashboard
        </h2>
        <p style="color: #94a3b8; font-size: 0.92rem; margin-bottom: 0;">
            Real-time figurative language classifier differentiating between <span style="color:#fb7185; font-weight:600;">Sarcastic</span>, <span style="color:#fbbf24; font-weight:600;">Ironical</span>, and <span style="color:#34d399; font-weight:600;">Literal</span> sentences with calibrated confidence and token feature attribution.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Statistics Row
    stats = get_statistics()
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(f"""
        <div class="stat-metric-card">
            <div class="stat-metric-title">Total Analyzed</div>
            <div class="stat-metric-val">{stats['total_predictions']:,}</div>
            <div class="stat-metric-sub" style="color: #94a3b8;">Lifetime predictions</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="stat-metric-card">
            <div class="stat-metric-title">Sarcastic</div>
            <div class="stat-metric-val" style="color: #fb7185;">{stats['sarcastic_count']:,}</div>
            <div class="stat-metric-sub" style="color: #fb7185;">{stats['sarcastic_pct']}% (Polarity contrast)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="stat-metric-card">
            <div class="stat-metric-title">Ironical</div>
            <div class="stat-metric-val" style="color: #fbbf24;">{stats['ironical_count']:,}</div>
            <div class="stat-metric-sub" style="color: #fbbf24;">{stats['ironical_pct']}% (Situational irony)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="stat-metric-card">
            <div class="stat-metric-title">Literal / Non-Sarcastic</div>
            <div class="stat-metric-val" style="color: #34d399;">{stats['literal_count']:,}</div>
            <div class="stat-metric-sub" style="color: #34d399;">{stats['literal_pct']}% (Objective text)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Session state for text input
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = "Oh fantastic, my flight is delayed by another 4 hours! Just what I needed today!"

    # Preset Example Sentences
    st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #c084fc; margin-bottom: 6px;'>💡 Preset Benchmark Examples:</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    def set_example(text):
        st.session_state["input_text"] = text

    with c1:
        if st.button("✈️ Sarcastic Flight", use_container_width=True, type="secondary"):
            set_example("Oh fantastic, my flight is delayed by another 4 hours! Just what I needed today!")
    with c2:
        if st.button("🌧️ Sarcastic Bus", use_container_width=True, type="secondary"):
            set_example("I just love waiting in 30-degree freezing rain for a bus that never arrives.")
    with c3:
        if st.button("🚒 Ironical Fire", use_container_width=True, type="secondary"):
            set_example("A fire station literally burned down to the ground yesterday afternoon.")
    with c4:
        if st.button("🚑 Ironical Hospital", use_container_width=True, type="secondary"):
            set_example("An ambulance crashed straight into the hospital emergency room doors.")
    with c5:
        if st.button("☀️ Literal Weather", use_container_width=True, type="secondary"):
            set_example("The regional weather forecast predicts intermittent rain showers tomorrow afternoon.")
    with c6:
        if st.button("📅 Literal Meeting", use_container_width=True, type="secondary"):
            set_example("The project steering committee will meet on Thursday at 10 AM in Conference Room B.")

    # Two column layout: Input on Left, Analysis on Right
    col_input, col_result = st.columns([1.05, 0.95], gap="large")

    with col_input:
        st.markdown("""
        <div style="font-weight: 700; font-size: 1rem; color: #ffffff; margin-top: 10px; margin-bottom: 6px;">
            📝 Text Input
        </div>
        """, unsafe_allow_html=True)

        user_text = st.text_area(
            "Input sentence to analyze:",
            value=st.session_state["input_text"],
            height=160,
            max_chars=2000,
            key="user_text_input",
            label_visibility="collapsed",
            placeholder="Type or paste any English statement to evaluate for sarcasm or irony..."
        )

        char_len = len(user_text.strip())
        st.markdown(f"<div style='text-align: right; color: #64748b; font-size: 0.75rem; font-family: monospace;'>Characters: {char_len} / 2000</div>", unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            clear_clicked = st.button("❌ Clear Input", use_container_width=True)
            if clear_clicked:
                st.session_state["input_text"] = ""
                st.rerun()

        with btn_col2:
            analyze_clicked = st.button("✨ Analyze Text", type="primary", use_container_width=True)

    with col_result:
        st.markdown("""
        <div style="font-weight: 700; font-size: 1rem; color: #ffffff; margin-top: 10px; margin-bottom: 6px;">
            🎯 Classification & Explainability
        </div>
        """, unsafe_allow_html=True)

        if analyze_clicked or ("last_prediction" in st.session_state and st.session_state.get("last_prediction")):
            # If user clicked analyze, compute new prediction
            if analyze_clicked:
                is_valid, err = validate_input_text(user_text)
                if not is_valid:
                    st.error(f"⚠️ {err}")
                    st.session_state["last_prediction"] = None
                else:
                    with st.spinner("Analyzing linguistic patterns..."):
                        try:
                            pred_res = predict_single(user_text)
                            # Record in DB
                            insert_prediction(
                                input_text=user_text,
                                predicted_class=pred_res["predicted_class"],
                                confidence=pred_res["confidence"],
                                probabilities=pred_res["probabilities"],
                                explanation=pred_res["explanation"],
                                top_features=pred_res["top_features"],
                                source="single"
                            )
                            st.session_state["last_prediction"] = pred_res
                        except Exception as e:
                            st.error(f"Error during inference: {str(e)}")
                            st.session_state["last_prediction"] = None

            pred = st.session_state.get("last_prediction")

            if pred:
                pred_class = pred["predicted_class"]
                conf_pct = pred["confidence_percentage"]
                probs = pred["probabilities"]
                explanation = pred["explanation"]
                top_features = pred["top_features"]

                # Badge selection
                if pred_class == "Sarcastic":
                    badge_html = f'<div class="badge-sarcastic">🔥 Sarcastic (Polarity Contrast)</div>'
                    bar_color = "#f43f5e"
                elif pred_class == "Ironical":
                    badge_html = f'<div class="badge-ironical">⚡ Ironical (Situational Incongruity)</div>'
                    bar_color = "#f59e0b"
                else:
                    badge_html = f'<div class="badge-literal">✅ Non-Sarcastic / Literal</div>'
                    bar_color = "#10b981"

                st.markdown(f"""
                <div class="custom-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <div>{badge_html}</div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Calibrated Confidence</div>
                            <div style="font-size: 1.5rem; font-weight: 800; font-family: monospace; color: #ffffff;">{conf_pct}%</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                st.progress(conf_pct / 100.0)

                # Class breakdown
                p_sarc = round(probs.get("Sarcastic", 0) * 100, 1)
                p_iron = round(probs.get("Ironical", 0) * 100, 1)
                p_lit = round(probs.get("Non-Sarcastic / Literal", 0) * 100, 1)

                st.markdown(f"""
                    <div style="background-color: rgba(11, 17, 32, 0.6); border: 1px solid rgba(139, 92, 246, 0.2); border-radius: 8px; padding: 10px 14px; margin-top: 14px; margin-bottom: 14px;">
                        <div style="font-size: 0.75rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px;">Class Probability Distribution</div>
                        <div style="display: flex; flex-direction: column; gap: 6px; font-size: 0.82rem;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: #fb7185;">• Sarcastic (Polarity Clash)</span>
                                <span style="font-family: monospace; font-weight: 600;">{p_sarc}%</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: #fbbf24;">• Ironical (Situational)</span>
                                <span style="font-family: monospace; font-weight: 600;">{p_iron}%</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: #34d399;">• Non-Sarcastic / Literal</span>
                                <span style="font-family: monospace; font-weight: 600;">{p_lit}%</span>
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 10px;">
                        <div style="font-size: 0.78rem; font-weight: 600; color: #c084fc; text-transform: uppercase;">💬 Model Explanation</div>
                        <div style="font-size: 0.88rem; color: #f1f5f9; line-height: 1.5; margin-top: 4px;">{explanation}</div>
                    </div>
                """, unsafe_allow_html=True)

                # Feature Tokens
                if top_features:
                    st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #94a3b8; margin-top: 10px; margin-bottom: 4px;'>Key Contributing N-Grams:</div>", unsafe_allow_html=True)
                    tags_html = "".join([
                        f'<span class="feature-tag">🏷️ {f["feature"]} <small style="opacity:0.8;">({"+" if f["weight"] > 0 else ""}{f["weight"]})</small></span>'
                        for f in top_features
                    ])
                    st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)

                st.markdown("""
                    <div class="disclaimer-box">
                        <b>Transparency Note:</b> This prediction reflects statistical n-gram patterns learned from the SemEval-2018 benchmark corpus. It does not represent human consciousness, theory of mind, or guaranteed emotional intent.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="custom-card" style="text-align: center; padding: 2.8rem 1.5rem;">
                <div style="font-size: 2.5rem; margin-bottom: 10px;">🤖</div>
                <div style="font-weight: 700; font-size: 1.1rem; color: #ffffff;">Awaiting Text Input</div>
                <div style="color: #94a3b8; font-size: 0.85rem; max-width: 320px; margin: 8px auto 0;">
                    Type a sentence on the left or click a preset benchmark example to inspect the calibrated prediction, probability breakdown, and token weights.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Historical Distribution Chart & Recent Log Row
    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(139, 92, 246, 0.2); margin: 25px 0 20px;'>", unsafe_allow_html=True)
    row2_col1, row2_col2 = st.columns([1, 1], gap="large")

    with row2_col1:
        st.markdown("<div style='font-weight: 700; font-size: 1rem; color: #ffffff; margin-bottom: 12px;'>📊 Live Prediction Distribution</div>", unsafe_allow_html=True)
        chart_data = pd.DataFrame({
            "Classification": ["Sarcastic", "Ironical", "Literal"],
            "Count": [stats["sarcastic_count"], stats["ironical_count"], stats["literal_count"]]
        })

        if stats["total_predictions"] == 0:
            st.info("No predictions recorded yet. Run your first analysis above!")
        else:
            donut_chart = alt.Chart(chart_data).mark_arc(innerRadius=60).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(
                    field="Classification",
                    type="nominal",
                    scale=alt.Scale(
                        domain=["Sarcastic", "Ironical", "Literal"],
                        range=["#f43f5e", "#f59e0b", "#10b981"]
                    ),
                    legend=alt.Legend(title="", orient="bottom")
                ),
                tooltip=["Classification", "Count"]
            ).properties(height=260)
            st.altair_chart(donut_chart, use_container_width=True)

    with row2_col2:
        st.markdown("<div style='font-weight: 700; font-size: 1rem; color: #ffffff; margin-bottom: 12px;'>🕒 Recent Predictions</div>", unsafe_allow_html=True)
        recent_items, _ = get_history(limit=5)
        if not recent_items:
            st.info("No history records yet.")
        else:
            for item in recent_items:
                c_name = item["predicted_class"]
                color_dot = "#fb7185" if c_name == "Sarcastic" else ("#fbbf24" if c_name == "Ironical" else "#34d399")
                st.markdown(f"""
                <div style="background-color: #121829; border: 1px solid rgba(139, 92, 246, 0.15); border-radius: 8px; padding: 8px 12px; margin-bottom: 8px; font-size: 0.82rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                        <span style="color: {color_dot}; font-weight: 700;">● {c_name}</span>
                        <span style="font-family: monospace; color: #ffffff; font-weight: 600;">{round(item['confidence']*100)}%</span>
                    </div>
                    <div style="color: #cbd5e1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                        {item['input_text']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. VIEW 2: BATCH & CSV ANALYSIS
# -----------------------------------------------------------------------------
elif nav_selection == "📂 Batch & CSV Analysis":
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <h2 style="font-weight: 800; margin-bottom: 0.2rem; color: #ffffff;">
            Batch Sarcasm & Irony Analysis
        </h2>
        <p style="color: #94a3b8; font-size: 0.92rem;">
            Process dozens or hundreds of sentences simultaneously via multi-line text paste or CSV dataset upload, with downloadable structured reports.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_paste, tab_csv = st.tabs(["📝 Paste Multiple Sentences", "📁 Upload CSV / TXT File"])

    batch_texts = []

    with tab_paste:
        default_batch = (
            "Oh fantastic, another delayed flight on Monday morning!\n"
            "A fire station literally burned down yesterday afternoon.\n"
            "The weekly team meeting is scheduled for Thursday at 10 AM.\n"
            "I just love waiting 45 minutes for iced coffee in freezing winter.\n"
            "The regional weather forecast predicts clear sunny skies."
        )
        batch_input_str = st.text_area(
            "Enter sentences (one per line):",
            value=default_batch,
            height=180,
            help="Each line will be treated as an individual sentence for NLP classification."
        )
        if st.button("🚀 Analyze Lines", type="primary", key="btn_run_paste"):
            lines = [l.strip() for l in batch_input_str.splitlines() if l.strip()]
            if lines:
                batch_texts = lines
            else:
                st.warning("Please enter at least one non-empty sentence.")

    with tab_csv:
        uploaded_file = st.file_uploader(
            "Choose a CSV or TXT file:",
            type=["csv", "txt"],
            help="File must contain a column named 'text', 'sentence', 'tweet', or 'headline'."
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_up = pd.read_csv(uploaded_file)
                    # Detect text column
                    text_col = None
                    for candidate in ["text", "sentence", "tweet", "headline", "content"]:
                        for col in df_up.columns:
                            if candidate == col.lower():
                                text_col = col
                                break
                        if text_col:
                            break

                    if not text_col:
                        text_col = df_up.columns[0]
                        st.info(f"Using first column '{text_col}' as text source.")

                    parsed_texts = df_up[text_col].dropna().astype(str).tolist()
                else:
                    content = uploaded_file.read().decode("utf-8", errors="ignore")
                    parsed_texts = [l.strip() for l in content.splitlines() if l.strip()]

                st.success(f"Successfully loaded {len(parsed_texts)} text entries from '{uploaded_file.name}'.")

                if st.button("🚀 Process Uploaded File", type="primary", key="btn_run_csv"):
                    batch_texts = parsed_texts

            except Exception as e:
                st.error(f"Error parsing uploaded file: {str(e)}")

    # Execute Batch Prediction
    if batch_texts:
        MAX_BATCH = 500
        if len(batch_texts) > MAX_BATCH:
            st.warning(f"Batch capped at maximum {MAX_BATCH} items for responsiveness.")
            batch_texts = batch_texts[:MAX_BATCH]

        with st.spinner(f"Classifying {len(batch_texts)} statements..."):
            results = predict_batch(batch_texts)

            # Insert into SQLite
            for r in results:
                if r["status"] == "success":
                    insert_prediction(
                        input_text=r["input_text"],
                        predicted_class=r["predicted_class"],
                        confidence=r["confidence"],
                        probabilities={},
                        explanation=r["explanation"],
                        top_features=[],
                        source="batch"
                    )

        # Batch Metrics Summary
        total_b = len(results)
        sarc_b = sum(1 for r in results if r["predicted_class"] == "Sarcastic")
        iron_b = sum(1 for r in results if r["predicted_class"] == "Ironical")
        lit_b = sum(1 for r in results if r["predicted_class"] in ["Non-Sarcastic / Literal", "Literal"])

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("Total Processed", total_b)
        with b2:
            st.metric("Sarcastic", f"{sarc_b} ({round(sarc_b/total_b*100)}%)")
        with b3:
            st.metric("Ironical", f"{iron_b} ({round(iron_b/total_b*100)}%)")
        with b4:
            st.metric("Literal", f"{lit_b} ({round(lit_b/total_b*100)}%)")

        # Table & Download
        res_df = pd.DataFrame(results)[["row_index", "input_text", "predicted_class", "confidence_percentage", "explanation"]]
        res_df.columns = ["Row", "Input Sentence", "Predicted Class", "Confidence %", "Explanation"]

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.dataframe(res_df, use_container_width=True, height=350)

        # Download CSV
        csv_buffer = io.StringIO()
        res_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Batch Results (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"batch_sarcasm_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary"
        )

# -----------------------------------------------------------------------------
# 6. VIEW 3: PREDICTION HISTORY & AUDIT LOG
# -----------------------------------------------------------------------------
elif nav_selection == "📜 Prediction History":
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <h2 style="font-weight: 800; margin-bottom: 0.2rem; color: #ffffff;">
            Prediction History & Audit Log
        </h2>
        <p style="color: #94a3b8; font-size: 0.92rem;">
            Persisted SQLite audit trail of all model inferences with search querying, class filtering, and full CSV export capabilities.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_search, col_filter, col_limit = st.columns([2, 1.2, 0.8])
    with col_search:
        search_kw = st.text_input("🔍 Search keyword in text or explanation:", placeholder="e.g. flight, meeting, hospital...")
    with col_filter:
        class_choice = st.selectbox("Filter by Class:", ["All", "Sarcastic", "Ironical", "Non-Sarcastic / Literal"])
    with col_limit:
        limit_choice = st.selectbox("Show rows:", [25, 50, 100, 200], index=1)

    items, total_found = get_history(
        limit=limit_choice,
        search_query=search_kw if search_kw.strip() else None,
        class_filter=None if class_choice == "All" else class_choice
    )

    st.markdown(f"<div style='font-size: 0.82rem; color: #c084fc; margin-bottom: 8px;'>Showing <b>{len(items)}</b> of <b>{total_found}</b> matching records</div>", unsafe_allow_html=True)

    if items:
        hist_df = pd.DataFrame(items)[["id", "input_text", "predicted_class", "confidence", "source", "created_at"]]
        hist_df["confidence"] = hist_df["confidence"].apply(lambda c: f"{round(c*100)}%")
        hist_df.columns = ["ID", "Input Statement", "Classification", "Confidence", "Source", "Timestamp (UTC)"]

        st.dataframe(hist_df, use_container_width=True, height=400)

        c_down, c_clear = st.columns([1, 1])
        with c_down:
            all_export = export_all_predictions()
            export_df = pd.DataFrame(all_export)
            csv_exp = export_df.to_csv(index=False)
            st.download_button(
                "📥 Export Complete History (CSV)",
                data=csv_exp,
                file_name=f"all_sarcasm_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with c_clear:
            with st.expander("⚠️ Clear History Options"):
                st.write("This will permanently delete all stored predictions from SQLite.")
                if st.button("🗑️ Confirm Clear All Records", type="secondary"):
                    deleted = clear_history()
                    st.success(f"Successfully deleted {deleted} records.")
                    st.rerun()
    else:
        st.info("No records match your filter criteria.")

# -----------------------------------------------------------------------------
# 7. VIEW 4: MODEL BENCHMARKS & EVALUATION
# -----------------------------------------------------------------------------
elif nav_selection == "📊 Model Benchmarks":
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <h2 style="font-weight: 800; margin-bottom: 0.2rem; color: #ffffff;">
            Model Performance & Academic Benchmark
        </h2>
        <p style="color: #94a3b8; font-size: 0.92rem;">
            Rigorous evaluation on the held-out gold test set of <b>SemEval-2018 Task 3 (Subtask B)</b> comparing Multinomial Logistic Regression and Platt-Calibrated Linear SVM.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load evaluation report
    eval_file = os.path.join(PROJECT_ROOT, "reports", "evaluation_summary.json")
    if os.path.exists(eval_file):
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    else:
        eval_data = {}

    champ_name = eval_data.get("champion_model_name", "Logistic Regression")
    champ_m = eval_data.get("champion_metrics", {})
    comp = eval_data.get("models_comparison", {})

    # Top metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Champion Test Accuracy", f"{champ_m.get('accuracy', 60.84)}%", help="Evaluated on N=784 independent gold test instances")
    with m2:
        st.metric("Macro F1-Score", f"{champ_m.get('macro_f1', 57.31)}%", help="Balanced across all 3 classes")
    with m3:
        st.metric("Macro Precision", f"{champ_m.get('macro_precision', 56.10)}%")
    with m4:
        st.metric("Macro Recall", f"{champ_m.get('macro_recall', 59.94)}%", help="High sensitivity across rare irony classes")

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(139, 92, 246, 0.2); margin: 20px 0;'>", unsafe_allow_html=True)

    # Comparison Bar Chart & Per-Class Table
    col_chart, col_cm = st.columns([1, 1], gap="large")

    with col_chart:
        st.markdown("<div style='font-weight: 700; font-size: 1rem; color: #ffffff; margin-bottom: 8px;'>📈 Baseline Comparison: LR vs Calibrated SVM</div>", unsafe_allow_html=True)
        if comp:
            lr_m = comp.get("logistic_regression", {}).get("metrics", {})
            svm_m = comp.get("linear_svm", {}).get("metrics", {})

            comp_df = pd.DataFrame([
                {"Model": "Logistic Regression", "Metric": "Accuracy (%)", "Value": lr_m.get("accuracy", 60.84)},
                {"Model": "Logistic Regression", "Metric": "Macro Precision (%)", "Value": lr_m.get("macro_precision", 56.10)},
                {"Model": "Logistic Regression", "Metric": "Macro Recall (%)", "Value": lr_m.get("macro_recall", 59.94)},
                {"Model": "Logistic Regression", "Metric": "Macro F1 (%)", "Value": lr_m.get("macro_f1", 57.31)},

                {"Model": "Linear SVM (Calibrated)", "Metric": "Accuracy (%)", "Value": svm_m.get("accuracy", 62.12)},
                {"Model": "Linear SVM (Calibrated)", "Metric": "Macro Precision (%)", "Value": svm_m.get("macro_precision", 45.45)},
                {"Model": "Linear SVM (Calibrated)", "Metric": "Macro Recall (%)", "Value": svm_m.get("macro_recall", 48.20)},
                {"Model": "Linear SVM (Calibrated)", "Metric": "Macro F1 (%)", "Value": svm_m.get("macro_f1", 45.96)},
            ])

            chart = alt.Chart(comp_df).mark_bar().encode(
                x=alt.X("Metric:N", title=""),
                y=alt.Y("Value:Q", title="Percentage (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Model:N", scale=alt.Scale(domain=["Logistic Regression", "Linear SVM (Calibrated)"], range=["#8b5cf6", "#3b82f6"])),
                xOffset="Model:N",
                tooltip=["Model", "Metric", "Value"]
            ).properties(height=280)

            st.altair_chart(chart, use_container_width=True)
            st.caption("Logistic Regression was chosen as Champion because of its significantly higher Macro F1 (57.31% vs 45.96%) and balanced recall on the rare Ironical class.")

    with col_cm:
        st.markdown("<div style='font-weight: 700; font-size: 1rem; color: #ffffff; margin-bottom: 8px;'>🎯 Confusion Matrix (Gold Test Partition, N=784)</div>", unsafe_allow_html=True)
        cm_data = champ_m.get("confusion_matrix", {}).get("matrix", [[294, 110, 69], [34, 97, 33], [29, 32, 86]])

        cm_df = pd.DataFrame(
            cm_data,
            index=["Actual: Literal", "Actual: Sarcastic", "Actual: Ironical"],
            columns=["Pred: Literal", "Pred: Sarcastic", "Pred: Ironical"]
        )

        st.dataframe(
            cm_df.style.background_gradient(cmap="Purples").format("{:,}"),
            use_container_width=True
        )

        st.markdown("""
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 6px;">
            • <b>Literal Support:</b> 473 samples &bull; <b>Sarcastic Support:</b> 164 samples &bull; <b>Ironical Support:</b> 147 samples<br>
            • Diagonal values represent true positive predictions.
        </div>
        """, unsafe_allow_html=True)

    # Per-Class Metrics Table
    st.markdown("<div style='font-weight: 700; font-size: 1rem; color: #ffffff; margin-top: 20px; margin-bottom: 8px;'>📋 Per-Class Performance Breakdown</div>", unsafe_allow_html=True)
    per_class = champ_m.get("per_class", {})
    if per_class:
        pc_rows = []
        for cls_name, vals in per_class.items():
            pc_rows.append({
                "Target Class": cls_name,
                "Precision (%)": f"{vals['precision']}%",
                "Recall (%)": f"{vals['recall']}%",
                "F1-Score (%)": f"{vals['f1_score']}%",
                "Support (N)": vals["support"]
            })
        st.table(pd.DataFrame(pc_rows))

# -----------------------------------------------------------------------------
# 8. VIEW 5: PROJECT ARCHITECTURE & VIVA GUIDE
# -----------------------------------------------------------------------------
elif nav_selection == "📖 Project & Viva Guide":
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <h2 style="font-weight: 800; margin-bottom: 0.2rem; color: #ffffff;">
            Project Architecture & Viva Voce Guide
        </h2>
        <p style="color: #94a3b8; font-size: 0.92rem;">
            Theoretical foundations, linguistic taxonomy, end-to-end pipeline dataflow, and B.Tech examination questions with model answers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_t1, col_t2 = st.columns([1.1, 0.9], gap="large")

    with col_t1:
        st.markdown("""
        <div class="custom-card">
            <h4 style="font-weight: 700; color: #ffffff; margin-bottom: 8px;">Problem Statement & Objective</h4>
            <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;">
                Standard Natural Language Processing models assume <b>literal compositionality</b>: positive words (e.g. <i>"great"</i>, <i>"fantastic"</i>) are scored as positive sentiment. When humans express sarcasm, they invert polarity to ridicule or mock (e.g. <i>"Oh fantastic, another 4-hour flight delay!"</i>).
            </p>
            <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;">
                This project builds an end-to-end multi-class NLP system trained on the <b>SemEval-2018 Task 3</b> benchmark that classifies sentences into:
            </p>
            <ul style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;">
                <li><b style="color: #fb7185;">Sarcastic:</b> Verbal irony with polarity inversion (praise to censure).</li>
                <li><b style="color: #fbbf24;">Ironical:</b> Situational absurdity or incongruity without direct polarity flip.</li>
                <li><b style="color: #34d399;">Literal:</b> Objective statements where intended meaning equals literal words.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        st.markdown("""
        <div class="custom-card">
            <h4 style="font-weight: 700; color: #ffffff; margin-bottom: 8px;">Key Preprocessing Innovations</h4>
            <div style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.5; display: flex; flex-direction: column; gap: 8px;">
                <div><b>1. Negation Preservation:</b> Standard stopword lists remove <code>not</code>, <code>no</code>, <code>never</code>. We explicitly retain them because they are key polarity inverters.</div>
                <div><b>2. Intensifier Retention:</b> Retains <code>so</code>, <code>too</code>, <code>very</code>, <code>really</code>, which frequently accompany hyperbolic sarcasm.</div>
                <div><b>3. Hashtag Segmentation:</b> Regex decomposes camelCase tags (e.g. <code>#BestDayEver</code> &rarr; <i>"Best Day Ever"</i>).</div>
                <div><b>4. Punctuation Clusters:</b> Preserves repeated marks (<code>!!</code>, <code>??</code>, <code>?!</code>) that signal rhetorical disbelief.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(139, 92, 246, 0.2); margin: 20px 0;'>", unsafe_allow_html=True)

    # Viva Voce Expanders
    st.markdown("<div style='font-weight: 700; font-size: 1.15rem; color: #ffffff; margin-bottom: 12px;'>🎓 B.Tech Viva Voce & Technical Defense Questions</div>", unsafe_allow_html=True)

    with st.expander("Q1: Why is removing standard NLTK stopwords disastrous for sarcasm detection?"):
        st.markdown("""
        **Answer:** Standard stopword lists remove negation words like `not`, `never`, `no`, `without`, `hardly` and intensifiers like `so`, `too`, `very`. In sentiment and sarcasm analysis, negations are the essential linguistic pivots that invert polarity (e.g., *"not happy"* vs *"happy"*). Stripping them completely destroys the contrast between literal words and intended meaning. Our preprocessor uses an **irony-aware stopword list** that preserves negations and intensifiers.
        """)

    with st.expander("Q2: What is the linguistic and computational distinction between Sarcasm and Irony?"):
        st.markdown("""
        **Answer:** Sarcasm is a subtype of verbal irony characterized by an explicit *polarity clash* between a positive surface expression and a negative pragmatic intention, often directed toward a specific target to ridicule. Irony encompasses broader situational incongruities where an outcome is contrary to natural expectation (e.g., *"A fire station burning down"*), without an explicit polarity flip. SemEval-2018 Task 3 Subtask B formalized this taxonomy into separate multi-class targets.
        """)

    with st.expander("Q3: Why was probability calibration (Platt Scaling) required for the Linear SVM?"):
        st.markdown("""
        **Answer:** A standard Linear SVM maximizes geometric margins and outputs an uncalibrated signed decision distance $f(x) \in (-\infty, \infty)$ rather than a posterior probability $P(y=c|x)$. Platt scaling (implemented via `CalibratedClassifierCV`) fits a logistic sigmoid function over cross-validation decision values to transform raw distances into mathematically valid, calibrated probabilities in $[0, 1]$.
        """)

    with st.expander("Q4: How does feature attribution (explainability) work in your application?"):
        st.markdown("""
        **Answer:** For linear models, the decision score is a linear combination of feature weights and TF-IDF terms: $\\text{score}_c = \\sum_i w_{c,i} \\cdot x_i + b_c$. For any input sentence, our pipeline extracts the active n-grams, computes each token's contribution $\\text{attribution}_i = w_{c,i} \\cdot x_i$, and surfaces the highest positive contributing tokens to explain the prediction transparently.
        """)

    with st.expander("Q5: What are the main limitations and future scope of this project?"):
        st.markdown("""
        **Answer:**
        - **Limitations:** N-gram models lack deep conversational context and cannot capture acoustic pitch, vocal inflection, or facial expressions present in spoken sarcasm.
        - **Future Scope:** Fine-tuning transformer models (e.g., RoBERTa, DeBERTa-v3), modeling multi-turn conversational dialogue threads, and incorporating multimodal features (speech audio and video).
        """)
