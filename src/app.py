"""app.py — TAIRA Streamlit Web Application

TAIRA (Traffic Accident Intelligent Risk Advisor)
CEN 352 Term Project

Main Streamlit application providing an interactive web interface for:

1. Home Page
   - System overview and introduction
   - PEAS framework explanation
   - Quick start guide

2. Interactive Lab (Risk Assessment Tool)
   - Real-time crash scenario evaluation
   - Risk level prediction (LOW/MEDIUM/HIGH/CRITICAL)
   - Injury severity prediction
   - Human-readable explanations

3. Crash Patterns & Story (Data Analysis Dashboard)
   - Historical crash statistics
   - Injury pattern analysis
   - Model performance metrics
   - Comparative model evaluation

4. About TAIRA
   - System architecture
   - AI techniques explanation
   - Ethical considerations
   - Project credits

Technology Stack:
- Streamlit for web UI
- SVM (Support Vector Machine) for ML predictions
- Rule-based expert system for risk assessment
- Matplotlib/Seaborn for visualizations

Run with:
    streamlit run src/app.py
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from agent import TrafficAccidentAgent
from config import (
    CRASH_TYPES,
    DATA_PATH,
    LIGHTING_CONDITIONS,
    ROAD_SURFACE,
    TRAFFIC_CONTROLS,
    WEATHER_CONDITIONS,
)
from data_loader import TARGET_COLUMN, clean_data, load_data
from logic_core import injury_damage_stats_for_scenario
from model_engine import (
    ModelMetrics,
    train_model,
    compare_models,
    save_model,
    load_model,
    model_exists,
)
from logger import app_logger


st.set_page_config(page_title="TAIRA", layout="wide")
sns.set_theme(style="whitegrid")


def set_page_style() -> None:
    """Inject global CSS for a clean, professional dashboard layout."""
    st.markdown(
        """
<style>
  :root {
    --taira-bg: #020617;
    --taira-card: rgba(15, 23, 42, 0.94);
    --taira-card-border: rgba(148, 163, 184, 0.45);
    --taira-text: rgba(248, 250, 252, 0.96);
    --taira-muted: rgba(148, 163, 184, 0.90);
    --taira-accent: #7dd3fc;
    --taira-accent-2: #a78bfa;
    --taira-success: #22c55e;
    --taira-warning: #fbbf24;
    --taira-danger: #fb7185;
  }

  /* Background + base typography */
  .stApp {
    background:
      radial-gradient(1100px 700px at 0% 0%, rgba(125,211,252,0.18), transparent 55%),
      radial-gradient(1100px 700px at 100% 0%, rgba(167,139,250,0.20), transparent 55%),
      linear-gradient(180deg, #020617, #020617);
    color: #ffffff;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Segoe UI", sans-serif;
    font-size: 17px;
  }
  
  /* Header/Toolbar styling */
  header[data-testid="stHeader"] {
    background: linear-gradient(180deg, rgba(2,6,23,0.98) 0%, rgba(15,23,42,0.95) 100%) !important;
    border-bottom: 2px solid rgba(125,211,252,0.3) !important;
    backdrop-filter: blur(10px) !important;
  }
  
  /* Toolbar buttons */
  header[data-testid="stHeader"] button,
  header[data-testid="stHeader"] a {
    color: #ffffff !important;
  }
  
  header[data-testid="stHeader"] svg {
    fill: #7dd3fc !important;
  }
  
  /* Hamburger menu icon */
  button[kind="header"] {
    color: #7dd3fc !important;
  }
  
  .stMarkdown, .stText, .stCaption {
    color: #ffffff !important;
    font-size: 17px;
  }
  .stMarkdown p, .stText p {
    color: rgba(248, 250, 252, 0.95) !important;
    line-height: 1.7;
  }
  h1 {
    font-size: 52px !important;
    font-weight: 900 !important;
    letter-spacing: -0.02em !important;
    color: #ffffff !important;
  }
  h2 {
    font-size: 38px !important;
    font-weight: 800 !important;
    color: #ffffff !important;
  }
  h3 {
    font-size: 30px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15,23,42,0.98) 0%, rgba(15,23,42,0.95) 100%);
    border-right: 2px solid rgba(125,211,252,0.4);
    box-shadow: 4px 0 25px rgba(0,0,0,0.6);
  }
  
  section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
  }
  
  section[data-testid="stSidebar"] * {
    color: #ffffff !important;
  }
  
  /* Sidebar header */
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-weight: 800 !important;
  }
  
  /* Sidebar radio buttons */
  section[data-testid="stSidebar"] .row-widget.stRadio > div {
    gap: 14px;
  }
  
  /* Hide radio button circles completely */
  section[data-testid="stSidebar"] input[type="radio"] {
    display: none !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
  }
  
  section[data-testid="stSidebar"] .row-widget.stRadio > div > label > div:first-child {
    display: none !important;
  }
  
  section[data-testid="stSidebar"] .row-widget.stRadio > div > label > div[data-testid="stMarkdownContainer"] {
    margin-left: 0 !important;
  }
  
  section[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {
    display: none !important;
  }
  
  section[data-testid="stSidebar"] div[role="radio"] {
    display: none !important;
  }
  
  section[data-testid="stSidebar"] .row-widget.stRadio > div > label {
    background: linear-gradient(135deg, rgba(15,23,42,0.8), rgba(30,41,59,0.6));
    border: 2px solid rgba(148,163,184,0.4);
    border-radius: 14px;
    padding: 16px 20px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-size: 17px;
    font-weight: 700;
    color: #ffffff !important;
  }
  
  section[data-testid="stSidebar"] .row-widget.stRadio > div > label:hover {
    background: linear-gradient(135deg, rgba(125,211,252,0.2), rgba(125,211,252,0.1));
    border-color: rgba(125,211,252,0.8);
    color: #7dd3fc !important;
    transform: translateX(6px);
    box-shadow: 0 4px 15px rgba(125,211,252,0.2);
  }
  
  section[data-testid="stSidebar"] .row-widget.stRadio > div > label[data-baseweb="radio"] > div:first-child {
    background-color: rgba(125,211,252,0.3);
    border-color: #7dd3fc;
  }
  
  section[data-testid="stSidebar"] input[type="radio"]:checked ~ label,
  section[data-testid="stSidebar"] label:has(input[type="radio"]:checked) {
    background: linear-gradient(135deg, rgba(125,211,252,0.3), rgba(125,211,252,0.2)) !important;
    border: 2px solid #7dd3fc !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    box-shadow: 0 4px 20px rgba(125,211,252,0.3);
  }

  /* Cards */
  .taira-card {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.96), rgba(15, 23, 42, 0.92));
    border: 1px solid transparent;
    background-clip: padding-box;
    position: relative;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 20px 50px rgba(0,0,0,0.6), inset 0 1px 0 rgba(148,163,184,0.1);
    transition: all 0.3s ease;
  }
  
  .taira-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 20px;
    padding: 1px;
    background: linear-gradient(135deg, rgba(125,211,252,0.4), rgba(167,139,250,0.4));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
  }
  
  .taira-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 25px 60px rgba(0,0,0,0.7), inset 0 1px 0 rgba(148,163,184,0.15);
  }
  
  .taira-card h1, .taira-card h2, .taira-card h3 {
    margin: 0 0 12px 0;
    background: linear-gradient(135deg, #7dd3fc, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  /* Hero badge */
  .taira-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 999px;
    font-size: 12px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 700;
    background: linear-gradient(135deg, rgba(125,211,252,0.2), rgba(167,139,250,0.2));
    border: 2px solid rgba(125,211,252,0.4);
    color: #7dd3fc;
    box-shadow: 0 4px 15px rgba(125,211,252,0.2);
    animation: pulse 3s ease-in-out infinite;
  }
  
  @keyframes pulse {
    0%, 100% {
      box-shadow: 0 4px 15px rgba(125,211,252,0.2);
    }
    50% {
      box-shadow: 0 4px 25px rgba(125,211,252,0.4);
    }
  }

  .taira-hero-title {
    font-size: 56px;
    font-weight: 900;
    letter-spacing: -0.04em;
    margin: 10px 0 4px 0;
    color: #ffffff !important;
  }

  .taira-hero-subtitle {
    font-size: 24px;
    font-weight: 600;
    color: #ffffff !important;
    margin-bottom: 8px;
  }

  .taira-hero-text {
    font-size: 17px;
    color: rgba(248, 250, 252, 0.95) !important;
    max-width: 540px;
    line-height: 1.7;
  }

  /* Separator */
  .taira-sep {
    margin: 36px 0 28px 0;
  }
  .taira-sep-title {
    font-size: 26px;
    font-weight: 800;
    color: #ffffff !important;
    margin-bottom: 12px;
    letter-spacing: -0.02em;
    text-shadow: 0 2px 10px rgba(125,211,252,0.3);
  }
  .taira-sep-line {
    height: 3px;
    width: 100%;
    background: linear-gradient(90deg, #7dd3fc 0%, #a78bfa 50%, transparent 100%);
    border-radius: 999px;
    box-shadow: 0 2px 15px rgba(125,211,252,0.4);
  }

  /* KPI */
  .taira-kpi {
    background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,41,59,0.95));
    border: 2px solid transparent;
    background-clip: padding-box;
    position: relative;
    border-radius: 18px;
    padding: 20px;
    transition: all 0.3s ease;
  }
  
  .taira-kpi::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 18px;
    padding: 2px;
    background: linear-gradient(135deg, rgba(125,211,252,0.5), rgba(167,139,250,0.5));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
  }
  
  .taira-kpi:hover {
    transform: scale(1.03);
    box-shadow: 0 10px 30px rgba(125,211,252,0.2);
  }
  
  .taira-kpi-label {
    font-size: 14px;
    color: rgba(148,163,184,0.90);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
  }
  .taira-kpi-value {
    font-size: 32px;
    font-weight: 900;
    margin-top: 8px;
    background: linear-gradient(135deg, #7dd3fc, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  /* Main-page nav buttons */
  .taira-nav-container .stButton > button {
    border-radius: 14px;
    padding: 12px 18px;
    font-weight: 700;
    font-size: 28px;
    width: 100%;
    border: 2px solid #7dd3fc;
    background: linear-gradient(135deg, rgba(125,211,252,0.25), rgba(125,211,252,0.15));
    color: #7dd3fc !important;
    transition: all 0.3s ease;
  }
  .taira-nav-container .stButton > button:hover {
    border-color: #7dd3fc;
    background: linear-gradient(135deg, rgba(125,211,252,0.35), rgba(125,211,252,0.25));
    color: #7dd3fc !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(125,211,252,0.3);
  }
  .taira-nav-container .stButton > button:active,
  .taira-nav-container .stButton > button:focus {
    border-color: #7dd3fc !important;
    background: linear-gradient(135deg, rgba(125,211,252,0.35), rgba(125,211,252,0.25)) !important;
    color: #7dd3fc !important;
    box-shadow: 0 0 0 3px rgba(125,211,252,0.2) !important;
  }

  /* Make Streamlit widgets feel tighter */
  div[data-testid="stForm"] {
    background: rgba(15,23,42,0.93);
    border: 1px solid rgba(148,163,184,0.45);
    border-radius: 16px;
    padding: 14px;
  }
  
  /* All buttons - always blue */
  .stButton > button {
    border-radius: 12px;
    padding: 12px 18px;
    font-weight: 700;
    font-size: 16px;
    border: 2px solid #7dd3fc;
    background: linear-gradient(135deg, rgba(125,211,252,0.25), rgba(125,211,252,0.15));
    color: #7dd3fc !important;
    transition: all 0.3s ease;
  }
  .stButton > button:hover {
    border-color: #7dd3fc;
    background: linear-gradient(135deg, rgba(125,211,252,0.35), rgba(125,211,252,0.25));
    color: #7dd3fc !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(125,211,252,0.3);
  }
  .stButton > button:active,
  .stButton > button:focus {
    border-color: #7dd3fc !important;
    background: linear-gradient(135deg, rgba(125,211,252,0.35), rgba(125,211,252,0.25)) !important;
    color: #7dd3fc !important;
    box-shadow: 0 0 0 3px rgba(125,211,252,0.2) !important;
  }
  
  /* Form submit buttons - force blue style */
  button[kind="formSubmit"],
  button[type="submit"],
  .stForm button[kind="primary"],
  div[data-testid="stForm"] button {
    border-radius: 12px !important;
    padding: 12px 18px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border: 2px solid #7dd3fc !important;
    background: linear-gradient(135deg, rgba(125,211,252,0.25), rgba(125,211,252,0.15)) !important;
    color: #7dd3fc !important;
    transition: all 0.3s ease !important;
  }
  
  button[kind="formSubmit"]:hover,
  button[type="submit"]:hover,
  .stForm button[kind="primary"]:hover,
  div[data-testid="stForm"] button:hover {
    border-color: #7dd3fc !important;
    background: linear-gradient(135deg, rgba(125,211,252,0.35), rgba(125,211,252,0.25)) !important;
    color: #7dd3fc !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(125,211,252,0.3) !important;
  }
  
  button[kind="formSubmit"]:active,
  button[kind="formSubmit"]:focus,
  button[type="submit"]:active,
  button[type="submit"]:focus,
  .stForm button[kind="primary"]:active,
  .stForm button[kind="primary"]:focus,
  div[data-testid="stForm"] button:active,
  div[data-testid="stForm"] button:focus {
    border-color: #7dd3fc !important;
    background: linear-gradient(135deg, rgba(125,211,252,0.35), rgba(125,211,252,0.25)) !important;
    color: #7dd3fc !important;
    box-shadow: 0 0 0 3px rgba(125,211,252,0.2) !important;
  }
  
  .stSelectbox div[data-baseweb="select"] {
    border-radius: 12px;
  }
  
  /* Increase label and text sizes */
  label, .stSelectbox label, .stRadio label {
    font-size: 17px !important;
    font-weight: 600 !important;
    color: #ffffff !important;
  }
  
  p {
    font-size: 17px !important;
    color: rgba(248, 250, 252, 0.95) !important;
  }
  
  /* Form text */
  div[data-testid="stForm"] label,
  div[data-testid="stForm"] p,
  div[data-testid="stForm"] span {
    color: #ffffff !important;
  }
  
  /* Caption text */
  .stCaption {
    color: rgba(248, 250, 252, 0.85) !important;
  }
</style>
""",
        unsafe_allow_html=True,
    )


def _col_any_case(df: pd.DataFrame, col_name: str) -> Optional[str]:
    col_map = {str(c).lower(): str(c) for c in df.columns}
    return col_map.get(col_name.lower())


def _sorted_unique_any_case(df: pd.DataFrame, col_name: str) -> Iterable[str]:
    actual = _col_any_case(df, col_name)
    if actual is None:
        return []
    values = df[actual].dropna().astype(str).tolist()
    values = [v.strip() for v in values if v.strip()]
    return sorted(set(values))


@st.cache_data(show_spinner=False)
def get_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_raw, df_model).

    - df_raw: raw CSV read (original column casing, used for historical lookups)
    - df_model: cleaned/training-ready dataset (uppercased columns per data_loader)
    """
    df_raw = pd.read_csv(DATA_PATH)
    df_model = clean_data(load_data(DATA_PATH))
    return df_raw, df_model


@st.cache_resource(show_spinner=True)
def get_agent_and_metrics() -> Tuple[TrafficAccidentAgent, Dict[str, float]]:
    """Train (cached) or load trained model and return the hybrid agent + evaluation metrics.
    
    Uses SVM (Support Vector Machine) - Current winner for accident prediction.
    Loads from disk if available, otherwise trains and saves.
    """
    _df_raw, df_model = get_datasets()
    
    # Try to load existing model first
    if model_exists():
        try:
            app_logger.info("Loading existing model from disk...")
            pipeline, metrics = load_model()
            agent = TrafficAccidentAgent(pipeline)
            
            metrics_dict = {
                "accuracy": float(metrics.accuracy),
                "f1_weighted": float(metrics.f1_weighted),
                "precision_weighted": float(metrics.precision_weighted),
                "recall_weighted": float(metrics.recall_weighted),
            }
            
            app_logger.info(f"Model loaded successfully: Accuracy={metrics.accuracy:.4f}")
            return agent, metrics_dict
        except Exception as e:
            app_logger.warning(f"Failed to load model, will retrain: {e}")
    
    # Train new model if load failed or no model exists
    app_logger.info("Training new model...")
    pipeline, metrics = train_model(df_model, model_type="svm")
    
    # Save the trained model
    try:
        save_model(pipeline, metrics)
        app_logger.info("Model saved successfully")
    except Exception as e:
        app_logger.warning(f"Failed to save model: {e}")
    
    agent = TrafficAccidentAgent(pipeline)
    
    metrics_dict = {
        "accuracy": float(metrics.accuracy),
        "f1_weighted": float(metrics.f1_weighted),
        "precision_weighted": float(metrics.precision_weighted),
        "recall_weighted": float(metrics.recall_weighted),
    }
    
    return agent, metrics_dict


def _taira_card(title: str, body_md: str) -> None:
    st.markdown(
        f"""
<div class="taira-card">
  <h3>{title}</h3>
  <div style="color: rgba(226,232,240,0.86); font-size:14px; line-height: 1.6;">
    {body_md}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _separator(title: str) -> None:
    st.markdown(
        f"""
<div class="taira-sep">
  <div class="taira-sep-title">{title}</div>
  <div class="taira-sep-line"></div>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------- HOME PAGE (updated, with clickable nav buttons) ----------

def page_home(df_raw: pd.DataFrame, metrics: Dict[str, float]) -> None:
    left, right = st.columns([1.7, 1.3], gap="large")

    with left:
        st.markdown('<span class="taira-badge">CEN 352 · Hybrid Intelligent Agent</span>', unsafe_allow_html=True)
        st.markdown(
            """
<div style="margin-top: 10px;">
  <div class="taira-hero-title">TAIRA</div>
  <div class="taira-hero-subtitle">
    Traffic Accident Intelligent Risk Advisor
  </div>
  <div class="taira-hero-text">
    TAIRA is a hybrid intelligent agent that transforms crash data into actionable
    risk assessments. It combines a trained <b>Support Vector Machine (SVM)</b>
    with a <b>rule-based expert system</b> to deliver accurate predictions with transparent explanations.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
<div class="taira-hero-text" style="margin-top: 10px;">
  <ul>
    <li><b>Assess crash scenarios</b> with predicted injury severity and dispatcher-ready risk levels (LOW / MEDIUM / HIGH / CRITICAL).</li>
    <li><b>Analyze historical patterns</b> to understand which conditions lead to severe injuries.</li>
    <li><b>Support emergency response</b> decisions with data-driven insights and human-readable explanations.</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        accuracy = metrics.get("accuracy", 0.0)
        f1 = metrics.get("f1_weighted", 0.0)
        precision = metrics.get("precision_weighted", 0.0)
        recall = metrics.get("recall_weighted", 0.0)
        total = len(df_raw)

        st.markdown(
            f"""
<div class="taira-card" style="min-height: 320px;">
  <div style="font-weight: 800; font-size: 16px; margin-bottom: 4px;">Model Snapshot</div>
  <div style="margin-top: 8px;">
    <div class="taira-kpi">
      <div class="taira-kpi-label">Test Accuracy</div>
      <div class="taira-kpi-value">{accuracy:.3f}</div>
    </div>
  </div>
  <div style="margin-top: 10px;">
    <div class="taira-kpi">
      <div class="taira-kpi-label">F1 (Weighted)</div>
      <div class="taira-kpi-value">{f1:.3f}</div>
    </div>
  </div>
  <div style="margin-top: 10px; display: flex; gap: 10px;">
    <div class="taira-kpi" style="flex: 1;">
      <div class="taira-kpi-label">Precision</div>
      <div class="taira-kpi-value" style="font-size: 24px;">{precision:.3f}</div>
    </div>
    <div class="taira-kpi" style="flex: 1;">
      <div class="taira-kpi-label">Recall</div>
      <div class="taira-kpi-value" style="font-size: 24px;">{recall:.3f}</div>
    </div>
  </div>
  <div style="margin-top: 12px; font-size: 13px; color: rgba(148,163,184,0.9);">
    Dataset size: <b>{total:,}</b> crash records
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    _separator("Jump into TAIRA")

    # Clickable navigation buttons
    st.markdown('<div class="taira-nav-container">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        if st.button("Interactive Lab", use_container_width=True):
            st.session_state["page"] = "Interactive Lab"
    with c2:
        if st.button("Crash Patterns & Story", use_container_width=True):
            st.session_state["page"] = "Crash Patterns & Story"
    with c3:
        if st.button("About TAIRA", use_container_width=True):
            st.session_state["page"] = "About TAIRA"
    st.markdown("</div>", unsafe_allow_html=True)

    _separator("What can you do here?")

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        _taira_card(
            "Interactive Lab",
            "<b>Run prediction tools</b> using dropdown-only scenario inputs and get a hybrid ML + rules explanation.",
        )
    with c2:
        _taira_card(
            "Crash Patterns & Story",
            "Explore a <b>dashboard</b> of time-based and environmental crash patterns with auto-generated insights.",
        )
    with c3:
        _taira_card(
            "Hybrid AI Model",
            "TAIRA blends <b>SVM (statistical learning)</b> with a <b>rule-based</b> reasoning layer for alerts.",
        )
    with c4:
        _taira_card(
            "Ethics & Impact",
            "Understand bias, limitations, and <b>automation bias</b> risks when using predictive systems for safety decisions.",
        )

    st.caption(f"Dataset loaded: {len(df_raw):,} crash records")


def page_crash_scenario_assessment(agent: TrafficAccidentAgent, df_model: pd.DataFrame, metrics: Dict[str, float]) -> None:
    """TAB A — Crash Scenario Assessment (Hybrid Agent).

    Note: Per spec, this section uses ONLY categorical dropdowns (no numeric inputs).
    """
    st.markdown("### Crash Scenario Assessment (Hybrid Agent)")

    # TOP-RIGHT model status card
    a, b = st.columns([3.2, 1.5], gap="large")
    with b:
        st.markdown(
            """
<div class="taira-card">
  <div style="font-weight: 800; font-size: 16px; margin-bottom: 8px;">Model Status</div>
  <div style="color: rgba(226,232,240,0.78);">Global test-set metrics:</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.write(f"Accuracy: **{metrics.get('accuracy', 0.0):.4f}**")
        st.write(f"Precision (weighted): **{metrics.get('precision_weighted', 0.0):.4f}**")
        st.write(f"Recall (weighted): **{metrics.get('recall_weighted', 0.0):.4f}**")
        st.write(f"F1-Score (weighted): **{metrics.get('f1_weighted', 0.0):.4f}**")

    with a:
        # Per requirements: use config lists for a subset, and df_model uniques for others
        roadway_surface_cond_options = list(_sorted_unique_any_case(df_model, "roadway_surface_cond"))
        if not roadway_surface_cond_options:
            roadway_surface_cond_options = list(ROAD_SURFACE)
        trafficway_type_options = list(_sorted_unique_any_case(df_model, "trafficway_type"))
        alignment_options = list(_sorted_unique_any_case(df_model, "alignment"))
        road_defect_options = list(_sorted_unique_any_case(df_model, "road_defect"))
        intersection_related_options = list(_sorted_unique_any_case(df_model, "intersection_related_i"))

        with st.form("pre_crash_form"):
            c1, c2 = st.columns(2, gap="large")

            with c1:
                traffic_control_device = st.selectbox("traffic_control_device", TRAFFIC_CONTROLS)
                weather_condition = st.selectbox("weather_condition", WEATHER_CONDITIONS)
                lighting_condition = st.selectbox("lighting_condition", LIGHTING_CONDITIONS)
                roadway_surface_cond = st.selectbox("roadway_surface_cond", roadway_surface_cond_options)

            with c2:
                trafficway_type = st.selectbox("trafficway_type", trafficway_type_options)
                alignment = st.selectbox("alignment", alignment_options)
                road_defect = st.selectbox("road_defect", road_defect_options)
                # Friendlier label; keep internal key unchanged for model compatibility
                intersection_related_i = st.selectbox("intersection_related", intersection_related_options)

            submitted = st.form_submit_button("Assess Crash Scenario")

        if not submitted:
            return

        features = {
            "traffic_control_device": traffic_control_device,
            "weather_condition": weather_condition,
            "lighting_condition": lighting_condition,
            "roadway_surface_cond": roadway_surface_cond,
            "trafficway_type": trafficway_type,
            "alignment": alignment,
            "road_defect": road_defect,
            "intersection_related_i": intersection_related_i,
            "prim_contributory_cause": "UNABLE TO DETERMINE",  # default value
            "crash_type": "NO INDICATION",  # default value
        }

        result = agent.predict(features)
        predicted = str(result.get("predicted_injury", "UNKNOWN"))
        risk = str(result.get("risk_level", "LOW")).upper()

        st.markdown("#### Results")
        st.write(f"Predicted Most Severe Injury: **{predicted}**")
        if risk == "CRITICAL":
            st.error("🚨 Risk Level: CRITICAL 🚨")
        elif risk == "HIGH":
            st.error("Risk Level: HIGH")
        elif risk == "MEDIUM":
            st.warning("Risk Level: MEDIUM")
        else:
            st.info("Risk Level: LOW")
        st.write(result.get("explanation", ""))


def page_post_crash_patterns(df_raw: pd.DataFrame, df_model: pd.DataFrame) -> None:
    """TAB B — Post-Crash Analysis."""
    st.markdown("### Post-Crash Analysis")
    st.caption("Analyze historical crash data by primary cause and crash type.")

    cause_options = list(_sorted_unique_any_case(df_model, "prim_contributory_cause"))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        prim_contributory_cause = st.selectbox("Primary Contributory Cause", cause_options)
    with c2:
        crash_type = st.selectbox("Crash Type", CRASH_TYPES)

    if not st.button("Analyze This Scenario"):
        return

    # Get detailed stats from raw data
    stats = injury_damage_stats_for_scenario(df_raw, prim_contributory_cause, crash_type)
    if stats.get("count", 0) == 0:
        st.warning("No matching records found for that scenario.")
        return

    # Also show injury severity breakdown from model data
    filtered_df = df_model[
        (df_model["PRIM_CONTRIBUTORY_CAUSE"].str.upper() == prim_contributory_cause.upper()) &
        (df_model["FIRST_CRASH_TYPE"].str.upper() == crash_type.upper())
    ]
    
    st.markdown(f"<div style='font-size: 20px; color: #ffffff; font-weight: 700; margin-bottom: 16px;'>Matching records: {int(stats['count']):,}</div>", unsafe_allow_html=True)
    
    # Show injury severity distribution with visual bars
    if len(filtered_df) > 0:
        st.markdown("---")
        st.markdown("<div style='font-size: 21px; font-weight: 800; color: #ffffff; margin-bottom: 12px;'>Injury Severity Distribution</div>", unsafe_allow_html=True)
        
        injury_counts = filtered_df["MOST_SEVERE_INJURY"].value_counts()
        injury_pct = (injury_counts / len(filtered_df) * 100).round(1)
        
        # Define injury severity order and colors
        injury_order = [
            "FATAL",
            "INCAPACITATING INJURY", 
            "NONINCAPACITATING INJURY",
            "REPORTED, NOT EVIDENT",
            "NO INDICATION OF INJURY"
        ]
        
        injury_colors = {
            "FATAL": "#dc2626",
            "INCAPACITATING INJURY": "#f97316",
            "NONINCAPACITATING INJURY": "#eab308",
            "REPORTED, NOT EVIDENT": "#22c55e",
            "NO INDICATION OF INJURY": "#10b981"
        }
        
        for injury_type in injury_order:
            if injury_type in injury_counts.index:
                count = injury_counts[injury_type]
                pct = injury_pct[injury_type]
                color = injury_colors.get(injury_type, "#7dd3fc")
                
                # Create visual bar
                bar_width = int(pct * 3)  # Scale for visual effect
                st.markdown(
                    f"""
<div style="margin-bottom: 12px;">
  <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
    <span style="color: rgba(226,232,240,0.95); font-weight: 600;">{injury_type}</span>
    <span style="color: {color}; font-weight: 700; font-size: 16px;">{pct}%</span>
  </div>
  <div style="background: rgba(15,23,42,0.8); border-radius: 8px; height: 32px; overflow: hidden;">
    <div style="background: {color}; width: {pct}%; height: 100%; border-radius: 8px; 
         display: flex; align-items: center; padding-left: 12px; padding-right: 12px; color: white; font-weight: 600; font-size: 13px; min-width: fit-content;">
      {count} crashes
    </div>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )
        
        # Calculate and show risk level
        severe_pct = injury_pct.get("FATAL", 0) + injury_pct.get("INCAPACITATING INJURY", 0)
        if severe_pct >= 15:
            risk_label = "HIGH RISK"
            risk_color = "#dc2626"
        elif severe_pct >= 8:
            risk_label = "MODERATE RISK"
            risk_color = "#f97316"
        else:
            risk_label = "LOW RISK"
            risk_color = "#22c55e"
        
        st.markdown(
            f"""
<div style="margin-top: 16px; padding: 12px; background: rgba(15,23,42,0.6); border-left: 4px solid {risk_color}; border-radius: 8px;">
  <div style="color: {risk_color}; font-weight: 800; font-size: 15px;">{risk_label}</div>
  <div style="color: rgba(226,232,240,0.85); margin-top: 4px;">Severe injury rate (FATAL + INCAPACITATING): <b>{severe_pct:.1f}%</b></div>
</div>
""",
            unsafe_allow_html=True,
        )
    
    # Show common conditions for this scenario
    st.markdown("---")
    st.markdown("<div style='font-size: 21px; font-weight: 800; color: #ffffff; margin-bottom: 12px;'>Common Conditions for This Scenario</div>", unsafe_allow_html=True)
    
    if len(filtered_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Most common weather
            if "WEATHER_CONDITION" in filtered_df.columns:
                most_common_weather = filtered_df["WEATHER_CONDITION"].mode()
                if len(most_common_weather) > 0:
                    st.markdown(f"<div style='font-size: 19px; margin-bottom: 8px;'>🌤️ Most common weather: <span style='color: #ffffff; font-weight: 700;'>{most_common_weather[0]}</span></div>", unsafe_allow_html=True)
            
            # Most common lighting
            if "LIGHTING_CONDITION" in filtered_df.columns:
                most_common_lighting = filtered_df["LIGHTING_CONDITION"].mode()
                if len(most_common_lighting) > 0:
                    st.markdown(f"<div style='font-size: 19px; margin-bottom: 8px;'>💡 Most common lighting: <span style='color: #ffffff; font-weight: 700;'>{most_common_lighting[0]}</span></div>", unsafe_allow_html=True)
        
        with col2:
            # Most common road surface
            if "ROADWAY_SURFACE_COND" in filtered_df.columns:
                most_common_surface = filtered_df["ROADWAY_SURFACE_COND"].mode()
                if len(most_common_surface) > 0:
                    st.markdown(f"<div style='font-size: 19px; margin-bottom: 8px;'>🛣️ Most common surface: <span style='color: #ffffff; font-weight: 700;'>{most_common_surface[0]}</span></div>", unsafe_allow_html=True)
            
            # Most common traffic control
            if "TRAFFIC_CONTROL_DEVICE" in filtered_df.columns:
                most_common_control = filtered_df["TRAFFIC_CONTROL_DEVICE"].mode()
                if len(most_common_control) > 0:
                    st.markdown(f"<div style='font-size: 19px; margin-bottom: 8px;'>🚦 Most common control: <span style='color: #ffffff; font-weight: 700;'>{most_common_control[0]}</span></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<div style='font-size: 21px; font-weight: 800; color: #ffffff; margin-bottom: 8px;'>Injury Impact Summary</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 16px; color: rgba(226,232,240,0.75); margin-bottom: 12px;'>Total injuries across all matching crashes in this scenario</div>", unsafe_allow_html=True)
    
    # Display in a cleaner format with totals
    total_crashes = int(stats.get('count', 1))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fatal = int(stats.get('avg_injuries_fatal', 0.0) * total_crashes)
        st.markdown(
            f"""
<div style="text-align: center; padding: 12px;">
  <div style="color: #ffffff; font-size: 14px; font-weight: 600; margin-bottom: 6px;">Fatal</div>
  <div style="color: #ffffff; font-size: 32px; font-weight: 800;">{fatal}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        
        incapacitating = int(stats.get('avg_incapacitating', 0.0) * total_crashes)
        st.markdown(
            f"""
<div style="text-align: center; padding: 12px;">
  <div style="color: #ffffff; font-size: 14px; font-weight: 600; margin-bottom: 6px;">Incapacitating</div>
  <div style="color: #ffffff; font-size: 32px; font-weight: 800;">{incapacitating}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    
    with col2:
        non_incap = int(stats.get('avg_non_incapacitating', 0.0) * total_crashes)
        st.markdown(
            f"""
<div style="text-align: center; padding: 12px;">
  <div style="color: #ffffff; font-size: 14px; font-weight: 600; margin-bottom: 6px;">Non-Incapacitating</div>
  <div style="color: #ffffff; font-size: 32px; font-weight: 800;">{non_incap}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        
        reported = int(stats.get('avg_reported_not_evident', 0.0) * total_crashes)
        st.markdown(
            f"""
<div style="text-align: center; padding: 12px;">
  <div style="color: #ffffff; font-size: 14px; font-weight: 600; margin-bottom: 6px;">Reported (Not Evident)</div>
  <div style="color: #ffffff; font-size: 32px; font-weight: 800;">{reported}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    
    with col3:
        no_indication = int(stats.get('avg_no_indication', 0.0) * total_crashes)
        st.markdown(
            f"""
<div style="text-align: center; padding: 12px;">
  <div style="color: #ffffff; font-size: 14px; font-weight: 600; margin-bottom: 6px;">No Indication</div>
  <div style="color: #ffffff; font-size: 32px; font-weight: 800;">{no_indication}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        
        total_injuries = fatal + incapacitating + non_incap + reported
        st.markdown(
            f"""
<div style="text-align: center; padding: 12px;">
  <div style="color: #ffffff; font-size: 14px; font-weight: 600; margin-bottom: 6px;">Total Injured</div>
  <div style="color: #ffffff; font-size: 32px; font-weight: 800;">{total_injuries}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    
    if "most_common_damage" in stats:
        st.markdown("---")
        damage_level = str(stats['most_common_damage'])
        st.markdown(
            f"""
<div style="margin-top: 16px;">
  <span style="color: rgba(226,232,240,0.9); font-size: 20px; font-weight: 600;">💰 Most common damage level: </span>
  <span style="color: #ffffff; font-size: 20px; font-weight: 700;">{damage_level}</span>
</div>
""",
            unsafe_allow_html=True,
        )


def page_interactive(agent: TrafficAccidentAgent, df_raw: pd.DataFrame, df_model: pd.DataFrame, metrics: Dict[str, float]) -> None:
    st.header("Interactive Lab")
    st.markdown(
        """
<div class="taira-card">
  <div style="font-weight: 900; font-size: 16px;">Two Tools, One Workspace</div>
  <div style="margin-top: 8px; color: rgba(226,232,240,0.80); line-height: 1.55;">
    Use TAIRA to <b>assess crash scenarios</b> using the hybrid agent,
    and analyze <b>post-crash injury patterns</b> from historical data.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    pre_tab, post_tab = st.tabs([
        "Crash Scenario Assessment",
        "Post-Crash Analysis",
    ])

    with pre_tab:
        page_crash_scenario_assessment(agent, df_model, metrics)

    with post_tab:
        page_post_crash_patterns(df_raw, df_model)


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def _compute_season_risk(df_raw: pd.DataFrame) -> str:
    month_col = _col_any_case(df_raw, "crash_month")
    injury_col = _col_any_case(df_raw, "most_severe_injury")
    if month_col is None:
        return "Unknown"

    tmp = df_raw.copy()
    tmp[month_col] = pd.to_numeric(tmp[month_col], errors="coerce")
    tmp = tmp.dropna(subset=[month_col])
    tmp["_season"] = tmp[month_col].astype(int).apply(_season_from_month)

    if injury_col is None:
        return str(tmp["_season"].value_counts().idxmax())

    sev = tmp[injury_col].astype(str).str.upper()
    severe_mask = sev.str.contains("FATAL") | sev.str.contains("INCAPACITATING")
    tmp["_severe"] = severe_mask.astype(int)
    season_rate = tmp.groupby("_season")["_severe"].mean().sort_values(ascending=False)
    return str(season_rate.index[0]) if not season_rate.empty else "Unknown"


def _kpi(label: str, value: str) -> None:
    st.markdown(
        f"""
<div class="taira-kpi">
  <div class="taira-kpi-label">{label}</div>
  <div class="taira-kpi-value">{value}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _bar_count_chart(df: pd.DataFrame, col_name: str, title: str, *, top_n: Optional[int] = None, hue: Optional[str] = None) -> None:
    actual = _col_any_case(df, col_name)
    if actual is None:
        st.info(f"Column not available: {col_name}")
        return

    colors_palette = ['#1e2a38', '#3e5968', '#00b8b8', '#006f4f', '#2b3d4f', '#4f6f7e', '#00a5a5', '#005b46', '#284f63', '#5f797b']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#020617')
    ax.set_facecolor('#0f172a')
    
    if hue is not None:
        hue_actual = _col_any_case(df, hue)
        if hue_actual is not None:
            sns.countplot(data=df, x=actual, hue=hue_actual, palette=colors_palette, ax=ax)
            ax.legend(title=hue, facecolor='#0f172a', edgecolor='#94a3b8', labelcolor='#f8fafc')
        else:
            sns.countplot(data=df, x=actual, palette=colors_palette, ax=ax)
    else:
        series = df[actual].dropna().astype(str)
        counts = series.value_counts()
        if top_n is not None:
            counts = counts.head(top_n)
        ax.bar(counts.index.astype(str), counts.values, color=colors_palette[:len(counts)])
    
    ax.set_title(title, color='#f8fafc', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(col_name.replace('_', ' ').title(), color='#94a3b8', fontsize=11)
    ax.set_ylabel('Count', color='#94a3b8', fontsize=11)
    ax.tick_params(axis='x', labelrotation=45, colors='#94a3b8')
    ax.tick_params(axis='y', colors='#94a3b8')
    ax.grid(True, alpha=0.2, color='#94a3b8', linestyle='--')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#94a3b8')
        spine.set_alpha(0.3)
    
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _hist_chart(df: pd.DataFrame, col_name: str, title: str) -> None:
    """Create a histogram with KDE for numerical columns."""
    actual = _col_any_case(df, col_name)
    if actual is None:
        st.info(f"Column not available: {col_name}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#020617')
    ax.set_facecolor('#0f172a')
    
    sns.histplot(df[actual], kde=True, color='#00b8b8', ax=ax, bins=30)
    
    ax.set_title(title, color='#f8fafc', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(col_name.replace('_', ' ').title(), color='#94a3b8', fontsize=11)
    ax.set_ylabel('Count', color='#94a3b8', fontsize=11)
    ax.tick_params(axis='x', colors='#94a3b8')
    ax.tick_params(axis='y', colors='#94a3b8')
    ax.grid(True, alpha=0.2, color='#94a3b8', linestyle='--')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#94a3b8')
        spine.set_alpha(0.3)
    
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def page_crash_patterns(df_raw: pd.DataFrame) -> None:
    st.header("Crash Patterns & Story")

    st.markdown(
        """
<div class="taira-card">
  <div style="font-weight: 900; font-size: 16px;">Urban Analytics Dashboard</div>
  <div style="margin-top: 8px; color: rgba(226,232,240,0.80); line-height: 1.55;">
    This dashboard visualizes crash patterns across time and environment conditions.
    It helps contextualize what the model learns by showing the underlying data distribution.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    crash_type_col = _col_any_case(df_raw, "crash_type") or _col_any_case(df_raw, "first_crash_type")
    surface_col = _col_any_case(df_raw, "roadway_surface_cond")

    total_crashes = f"{len(df_raw):,}"
    most_common_crash_type = "Unknown"
    if crash_type_col is not None:
        most_common_crash_type = str(df_raw[crash_type_col].dropna().astype(str).value_counts().idxmax())
    highest_risk_season = _compute_season_risk(df_raw)
    most_common_surface = "Unknown"
    if surface_col is not None:
        most_common_surface = str(df_raw[surface_col].dropna().astype(str).value_counts().idxmax())

    k1, k2, k3, k4 = st.columns(4, gap="large")
    with k1:
        _kpi("Total Crashes", total_crashes)
    with k2:
        _kpi("Most Common Crash Type", most_common_crash_type)
    with k3:
        _kpi("Highest-Risk Season", highest_risk_season)
    with k4:
        _kpi("Most Common Road Surface", most_common_surface)

    _separator("Time-Based Patterns")
    t1, t2, t3 = st.columns(3, gap="large")
    with t1:
        st.markdown('<div class="taira-card">', unsafe_allow_html=True)
        _bar_count_chart(df_raw, "crash_month", "Distribution of Crashes by Month")
        st.markdown("</div>", unsafe_allow_html=True)
    with t2:
        st.markdown('<div class="taira-card">', unsafe_allow_html=True)
        _bar_count_chart(df_raw, "crash_hour", "Distribution of Crashes by Hour")
        st.markdown("</div>", unsafe_allow_html=True)
    with t3:
        st.markdown('<div class="taira-card">', unsafe_allow_html=True)
        _bar_count_chart(df_raw, "crash_day_of_week", "Distribution of Crashes by Day of the Week")
        st.markdown("</div>", unsafe_allow_html=True)

    _separator("Environmental Conditions")
    e1, e2 = st.columns(2, gap="large")
    with e1:
        st.markdown('<div class="taira-card">', unsafe_allow_html=True)
        _bar_count_chart(df_raw, "weather_condition", "Distribution of Weather Conditions")
        st.markdown("</div>", unsafe_allow_html=True)
    with e2:
        st.markdown('<div class="taira-card">', unsafe_allow_html=True)
        _bar_count_chart(df_raw, "lighting_condition", "Distribution of Lighting Conditions")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="taira-card">', unsafe_allow_html=True)
    _bar_count_chart(df_raw, "weather_condition", "Weather Condition vs Most Severe Injury", hue="most_severe_injury")
    st.markdown("</div>", unsafe_allow_html=True)
    
    e3, e4 = st.columns(2, gap="large")
    with e3:
        st.markdown('<div class="taira-card">', unsafe_allow_html=True)
        _bar_count_chart(df_raw, "roadway_surface_cond", "Road Surface Condition Distribution")
        st.markdown("</div>", unsafe_allow_html=True)
    with e4:
        st.markdown('<div class="taira-card">', unsafe_allow_html=True)
        _bar_count_chart(df_raw, "most_severe_injury", "Distribution of Most Severe Injury Types")
        st.markdown("</div>", unsafe_allow_html=True)

    _separator("Crash Types & Causes")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="taira-card">', unsafe_allow_html=True)
        _bar_count_chart(df_raw, "crash_type", "Distribution of Crash Types")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="taira-card">', unsafe_allow_html=True)
        _bar_count_chart(df_raw, "prim_contributory_cause", "Primary Contributory Cause (Top 15)", top_n=15)
        st.markdown("</div>", unsafe_allow_html=True)
    
    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown('<div class="taira-card">', unsafe_allow_html=True)
        _bar_count_chart(df_raw, "traffic_control_device", "Traffic Control Device Distribution")
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="taira-card">', unsafe_allow_html=True)
        _hist_chart(df_raw, "injuries_total", "Distribution of Total Injuries")
        st.markdown("</div>", unsafe_allow_html=True)

    # Optional charts
    optional_left, optional_right = st.columns(2, gap="large")
    with optional_left:
        st.markdown('<div class="taira-card">', unsafe_allow_html=True)
        _bar_count_chart(df_raw, "road_defect", "Road Defects Distribution", top_n=15)
        st.markdown("</div>", unsafe_allow_html=True)
    with optional_right:
        st.markdown('<div class="taira-card">', unsafe_allow_html=True)
        _bar_count_chart(df_raw, "trafficway_type", "Trafficway Type Distribution", top_n=15)
        st.markdown("</div>", unsafe_allow_html=True)

    _separator("Story")
    month_col = _col_any_case(df_raw, "crash_month")
    hour_col = _col_any_case(df_raw, "crash_hour")
    weather_col = _col_any_case(df_raw, "weather_condition")
    light_col = _col_any_case(df_raw, "lighting_condition")

    highest_month = "Unknown"
    if month_col is not None:
        highest_month = str(pd.to_numeric(df_raw[month_col], errors="coerce").dropna().astype(int).value_counts().idxmax())
    peak_hour = "Unknown"
    if hour_col is not None:
        peak_hour = str(pd.to_numeric(df_raw[hour_col], errors="coerce").dropna().astype(int).value_counts().idxmax())
    common_weather = "Unknown"
    if weather_col is not None:
        common_weather = str(df_raw[weather_col].dropna().astype(str).value_counts().idxmax())
    common_light = "Unknown"
    if light_col is not None:
        common_light = str(df_raw[light_col].dropna().astype(str).value_counts().idxmax())

    most_dangerous_light = "Unknown"
    injury_col = _col_any_case(df_raw, "most_severe_injury")
    if light_col is not None and injury_col is not None:
        tmp = df_raw[[light_col, injury_col]].dropna()
        sev = tmp[injury_col].astype(str).str.upper()
        tmp["_severe"] = (sev.str.contains("FATAL") | sev.str.contains("INCAPACITATING")).astype(int)
        rates = tmp.groupby(light_col)["_severe"].mean().sort_values(ascending=False)
        if not rates.empty:
            most_dangerous_light = str(rates.index[0])

    st.markdown(
        f"""
<div class="taira-card">
  <div style="font-weight: 900; font-size: 16px;">Automatically Generated Observations</div>
  <div style="margin-top: 10px; color: rgba(226,232,240,0.90); font-size:14px; line-height: 1.65;">
    <ul>
      <li><b>Highest crash month:</b> {highest_month}</li>
      <li><b>Peak hourly period:</b> {peak_hour}:00</li>
      <li><b>Most frequent weather condition:</b> {common_weather}</li>
      <li><b>Most frequent lighting condition:</b> {common_light}</li>
      <li><b>Most dangerous lighting (severe-rate):</b> {most_dangerous_light}</li>
    </ul>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    fun_facts = [
        "Even 'CLEAR' weather can dominate simply because it happens most often.",
        "Peak crash hours often track commute patterns more than road design.",
        "A small severe-injury rate can still matter at city scale.",
        "Rule-based alerts are useful because they stay readable under uncertainty.",
    ]
    st.info(random.choice(fun_facts))


def page_about(metrics: Dict[str, float]) -> None:
    st.header("About TAIRA")

    _taira_card(
        "Project Overview",
        "TAIRA is a CEN 352 hybrid intelligent agent that predicts crash injury severity using SVM and applies safety rules for risk assessment.",
    )

    _taira_card(
        "PEAS",
        """
<b>Performance measure:</b> Accuracy and weighted F1-score (plus usefulness of risk alerts).<br/>
<b>Environment:</b> traffic accident records with road, weather, lighting, and crash characteristics.<br/>
<b>Actuators:</b> severity prediction, risk level, and warning/explanation messages in the UI.<br/>
<b>Sensors:</b> dataset features and user-selected scenario inputs.
""",
    )

    _taira_card(
        "Hybrid AI Architecture",
        """
<b>Layer 1:</b> SVM (Support Vector Machine) predicts <i>Most Severe Injury</i> from categorical conditions.<br/>
<b>Layer 2:</b> Rule-based reasoning converts conditions + prediction into LOW/MEDIUM/HIGH risk and a short explanation.
""",
    )

    _taira_card(
        "Model Evaluation",
        f"Current cached model metrics: <b>Accuracy</b> = {metrics.get('accuracy', 0.0):.4f}, <b>F1 (weighted)</b> = {metrics.get('f1_weighted', 0.0):.4f}.",
    )

    _taira_card(
        "Ethical Reflection",
        """
Predictive safety tools can help awareness and planning, but can also introduce <b>automation bias</b> if users trust outputs blindly.
The dataset may reflect reporting and location biases; performance may differ across under-represented conditions.
TAIRA is a learning tool and should not be treated as a real-time safety guarantee.
""",
    )


def main() -> None:
    set_page_style()

    # Session state for central navigation
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    df_raw, df_model = get_datasets()
    agent, metrics = get_agent_and_metrics()

    # Sidebar navigation
    st.sidebar.markdown("<div class='taira-badge'>TAIRA Navigation</div>", unsafe_allow_html=True)
    sidebar_choice = st.sidebar.radio(
        "Go to",
        ["Home", "Interactive Lab", "Crash Patterns & Story", "About TAIRA"],
        index=["Home", "Interactive Lab", "Crash Patterns & Story", "About TAIRA"].index(st.session_state["page"]),
    )
    # Keep session_state["page"] synced with sidebar choice
    st.session_state["page"] = sidebar_choice

    page = st.session_state["page"]

    if page == "Home":
        page_home(df_raw, metrics)
    elif page == "Interactive Lab":
        page_interactive(agent, df_raw, df_model, metrics)
    elif page == "Crash Patterns & Story":
        page_crash_patterns(df_raw)
    else:
        page_about(metrics)


if __name__ == "__main__":
    main()
