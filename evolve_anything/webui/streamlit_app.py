#!/usr/bin/env python3
"""
Evolve-Anything Streamlit Visualization Application

A modern, interactive web interface for exploring evolution databases and meta files.
Built with Streamlit for improved usability and aesthetics.
"""

import os
import re
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from evolve_anything.database import DatabaseConfig, ProgramDatabase

# ============================================================================
# Configuration & Constants
# ============================================================================
CACHE_EXPIRATION_SECONDS = 5
PAGE_SIZE = 50

# Color palette - Modern dark theme with vibrant accents
COLORS = {
    "primary": "#FF6B6B",  # Coral red
    "secondary": "#4ECDC4",  # Teal
    "accent": "#FFE66D",  # Yellow
    "success": "#2ECC71",  # Green
    "warning": "#F39C12",  # Orange
    "error": "#E74C3C",  # Red
    "dark": "#1A1A2E",  # Dark blue
    "darker": "#16213E",  # Darker blue
    "light": "#EAEAEA",  # Light gray
    "muted": "#8E8E93",  # Muted gray
}

# ============================================================================
# Page Configuration & Custom Styling
# ============================================================================
st.set_page_config(
    page_title="Evolve-Anything Viewer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for premium aesthetics
st.markdown(
    """
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global styles */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1A2E 0%, #16213E 100%);
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #EAEAEA !important;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        color: #8E8E93;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, #1A1A2E, #16213E);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(78, 205, 196, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(78, 205, 196, 0.2);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4ECDC4;
        margin-bottom: 0.25rem;
    }

    .metric-label {
        color: #8E8E93;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Program card */
    .program-card {
        background: linear-gradient(145deg, #1E1E32, #252542);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4ECDC4;
        transition: all 0.3s ease;
    }

    .program-card:hover {
        border-left-color: #FF6B6B;
        background: linear-gradient(145deg, #252542, #2A2A52);
    }

    .program-id {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #FFE66D;
    }

    .program-score {
        font-size: 1.5rem;
        font-weight: 600;
        color: #4ECDC4;
    }

    /* Code block styling */
    .code-block {
        background: #0D1117;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        overflow-x: auto;
        border: 1px solid rgba(78, 205, 196, 0.3);
    }

    /* Meta content styling */
    .meta-content {
        background: linear-gradient(145deg, #1A1A2E, #16213E);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 107, 107, 0.2);
        line-height: 1.8;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        background: #1A1A2E;
        border: 1px solid rgba(78, 205, 196, 0.3);
        border-radius: 8px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(145deg, #1A1A2E, #16213E);
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        color: #EAEAEA;
        border: 1px solid rgba(78, 205, 196, 0.2);
        border-bottom: none;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
        color: #1A1A2E;
        font-weight: 600;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(145deg, #1E1E32, #252542);
        border-radius: 8px;
        color: #EAEAEA !important;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
    }

    /* Dataframe styling */
    .dataframe {
        font-family: 'Inter', sans-serif !important;
    }

    /* Status badges */
    .status-correct {
        background: linear-gradient(135deg, #2ECC71, #27AE60);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .status-incorrect {
        background: linear-gradient(135deg, #E74C3C, #C0392B);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Island badge */
    .island-badge {
        background: linear-gradient(135deg, #9B59B6, #8E44AD);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    /* Animation keyframes */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #1A1A2E;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #4ECDC4, #44A08D);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #5EDDD4, #54B09D);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Text readability fixes */
    .stMarkdown, .stText, p, span, label, .stRadio label, .stCheckbox label {
        color: #EAEAEA !important;
    }

    .stSlider label, .stSelectbox label, .stTextInput label, .stTextArea label {
        color: #EAEAEA !important;
    }

    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #4ECDC4 !important;
    }

    /* Metric labels */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #EAEAEA !important;
    }

    /* Expander text */
    .streamlit-expanderHeader p {
        color: #EAEAEA !important;
    }

    /* JSON viewer */
    .stJson {
        background: #1A1A2E !important;
    }

    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        color: #EAEAEA !important;
        background-color: #1A1A2E !important;
    }

    /* Number input */
    .stNumberInput input {
        color: #EAEAEA !important;
        background-color: #1A1A2E !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# Caching & Data Loading
# ============================================================================
@st.cache_data(ttl=CACHE_EXPIRATION_SECONDS)
def scan_databases(search_root: str) -> List[Dict[str, Any]]:
    """Scan the search root directory for .db files."""
    db_files = []
    date_pattern = re.compile(r"_(\d{8}_\d{6})")
    task_name = os.path.basename(search_root)

    if os.path.exists(search_root):
        for root, _, files in os.walk(search_root):
            for f in files:
                if f.lower().endswith((".db", ".sqlite")):
                    full_path = os.path.join(root, f)
                    client_path = os.path.relpath(full_path, search_root)
                    display_name = f"{Path(f).stem} - {Path(client_path).parent}"

                    # Extract date for sorting
                    sort_key = "0"
                    match = date_pattern.search(client_path)
                    if match:
                        sort_key = match.group(1)

                    db_info = {
                        "path": full_path,
                        "relative_path": client_path,
                        "name": display_name,
                        "sort_key": sort_key,
                        "task": task_name,
                    }
                    db_files.append(db_info)

    # Sort by date, newest first
    db_files.sort(key=lambda x: x.get("sort_key", "0"), reverse=True)
    return db_files


@st.cache_data(ttl=CACHE_EXPIRATION_SECONDS)
def load_programs(db_path: str) -> List[Dict[str, Any]]:
    """Load all programs from a database file with retry logic."""
    max_retries = 5
    delay = 0.1

    for i in range(max_retries):
        db = None
        try:
            config = DatabaseConfig(db_path=db_path)
            db = ProgramDatabase(config, read_only=True)

            if db.cursor:
                db.cursor.execute("PRAGMA busy_timeout = 10000;")
                db.cursor.execute("PRAGMA journal_mode = WAL;")

            programs = db.get_all_programs()
            return [p.to_dict() for p in programs]

        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            error_str = str(e).lower()
            if "database is locked" in error_str or "busy" in error_str:
                if i < max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 1.5, 2.0)
                    continue
            raise e
        finally:
            if db and hasattr(db, "close"):
                try:
                    db.close()
                except Exception:
                    pass

    return []


def get_meta_files(db_path: str) -> List[Dict[str, Any]]:
    """List available meta_{gen}.txt files for a given database."""
    db_dir = os.path.dirname(db_path)
    meta_files = []

    if os.path.exists(db_dir):
        for file in os.listdir(db_dir):
            if file.startswith("meta_") and file.endswith(".txt"):
                gen_str = file[5:-4]
                try:
                    generation = int(gen_str)
                    meta_files.append(
                        {
                            "generation": generation,
                            "filename": file,
                            "path": os.path.join(db_dir, file),
                        }
                    )
                except ValueError:
                    continue

    meta_files.sort(key=lambda x: x["generation"])
    return meta_files


def load_meta_content(meta_path: str) -> str:
    """Load the content of a meta file."""
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error loading file: {e}"


# ============================================================================
# Visualization Components
# ============================================================================
def render_header():
    """Render the main header with logo and title."""
    col1, col2 = st.columns([1, 10])
    with col1:
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=85)
        else:
            st.markdown("# 🧬")
    with col2:
        st.markdown(
            '<h1 class="main-header">Evolve-Anything Viewer</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="sub-header">Explore and analyze your evolutionary runs</p>',
            unsafe_allow_html=True,
        )


def render_metrics_dashboard(programs: List[Dict[str, Any]]):
    """Render the metrics dashboard with key statistics."""
    if not programs:
        st.warning("No programs found in this database.")
        return

    df = pd.DataFrame(programs)

    # Calculate metrics
    total_programs = len(programs)
    max_generation = df["generation"].max() if "generation" in df.columns else 0
    best_score = df["combined_score"].max() if "combined_score" in df.columns else 0
    correct_count = df["correct"].sum() if "correct" in df.columns else 0
    correct_pct = (correct_count / total_programs * 100) if total_programs > 0 else 0
    num_islands = df["island_idx"].nunique() if "island_idx" in df.columns else 0

    # Render metric cards
    cols = st.columns(5)

    with cols[0]:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{total_programs:,}</div>
            <div class="metric-label">Total Programs</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with cols[1]:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{max_generation}</div>
            <div class="metric-label">Generations</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with cols[2]:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{best_score:.4f}</div>
            <div class="metric-label">Best Score</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with cols[3]:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{correct_pct:.1f}%</div>
            <div class="metric-label">Correct Rate</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with cols[4]:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{num_islands}</div>
            <div class="metric-label">Islands</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_evolution_chart(programs: List[Dict[str, Any]]):
    """Render the evolution progress chart."""
    if not programs:
        return

    df = pd.DataFrame(programs)

    if "generation" not in df.columns or "combined_score" not in df.columns:
        st.warning("Missing required columns for evolution chart.")
        return

    # Aggregate by generation
    gen_stats = (
        df.groupby("generation")
        .agg({"combined_score": ["mean", "max", "min", "std"], "id": "count"})
        .reset_index()
    )
    gen_stats.columns = [
        "generation",
        "mean_score",
        "max_score",
        "min_score",
        "std_score",
        "program_count",
    ]
    gen_stats["std_score"] = gen_stats["std_score"].fillna(0)

    # Create the plot
    fig = go.Figure()

    # Add confidence band
    fig.add_trace(
        go.Scatter(
            x=list(gen_stats["generation"]) + list(gen_stats["generation"][::-1]),
            y=list(gen_stats["mean_score"] + gen_stats["std_score"])
            + list((gen_stats["mean_score"] - gen_stats["std_score"])[::-1]),
            fill="toself",
            fillcolor="rgba(78, 205, 196, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="Std Dev",
        )
    )

    # Max score line
    fig.add_trace(
        go.Scatter(
            x=gen_stats["generation"],
            y=gen_stats["max_score"],
            mode="lines+markers",
            name="Best Score",
            line=dict(color="#FF6B6B", width=3),
            marker=dict(size=8, symbol="diamond"),
        )
    )

    # Mean score line
    fig.add_trace(
        go.Scatter(
            x=gen_stats["generation"],
            y=gen_stats["mean_score"],
            mode="lines+markers",
            name="Mean Score",
            line=dict(color="#4ECDC4", width=2),
            marker=dict(size=6),
        )
    )

    # Min score line
    fig.add_trace(
        go.Scatter(
            x=gen_stats["generation"],
            y=gen_stats["min_score"],
            mode="lines",
            name="Min Score",
            line=dict(color="#8E8E93", width=1, dash="dot"),
        )
    )

    fig.update_layout(
        title=dict(
            text="Evolution Progress",
            font=dict(size=20, color="#EAEAEA"),
        ),
        xaxis=dict(
            title="Generation",
            gridcolor="rgba(255,255,255,0.1)",
            color="#EAEAEA",
        ),
        yaxis=dict(
            title="Combined Score",
            gridcolor="rgba(255,255,255,0.1)",
            color="#EAEAEA",
        ),
        plot_bgcolor="#1A1A2E",
        paper_bgcolor="#1A1A2E",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#EAEAEA"),
        ),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=80, b=60),
    )

    st.plotly_chart(fig, width="stretch")


def render_embedding_visualization(programs: List[Dict[str, Any]]):
    """Render 2D or 3D embedding visualization."""
    if not programs:
        return

    df = pd.DataFrame(programs)

    # Check for 2D embeddings
    has_2d = (
        "embedding_pca_2d" in df.columns
        and df["embedding_pca_2d"]
        .apply(lambda x: isinstance(x, list) and len(x) >= 2)
        .any()
    )

    # Check for 3D embeddings
    has_3d = (
        "embedding_pca_3d" in df.columns
        and df["embedding_pca_3d"]
        .apply(lambda x: isinstance(x, list) and len(x) >= 3)
        .any()
    )

    if not has_2d and not has_3d:
        st.info("No PCA embeddings available for visualization.")
        return

    viz_type = st.radio(
        "Visualization", ["2D", "3D"], horizontal=True, disabled=not has_3d
    )

    if viz_type == "2D" and has_2d:
        # Extract 2D coordinates
        valid_df = df[
            df["embedding_pca_2d"].apply(lambda x: isinstance(x, list) and len(x) >= 2)
        ].copy()
        valid_df["x"] = valid_df["embedding_pca_2d"].apply(lambda x: x[0])
        valid_df["y"] = valid_df["embedding_pca_2d"].apply(lambda x: x[1])

        fig = px.scatter(
            valid_df,
            x="x",
            y="y",
            color="generation",
            size="combined_score",
            hover_data=["id", "combined_score", "correct"],
            custom_data=["id"],
            color_continuous_scale="Viridis",
            title="Program Embeddings (2D PCA)",
        )

        fig.update_layout(
            plot_bgcolor="#1A1A2E",
            paper_bgcolor="#1A1A2E",
            font=dict(color="#EAEAEA"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            clickmode="event+select",
            dragmode="select",  # Default to select for easier clicking
        )

        event = st.plotly_chart(
            fig,
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key="emb_2d",
        )

        if event and event.selection and event.selection["points"]:
            point = event.selection["points"][0]
            if "customdata" in point:
                # px puts custom_data in a list
                selected_id = point["customdata"][0]
                st.session_state["target_program_id"] = selected_id
                st.session_state["current_view"] = "📦 Programs"
                if "nav_selection" in st.session_state:
                    del st.session_state["nav_selection"]
                st.rerun()

    elif viz_type == "3D" and has_3d:
        # Extract 3D coordinates
        valid_df = df[
            df["embedding_pca_3d"].apply(lambda x: isinstance(x, list) and len(x) >= 3)
        ].copy()
        valid_df["x"] = valid_df["embedding_pca_3d"].apply(lambda x: x[0])
        valid_df["y"] = valid_df["embedding_pca_3d"].apply(lambda x: x[1])
        valid_df["z"] = valid_df["embedding_pca_3d"].apply(lambda x: x[2])

        fig = px.scatter_3d(
            valid_df,
            x="x",
            y="y",
            z="z",
            color="generation",
            size="combined_score",
            hover_data=["id", "combined_score", "correct"],
            custom_data=["id"],
            color_continuous_scale="Viridis",
            title="Program Embeddings (3D PCA)",
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    backgroundcolor="#1A1A2E", gridcolor="rgba(255,255,255,0.1)"
                ),
                yaxis=dict(
                    backgroundcolor="#1A1A2E", gridcolor="rgba(255,255,255,0.1)"
                ),
                zaxis=dict(
                    backgroundcolor="#1A1A2E", gridcolor="rgba(255,255,255,0.1)"
                ),
            ),
            paper_bgcolor="#1A1A2E",
            font=dict(color="#EAEAEA"),
            clickmode="event+select",
        )

        event = st.plotly_chart(
            fig,
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key="emb_3d",
        )

        if event and event.selection and event.selection["points"]:
            point = event.selection["points"][0]
            if "customdata" in point:
                # px puts custom_data in a list
                selected_id = point["customdata"][0]
                st.session_state["target_program_id"] = selected_id
                st.session_state["current_view"] = "📦 Programs"
                if "nav_selection" in st.session_state:
                    del st.session_state["nav_selection"]
                st.rerun()


def render_island_distribution(programs: List[Dict[str, Any]]):
    """Render island distribution chart."""
    if not programs:
        return

    df = pd.DataFrame(programs)

    if "island_idx" not in df.columns:
        st.info("No island information available.")
        return

    island_stats = (
        df.groupby("island_idx")
        .agg({"id": "count", "combined_score": ["mean", "max"], "correct": "sum"})
        .reset_index()
    )
    island_stats.columns = [
        "island",
        "count",
        "mean_score",
        "best_score",
        "correct_count",
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=island_stats["island"],
            y=island_stats["count"],
            name="Program Count",
            marker_color="#4ECDC4",
            text=island_stats["count"],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=island_stats["island"],
            y=island_stats["best_score"],
            name="Best Score",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#FF6B6B", width=3),
            marker=dict(size=10, symbol="star"),
        )
    )

    fig.update_layout(
        title=dict(
            text="Island Distribution",
            font=dict(size=20, color="#EAEAEA"),
        ),
        xaxis=dict(
            title="Island Index",
            gridcolor="rgba(255,255,255,0.1)",
            color="#EAEAEA",
        ),
        yaxis=dict(
            title="Program Count",
            gridcolor="rgba(255,255,255,0.1)",
            color="#EAEAEA",
        ),
        yaxis2=dict(
            title="Best Score",
            overlaying="y",
            side="right",
            color="#FF6B6B",
        ),
        plot_bgcolor="#1A1A2E",
        paper_bgcolor="#1A1A2E",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#EAEAEA"),
        ),
        bargap=0.3,
    )

    st.plotly_chart(fig, width="stretch")


def render_program_list(
    programs: List[Dict[str, Any]],
    db_path: Optional[str] = None,
    highlight_id: Optional[str] = None,
):
    """Render the program list with filtering and details."""
    if not programs:
        return

    df = pd.DataFrame(programs)

    # If highlight_id is provided, show a banner and default filter to that program
    if highlight_id:
        st.info(f"Targeting program: `{highlight_id}`")
        # Ensure the view focuses on this program (could force filter)
        df_highlight = df[df["id"] == highlight_id]
        if not df_highlight.empty:
            # We could just show this one, or show it at the top
            pass

    # Filtering controls
    st.markdown("### 🔍 Filter Programs")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gen_range = st.slider(
            "Generation Range",
            min_value=int(df["generation"].min()),
            max_value=int(df["generation"].max()),
            value=(int(df["generation"].min()), int(df["generation"].max())),
        )

    with col2:
        score_range = st.slider(
            "Score Range",
            min_value=float(df["combined_score"].min()),
            max_value=float(df["combined_score"].max()),
            value=(
                float(df["combined_score"].min()),
                float(df["combined_score"].max()),
            ),
        )

    with col3:
        correct_filter = st.selectbox(
            "Correctness",
            ["All", "Correct Only", "Incorrect Only"],
        )

    with col4:
        if "island_idx" in df.columns:
            islands = ["All"] + sorted(df["island_idx"].dropna().unique().tolist())
            island_filter = st.selectbox("Island", islands)
        else:
            island_filter = "All"

    # Apply filters
    if highlight_id:
        filtered_df = df[df["id"] == highlight_id].copy()
    else:
        filtered_df = df[
            (df["generation"] >= gen_range[0])
            & (df["generation"] <= gen_range[1])
            & (df["combined_score"] >= score_range[0])
            & (df["combined_score"] <= score_range[1])
        ].copy()

        # Handle 'correct' column filtering - ensure boolean type
        if "correct" in filtered_df.columns:
            filtered_df["correct"] = filtered_df["correct"].astype(bool)
            if correct_filter == "Correct Only":
                filtered_df = filtered_df[filtered_df["correct"]]
            elif correct_filter == "Incorrect Only":
                filtered_df = filtered_df[~filtered_df["correct"]]

        if island_filter != "All" and "island_idx" in df.columns:
            filtered_df = filtered_df[filtered_df["island_idx"] == island_filter]

    # Sort options
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Showing {len(filtered_df)} of {len(df)} programs**")
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ["combined_score", "generation", "complexity", "timestamp"],
            label_visibility="collapsed",
        )

    filtered_df = filtered_df.sort_values(sort_by, ascending=False)

    # Pagination
    page = st.number_input(
        "Page",
        min_value=1,
        max_value=max(1, len(filtered_df) // PAGE_SIZE + 1),
        value=1,
    )
    start_idx = (page - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
    page_df = filtered_df.iloc[start_idx:end_idx]

    # Display programs
    for _, program in page_df.iterrows():
        # Auto-expand if this is the highlighted program
        is_highlighted = highlight_id and program["id"] == highlight_id

        with st.expander(
            f"**{program['id'][:16]}...** | "
            f"Score: {program['combined_score']:.4f} | "
            f"Gen: {program['generation']} | "
            f"{'✅' if program.get('correct') else '❌'}",
            expanded=is_highlighted,
        ):
            render_program_details(program.to_dict(), db_path, key_prefix="list_")


def render_program_details(
    program: Dict[str, Any], db_path: Optional[str] = None, key_prefix: str = ""
):
    """Render detailed view of a single program."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 📊 Metrics")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

        with metrics_col1:
            st.metric("Combined Score", f"{program.get('combined_score', 0):.4f}")
        with metrics_col2:
            st.metric("Generation", program.get("generation", 0))
        with metrics_col3:
            st.metric("Complexity", f"{program.get('complexity', 0):.2f}")

        # Public metrics
        if program.get("public_metrics"):
            st.markdown("##### Public Metrics")
            st.json(program["public_metrics"])

        # Private metrics
        if program.get("private_metrics"):
            st.markdown("##### Private Metrics")
            st.json(program["private_metrics"])

    with col2:
        st.markdown("#### 📋 Info")
        st.markdown(f"**ID:** `{program.get('id', 'N/A')}`")
        st.markdown(f"**Parent:** `{program.get('parent_id', 'None')}`")
        st.markdown(f"**Island:** {program.get('island_idx', 'N/A')}")
        st.markdown(f"**Children:** {program.get('children_count', 0)}")
        st.markdown(f"**Correct:** {'✅ Yes' if program.get('correct') else '❌ No'}")

        if program.get("timestamp"):
            ts = datetime.fromtimestamp(program["timestamp"])
            st.markdown(f"**Created:** {ts.strftime('%Y-%m-%d %H:%M:%S')}")

    # Visualization section - look for visualization.png
    if db_path:
        viz_path = find_program_visualization(db_path, program)
        if viz_path and os.path.exists(viz_path):
            st.markdown("#### 🖼️ Visualization")
            st.image(viz_path)
        else:
            # Try to generate visualization from extra.npz if available
            extra_path = find_program_extra_data(db_path, program)
            if extra_path and os.path.exists(extra_path):
                # Auto-generate without button press
                viz_path = generate_circle_packing_viz(extra_path, program)
                if viz_path:
                    st.markdown("#### 🖼️ Visualization (Generated)")
                    st.image(viz_path)

    # Code section
    st.markdown("#### 💻 Code")
    language = program.get("language", "python")
    code = program.get("code", "# No code available")
    st.code(code, language=language, line_numbers=True)

    # Code diff if available
    if program.get("code_diff"):
        st.markdown("#### 📝 Code Diff")
        st.code(program["code_diff"], language="diff")

    # Text feedback if available
    if program.get("text_feedback"):
        st.markdown("#### 💬 Feedback")
        st.markdown(
            f"""
        <div class="meta-content">
            {program["text_feedback"]}
        </div>
        """,
            unsafe_allow_html=True,
        )


def find_program_visualization(db_path: str, program: Dict[str, Any]) -> Optional[str]:
    """Find visualization.png for a program based on its generation."""
    db_dir = os.path.dirname(db_path)
    gen = program.get("generation", 0)

    # Try common path patterns
    patterns = [
        os.path.join(db_dir, f"gen_{gen}", "results", "visualization.png"),
        os.path.join(db_dir, f"gen_{gen}", "visualization.png"),
        os.path.join(db_dir, "best", "visualization.png")
        if gen == program.get("generation", -1)
        else None,
    ]

    for pattern in patterns:
        if pattern and os.path.exists(pattern):
            return pattern

    return None


def find_program_extra_data(db_path: str, program: Dict[str, Any]) -> Optional[str]:
    """Find extra.npz file for a program."""
    db_dir = os.path.dirname(db_path)
    gen = program.get("generation", 0)

    patterns = [
        os.path.join(db_dir, f"gen_{gen}", "results", "extra.npz"),
        os.path.join(db_dir, f"gen_{gen}", "extra.npz"),
    ]

    for pattern in patterns:
        if os.path.exists(pattern):
            return pattern

    return None


def generate_circle_packing_viz(
    extra_path: str, program: Dict[str, Any]
) -> Optional[str]:
    """Generate a circle packing visualization from extra.npz data."""
    try:
        import numpy as np
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from matplotlib.collections import PatchCollection

        # Load data
        data = np.load(extra_path)
        centers = data["centers"]
        radii = data["radii"]

        is_valid = program.get("correct", False)

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), facecolor="#1A1A2E")
        ax.set_facecolor("#1A1A2E")

        # Draw unit square boundary
        ax.plot(
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            color="#4ECDC4",
            linewidth=2,
            linestyle="--",
        )

        # Create circle patches
        circles = [Circle(centers[i], radii[i]) for i in range(len(radii))]
        colors = list(radii)

        pc = PatchCollection(
            circles, cmap="viridis", alpha=0.8, edgecolor="white", linewidth=0.5
        )
        pc.set_array(np.array(colors))
        ax.add_collection(pc)

        # Colorbar
        cbar = plt.colorbar(pc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Radius", color="#EAEAEA", fontsize=10)
        cbar.ax.yaxis.set_tick_params(color="#EAEAEA")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#EAEAEA")

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")

        total_radius = np.sum(radii)
        status = "✓ Valid" if is_valid else "✗ Invalid"
        ax.set_title(
            f"Circle Packing (n={len(radii)})\nSum of Radii: {total_radius:.4f} {status}",
            color="#EAEAEA",
            fontsize=14,
            fontweight="bold",
        )

        ax.tick_params(colors="#8E8E93")
        for spine in ax.spines.values():
            spine.set_color("#4ECDC4")
            spine.set_alpha(0.3)

        ax.set_xlabel("x", color="#EAEAEA")
        ax.set_ylabel("y", color="#EAEAEA")

        # Save to same directory as extra.npz
        viz_path = os.path.join(os.path.dirname(extra_path), "visualization.png")
        plt.savefig(
            viz_path,
            dpi=150,
            bbox_inches="tight",
            facecolor="#1A1A2E",
            edgecolor="none",
        )
        plt.close(fig)

        return viz_path

    except Exception as e:
        st.error(f"Error generating visualization: {e}")
        return None


def render_meta_explorer(db_path: str):
    """Render the meta files explorer."""
    meta_files = get_meta_files(db_path)

    if not meta_files:
        st.info("No meta files found for this database.")
        return

    st.markdown("### 📚 Meta Files")
    st.markdown(f"Found **{len(meta_files)}** meta scratchpad files.")

    # Slider for generation selection
    generations = [m["generation"] for m in meta_files]
    selected_gen = st.select_slider(
        "Select Generation",
        options=generations,
        value=generations[-1] if generations else None,
        key="meta_gen_selector_slider",
    )

    if selected_gen is not None:
        meta_info = next(
            (m for m in meta_files if m["generation"] == selected_gen), None
        )

        if meta_info:
            content = load_meta_content(meta_info["path"])

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"#### Generation {selected_gen} Meta Scratchpad")
            with col2:
                st.download_button(
                    "📥 Download",
                    data=content,
                    file_name=meta_info["filename"],
                    mime="text/plain",
                )

            st.markdown(
                f"""
            <div class="meta-content" style="max-height: 600px; overflow-y: auto;">
                <pre style="white-space: pre-wrap; word-wrap: break-word; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; margin: 0;">{content}</pre>
            </div>
            """,
                unsafe_allow_html=True,
            )


def render_genealogy_tree(
    programs: List[Dict[str, Any]],
    db_path: Optional[str] = None,
    selected_id: Optional[str] = None,
):
    """Render a hierarchical genealogy tree visualization."""
    if not programs:
        return

    df = pd.DataFrame(programs)

    # Build lookup maps
    id_to_generation = {p["id"]: p.get("generation", 0) for p in programs}
    id_to_score = {p["id"]: p.get("combined_score", 0) for p in programs}
    id_to_correct = {p["id"]: p.get("correct", False) for p in programs}
    id_to_island = {p["id"]: p.get("island_idx", 0) for p in programs}
    id_to_patch_type = {}
    for p in programs:
        pid = p.get("parent_id")
        if not pid and "metadata" in p:
            pid = p["metadata"].get("parent_id")

        if not pid:
            id_to_patch_type[p["id"]] = "initial"
        else:
            meta = p.get("metadata", {})
            id_to_patch_type[p["id"]] = meta.get("patch_type", "diff")

    # Build parent-child relationships
    children_map: Dict[str, List[str]] = {}  # parent_id -> list of child_ids
    root_nodes = []

    for prog in programs:
        parent_id = prog.get("parent_id")
        # Check metadata for parent_id if not present (e.g. full edits)
        if not parent_id and "metadata" in prog:
            parent_id = prog["metadata"].get("parent_id")

        prog_id = prog["id"]

        if parent_id:
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(prog_id)
        else:
            root_nodes.append(prog_id)

    if not root_nodes and not children_map:
        st.info("No genealogy data available (no parent-child relationships found).")
        return

    # If no roots found but we have edges, find nodes with no parents
    if not root_nodes:
        all_children = set()
        for children in children_map.values():
            all_children.update(children)
        all_parents = set(children_map.keys())
        root_nodes = list(all_parents - all_children)

    if not root_nodes:
        # Fallback: use nodes with generation 0
        root_nodes = [p["id"] for p in programs if p.get("generation", 0) == 0]

    if not root_nodes:
        st.warning("Could not determine root nodes for the tree.")
        return

    # Controls
    st.markdown("### 🌳 Evolution Tree")

    col1, col2, col3 = st.columns(3)
    with col1:
        max_gen = int(df["generation"].max()) if "generation" in df.columns else 10
        max_depth = st.slider(
            "Max Generations to Display",
            min_value=1,
            max_value=max(max_gen, 1),
            value=min(max_gen, 15),
        )
    with col2:
        show_only_correct = st.checkbox("Show only correct programs", value=False)
    with col3:
        highlight_best = st.checkbox("Highlight best programs", value=True)

    # Calculate hierarchical positions using BFS
    # Y-axis = generation (inverted so roots are at top)
    # X-axis = horizontal position within generation

    node_positions = {}
    node_data = {}

    # Group nodes by generation for proper horizontal spacing
    nodes_by_generation: Dict[int, List[str]] = {}

    def collect_nodes(node_id: str, depth: int = 0):
        """Recursively collect nodes and their depths."""
        gen = id_to_generation.get(node_id, depth)

        if gen > max_depth:
            return

        # Skip incorrect programs if filter is on
        if show_only_correct and not id_to_correct.get(node_id, False):
            return

        if gen not in nodes_by_generation:
            nodes_by_generation[gen] = []
        if node_id not in nodes_by_generation[gen]:
            nodes_by_generation[gen].append(node_id)

        # Process children
        for child_id in children_map.get(node_id, []):
            collect_nodes(child_id, depth + 1)

    # Collect all nodes starting from roots
    for root_id in root_nodes:
        collect_nodes(root_id)

    if not nodes_by_generation:
        st.info("No nodes to display with current filters.")
        return

    # Calculate positions using INDEPENDENT ISLAND TREES with DUPLICATE NODES
    # 1. Identify nodes for each island
    # 2. For migrations P(Island A) -> C(Island B):
    #    - Add P_mirror to Island B's set
    #    - Add edge P_mirror -> C in Island B's local tree
    # 3. Layout each island independently

    # Data structures for per-island layout
    island_nodes_set: Dict[int, Set[str]] = {}
    island_adjacency: Dict[int, Dict[str, List[str]]] = {}
    distinct_islands = sorted(list(set(id_to_island.values())))

    # Initialize structures
    for isl in distinct_islands:
        island_nodes_set[isl] = set()
        island_adjacency[isl] = {}

    # Helper to clean ID (remove mirror suffix)
    def get_real_id(nid: str) -> str:
        return nid.split("__mirror")[0]

    # Populate island nodes and connections
    for parent_id, children in children_map.items():
        parent_island = id_to_island.get(parent_id, 0)

        for child_id in children:
            child_island = id_to_island.get(child_id, 0)

            # Filter visibility
            if id_to_generation.get(child_id, 0) > max_depth:
                continue
            if show_only_correct and not id_to_correct.get(child_id, False):
                continue

            # Intra-island (Same Island)
            if parent_island == child_island:
                isl = parent_island
                island_nodes_set[isl].add(parent_id)
                island_nodes_set[isl].add(child_id)

                if parent_id not in island_adjacency[isl]:
                    island_adjacency[isl][parent_id] = []
                island_adjacency[isl][parent_id].append(child_id)

            # Inter-island (Migration)
            else:
                # Add child to its island
                island_nodes_set[child_island].add(child_id)
                # Create MIRROR pair of parent in child's island
                parent_mirror_id = f"{parent_id}__mirror_{child_island}"
                island_nodes_set[child_island].add(parent_mirror_id)

                if parent_mirror_id not in island_adjacency[child_island]:
                    island_adjacency[child_island][parent_mirror_id] = []
                island_adjacency[child_island][parent_mirror_id].append(child_id)

    # Ensure locally-root nodes are added
    for root_id in root_nodes:
        isl = id_to_island.get(root_id, 0)
        if isl in island_nodes_set:
            island_nodes_set[isl].add(root_id)

    # Layout each island
    island_width = 15.0
    island_gap = 5.0
    island_x_centers = {}  # To store center for labels

    for i, isl in enumerate(distinct_islands):
        local_nodes = island_nodes_set[isl]
        local_adj = island_adjacency[isl]

        # Find local roots
        all_local_children = set()
        for kids in local_adj.values():
            all_local_children.update(kids)

        local_roots = [n for n in local_nodes if n not in all_local_children]
        local_roots.sort(
            key=lambda n: (
                # Sort by Source Island ASCENDING (negate for reverse=True)
                # This puts mirrors from Left islands on Left, Native in Middle, Mirrors from Right on Right
                -id_to_island.get(get_real_id(n), isl),
                id_to_generation.get(get_real_id(n), 0),
                id_to_score.get(get_real_id(n), 0),
            ),
            reverse=True,
        )

        # Subtree sizing
        local_subtree_sizes = {}

        def get_local_subtree_size(nid):
            if nid in local_subtree_sizes:
                return local_subtree_sizes[nid]
            children = local_adj.get(nid, [])
            if not children:
                local_subtree_sizes[nid] = 1
                return 1
            size = sum(get_local_subtree_size(c) for c in children)
            local_subtree_sizes[nid] = max(size, 1)
            return size

        for r in local_roots:
            get_local_subtree_size(r)

        island_start_x = i * (island_width + island_gap)

        # Determine scaling factor
        total_island_size = sum(local_subtree_sizes.get(r, 1) for r in local_roots)
        effective_island_width = max(island_width, total_island_size * 1.5)

        # Placement recursive func
        def place_node(nid, x_min, x_max):
            real_id = get_real_id(nid)
            gen = id_to_generation.get(real_id, 0)
            x = (x_min + x_max) / 2
            y = -gen * 2.0
            node_positions[nid] = (x, y)

            # Store metadata
            is_mirror = "__mirror" in nid
            node_data[nid] = {
                "score": id_to_score.get(real_id, 0),
                "generation": gen,
                "correct": id_to_correct.get(real_id, False),
                "island": isl,
                "is_mirror": is_mirror,
                "real_id": real_id,
            }

            children = local_adj.get(nid, [])
            if not children:
                return

            total_size = sum(local_subtree_sizes.get(c, 1) for c in children)
            curr_x = x_min
            for c in children:
                c_size = local_subtree_sizes.get(c, 1)
                c_width = (x_max - x_min) * (c_size / total_size)
                place_node(c, curr_x, curr_x + c_width)
                curr_x += c_width

        # Place roots
        current_root_x = island_start_x
        for r in local_roots:
            r_size = local_subtree_sizes.get(r, 1)
            r_width = (
                effective_island_width * (r_size / total_island_size)
                if total_island_size > 0
                else effective_island_width
            )
            place_node(r, current_root_x, current_root_x + r_width)
            current_root_x += r_width

        # Calculate center for label based on actual placements
        placed_xs = [node_positions[n][0] for n in local_nodes if n in node_positions]
        if placed_xs:
            island_x_centers[isl] = (min(placed_xs) + max(placed_xs)) / 2
        else:
            island_x_centers[isl] = island_start_x + effective_island_width / 2

    # Build edges from local adjacency
    edges = []
    for isl in distinct_islands:
        for p, kids in island_adjacency[isl].items():
            if p in node_positions:
                for k in kids:
                    if k in node_positions:
                        edges.append((p, k))

    # Find best score for highlighting
    best_score = max(id_to_score.values()) if id_to_score else 0
    best_nodes = {nid for nid, score in id_to_score.items() if score == best_score}

    # VISUAL INTEGRATION: Connect mirrors to closest local branch at same generation
    # Map (Island, Gen) -> List of Local (non-mirror) Nodes
    local_nodes_by_gen: Dict[tuple, List[str]] = {}
    for nid in node_positions:
        if "__mirror" not in nid:
            d = node_data[nid]
            key = (d["island"], d["generation"])
            if key not in local_nodes_by_gen:
                local_nodes_by_gen[key] = []
            local_nodes_by_gen[key].append(nid)

    integration_edges_traces = []
    for nid, (mx, my) in node_positions.items():
        if "__mirror" in nid:
            d = node_data[nid]
            key = (d["island"], d["generation"])

            candidates = local_nodes_by_gen.get(key, [])
            if candidates:
                # Find closest neighbor by X coordinate
                closest = min(candidates, key=lambda c: abs(node_positions[c][0] - mx))
                cx, cy = node_positions[closest]

                integration_edges_traces.append(
                    go.Scatter(
                        x=[cx, mx],
                        y=[cy, my],
                        mode="lines",
                        line=dict(
                            width=1,
                            color="rgba(255, 255, 255, 0.3)",  # Faint white/grey
                            dash="dot",
                        ),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )

    # Create edge traces - straight lines for tree layout
    edge_traces = []
    for parent_id, child_id in edges:
        x0, y0 = node_positions[parent_id]
        x1, y1 = node_positions[child_id]

        parent_island = node_data[parent_id]["island"]
        child_island = node_data[child_id]["island"]
        is_migration = parent_island != child_island

        # Simple elbow line: vertical from parent, then horizontal to child
        mid_y = (y0 + y1) / 2

        edge_traces.append(
            go.Scatter(
                x=[x0, x0, x1, x1],
                y=[y0, mid_y, mid_y, y1],
                mode="lines",
                line=dict(
                    width=2 if is_migration else 1.5,
                    color="#FF9F1C" if is_migration else "rgba(78, 205, 196, 0.5)",
                    dash="dot" if is_migration else "solid",
                ),
                hoverinfo="text" if is_migration else "none",
                text=f"<b>Cross-island:</b> {parent_island} → {child_island}"
                if is_migration
                else None,
                showlegend=False,
            )
        )

    # ADD CONNECTORS for MIRRORS (Parent -> Parent Mirror)
    # This visually connects the trees across islands
    mirror_edges = []
    for node_id, (mx, my) in node_positions.items():
        if "__mirror" in node_id:
            real_id = node_data[node_id].get("real_id")
            if real_id and real_id in node_positions:
                px, py = node_positions[real_id]

                # Draw horizontal line from Parent to Mirror
                mirror_edges.append(
                    go.Scatter(
                        x=[px, mx],
                        y=[py, my],
                        mode="lines",
                        line=dict(
                            width=1.5,
                            color="rgba(255, 159, 28, 0.6)",  # Orangeish transparency
                            dash="dash",
                        ),
                        hoverinfo="text",
                        text=f"<b>Migration Link</b><br>{real_id[:8]}... → Island {node_data[node_id]['island']}",
                        showlegend=False,
                    )
                )

    edge_traces.extend(mirror_edges)
    edge_traces.extend(integration_edges_traces)

    # Create node traces
    # Correct nodes (including Best)
    (
        node_x_c,
        node_y_c,
        node_text_c,
        node_color_c,
        node_size_c,
        node_symbol_c,
        node_line_color_c,
        node_line_width_c,
        node_customdata_c,
    ) = [], [], [], [], [], [], [], [], []

    # Incorrect nodes
    node_x_i, node_y_i, node_text_i, node_size_i, node_customdata_i = [], [], [], [], []

    for node_id, (x, y) in node_positions.items():
        data = node_data[node_id]
        score = data["score"]
        gen = data["generation"]
        correct = data["correct"]
        island = data["island"]
        is_mirror = data.get("is_mirror", False)
        real_id = data.get("real_id", node_id)

        patch_type = id_to_patch_type.get(real_id, "diff")

        hover_text = (
            f"<b>ID:</b> {real_id[:16]}...<br>"
            f"<b>Generation:</b> {gen}<br>"
            f"<b>Island:</b> {island}<br>"
            f"<b>Score:</b> {score:.4f}<br>"
            f"<b>Correct:</b> {'✅' if correct else '❌'}<br>"
            f"<b>Type:</b> {patch_type}"
        )
        if is_mirror:
            hover_text += "<br><i>(Migration Source)</i>"

        if correct:
            node_x_c.append(x)
            node_y_c.append(y)
            node_customdata_c.append(real_id)  # Point to real ID
            node_text_c.append(hover_text)
            node_color_c.append(score)

            is_best = highlight_best and real_id in best_nodes

            # Mirrors are smaller
            base_size = 20 if is_best else 14
            node_size_c.append(base_size * 0.8 if is_mirror else base_size)

            # Determine shape and style
            if is_best:
                node_symbol_c.append("star")
                node_line_color_c.append("#FFE66D")
                node_line_width_c.append(3 if not is_mirror else 1)
            else:
                # Shape based on patch type
                if patch_type == "initial":
                    node_symbol_c.append("triangle-up")
                elif patch_type == "full":
                    node_symbol_c.append("square")
                elif patch_type == "cross":
                    node_symbol_c.append("cross")
                else:  # diff
                    node_symbol_c.append("circle")

                node_line_color_c.append(
                    "#2ECC71" if not is_mirror else "rgba(46, 204, 113, 0.5)"
                )
                node_line_width_c.append(2 if not is_mirror else 1)
        else:
            node_x_i.append(x)
            node_y_i.append(y)
            node_customdata_i.append(real_id)
            node_text_i.append(hover_text)
            node_size_i.append(14 * 0.8 if is_mirror else 14)

    trace_correct = go.Scatter(
        x=node_x_c,
        y=node_y_c,
        mode="markers",
        hoverinfo="text",
        text=node_text_c,
        customdata=node_customdata_c,
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            color=node_color_c,
            size=node_size_c,
            symbol=node_symbol_c,
            colorbar=dict(
                thickness=15,
                title=dict(text="Score", side="right"),
                xanchor="left",
                y=0.5,
            ),
            line=dict(color=node_line_color_c, width=node_line_width_c),
        ),
        showlegend=False,
    )

    trace_incorrect = go.Scatter(
        x=node_x_i,
        y=node_y_i,
        mode="markers",
        hoverinfo="text",
        text=node_text_i,
        customdata=node_customdata_i,
        marker=dict(
            showscale=False,
            color="#E74C3C",
            size=node_size_i,
            symbol="x",
            line=dict(color="#E74C3C", width=2),
        ),
        showlegend=False,
    )

    # Dummy traces for legend
    dummy_traces = [
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(symbol="star", size=15, color="#FFE66D"),
            name="Best Score",
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="#2ECC71"),
            name="Initial",
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(symbol="circle", size=10, color="#2ECC71"),
            name="Diff Edit",
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(symbol="square", size=10, color="#2ECC71"),
            name="Full Edit",
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(symbol="cross", size=10, color="#2ECC71", line=dict(width=2)),
            name="Cross-Over",
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                symbol="x",
                size=10,
                color="#E74C3C",
                line=dict(width=2, color="#E74C3C"),
            ),
            name="Incorrect",
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(width=2, color="#FF9F1C", dash="dot"),
            name="Migration",
        ),
    ]

    # Create figure
    fig = go.Figure(
        data=edge_traces + [trace_correct, trace_incorrect] + dummy_traces,
        layout=go.Layout(
            title=dict(
                text=f"Evolution Tree ({len(node_positions)} programs)",
                font=dict(size=20, color="#EAEAEA"),
            ),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
                font=dict(color="#EAEAEA"),
                bgcolor="rgba(26, 26, 46, 0.9)",
                bordercolor="rgba(255, 255, 255, 0.3)",
                borderwidth=1,
            ),
            hovermode="closest",
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
                zeroline=False,
                showticklabels=True,
                tickmode="array",
                tickvals=[island_x_centers[i] for i in distinct_islands],
                ticktext=[f"Island {i}" for i in distinct_islands],
                title="",
                tickfont=dict(color="#EAEAEA", size=14),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.1)",
                zeroline=False,
                showticklabels=True,
                tickmode="array",
                tickvals=[-i * 2.0 for i in range(max_depth + 1)],
                ticktext=[f"Gen {i}" for i in range(max_depth + 1)],
                title="",
                tickfont=dict(color="#EAEAEA", size=12),
            ),
            plot_bgcolor="#1A1A2E",
            paper_bgcolor="#1A1A2E",
            font=dict(color="#EAEAEA"),
            height=max(600, max_depth * 80 + 200),
            margin=dict(l=80, r=40, t=60, b=40),
            clickmode="event+select",
            dragmode="pan",
        ),
    )

    # Interactive chart with selection event
    event = st.plotly_chart(
        fig,
        width="stretch",
        key="genealogy_chart",
        on_select="rerun",
        selection_mode="points",
    )

    # Handle selection redirection
    if event and event.selection and event.selection["points"]:
        # Get the first selected point
        point = event.selection["points"][0]
        # Check if customdata is available (it should be)
        if "customdata" in point:
            selected_id = point["customdata"]
            # Set state for redirection
            st.session_state["target_program_id"] = selected_id
            st.session_state["current_view"] = "📦 Programs"
            # Force reset of navigation widget to pick up new current_view
            if "nav_selection" in st.session_state:
                del st.session_state["nav_selection"]
            st.rerun()

    # Show statistics
    with st.expander("📊 Tree Statistics"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Nodes", len(node_positions))
        with col2:
            st.metric("Generations", len(nodes_by_generation))
        with col3:
            st.metric("Root Nodes", len(root_nodes))
        with col4:
            correct_count = sum(1 for d in node_data.values() if d["correct"])
            st.metric("Correct Programs", f"{correct_count}/{len(node_positions)}")


# ============================================================================
# Main Application
# ============================================================================
def main():
    """Main application entry point."""
    render_header()

    # Sidebar - Database Selection
    with st.sidebar:
        st.markdown("## 🗂️ Database")

        selected_db_path = None
        selected_db_name = None

        # Initialize session state
        if "recent_paths" not in st.session_state:
            st.session_state.recent_paths = []

        # Path input
        direct_path = st.text_input(
            "Results Directory Path",
            placeholder="/path/to/results_folder",
            help="Paste the full path to your evolution results directory",
        )

        # Recent paths dropdown
        if st.session_state.recent_paths:
            recent_choice = st.selectbox(
                "Or select from recent",
                options=[""] + st.session_state.recent_paths[:5],
                format_func=lambda x: os.path.basename(x)
                if x
                else "— Recent databases —",
            )
            if recent_choice:
                direct_path = recent_choice

        if direct_path:
            # Expand ~ to home directory
            direct_path = os.path.expanduser(direct_path)

            # Look for .sqlite file in directory
            db_file = None
            if os.path.exists(direct_path):
                sqlite_files = [
                    f for f in os.listdir(direct_path) if f.endswith(".sqlite")
                ]
                if sqlite_files:
                    db_file = os.path.join(direct_path, sqlite_files[0])

            if db_file and os.path.exists(db_file):
                selected_db_path = db_file
                selected_db_name = os.path.basename(direct_path)
                st.success(f"✅ {selected_db_name}")

                # Add to recent paths
                if direct_path not in st.session_state.recent_paths:
                    st.session_state.recent_paths.insert(0, direct_path)
                    st.session_state.recent_paths = st.session_state.recent_paths[:10]
            elif os.path.exists(direct_path):
                st.error("No `.sqlite` database found in this directory.")
                st.stop()
            else:
                st.error("Directory does not exist.")
                st.stop()
        else:
            st.info("Paste the path to a results folder.")
            st.stop()

    # Main content
    if selected_db_name and selected_db_path:
        # Load programs
        with st.spinner("Loading programs..."):
            try:
                programs = load_programs(selected_db_path)
            except Exception as e:
                st.error(f"Failed to load database: {e}")
                st.stop()

        if not programs:
            st.warning("No programs found in this database.")
            st.stop()

        # Metrics dashboard
        render_metrics_dashboard(programs)

        st.markdown("---")

        # Initialize session state for navigation if not exists
        if "current_view" not in st.session_state:
            st.session_state.current_view = "📈 Evolution"

        # Navigation bar
        view_options = [
            "📈 Evolution",
            "📦 Programs",
            "🌐 Embeddings",
            "📚 Meta Files",
            "🔍 Research",
            "🌳 Genealogy",
        ]

        # Helper to handle manual navigation
        def on_nav_change():
            st.session_state.current_view = st.session_state.nav_selection

        # Sync radio with session state
        selection = st.radio(
            "Navigation",
            view_options,
            index=view_options.index(st.session_state.current_view),
            horizontal=True,
            label_visibility="collapsed",
            key="nav_selection",
            on_change=on_nav_change,
        )

        # Update current view from selection (redundant but safe)
        st.session_state.current_view = selection

        # Render content based on current view
        if st.session_state.current_view == "📈 Evolution":
            col1, col2 = st.columns(2)
            with col1:
                render_evolution_chart(programs)
            with col2:
                render_island_distribution(programs)

        elif st.session_state.current_view == "📦 Programs":
            # Check if we have a selected program from redirection
            target_id = st.session_state.get("target_program_id")
            if target_id:
                st.session_state.pop("target_program_id")  # Clear after using
                render_program_list(programs, selected_db_path, highlight_id=target_id)
            else:
                render_program_list(programs, selected_db_path)

        elif st.session_state.current_view == "🌐 Embeddings":
            render_embedding_visualization(programs)

        elif st.session_state.current_view == "📚 Meta Files":
            render_meta_explorer(selected_db_path)

        elif st.session_state.current_view == "🔍 Research":
            render_research_view(selected_db_path)

        elif st.session_state.current_view == "🌳 Genealogy":
            render_genealogy_tree(programs, selected_db_path)


def run_streamlit_app():
    """Run the Streamlit app for in-module execution."""
    main()


def render_research_view(db_path: str):
    """Render the research history view."""
    import time

    st.header("🔍 Research History")

    # Construct paths
    db_path_obj = Path(db_path)
    # Assume results dir is parent of db file
    results_dir = db_path_obj.parent
    meta_memory_path = results_dir / "meta_memory.json"

    if not meta_memory_path.exists():
        st.warning(f"No meta memory found at {meta_memory_path}")
        return

    try:
        with open(meta_memory_path, "r", encoding="utf-8") as f:
            memory_data = json.load(f)

        research_history = memory_data.get("meta_research_history", [])

        if not research_history:
            st.info("No research history found yet.")
            return

        # Display research sessions in reverse chronological order
        for i, session in enumerate(reversed(research_history)):
            timestamp = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(session.get("timestamp", 0))
            )
            actions = session.get("actions", [])
            summary = session.get("summary", "")

            with st.expander(
                f"Session {len(research_history) - i}: {timestamp}", expanded=(i == 0)
            ):
                st.subheader("Actions Taken")
                if actions:
                    for action in actions:
                        st.code(action, language="text")
                else:
                    st.write("*No actions recorded*")

                st.subheader("Research Summary")
                st.markdown(summary)

    except Exception as e:
        st.error(f"Error loading research history: {e}")


def cli_main():
    """CLI entry point that launches Streamlit with proper arguments."""
    import argparse
    import subprocess
    import sys

    parser = argparse.ArgumentParser(
        description="Launch the Evolve-Anything Streamlit Evolution Viewer"
    )
    parser.add_argument(
        "root_directory",
        nargs="?",
        default=os.getcwd(),
        help="Root directory to search for database files (default: cwd)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8501,
        help="Port to run Streamlit on (default: 8501)",
    )
    parser.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        help="Open browser automatically",
    )
    args = parser.parse_args()

    # Get the path to this script
    script_path = Path(__file__).resolve()

    # Build the streamlit command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script_path),
        "--server.port",
        str(args.port),
        "--server.headless",
        "false" if args.open_browser else "true",
        "--",
        "--search-root",
        args.root_directory,
    ]

    print(f"🧬 Starting Evolve-Anything Viewer on port {args.port}...")
    print(f"📁 Search root: {args.root_directory}")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[*] Shutting down.")


if __name__ == "__main__":
    # Check if we're being run directly by Streamlit or as CLI
    import sys

    # When Streamlit runs this file, it doesn't pass __file__ arguments
    # When CLI runs this, we get streamlit as subprocess
    if len(sys.argv) > 1 and sys.argv[1] == "--search-root":
        # Called from CLI launcher with search-root argument
        os.chdir(sys.argv[2])
        main()
    else:
        # Direct execution or Streamlit run
        main()
