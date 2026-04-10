import streamlit as st
import json
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add path to import config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    import vqa_config as config
except ImportError:
    st.error("Could not find vqa_config.py")
    st.stop()

from models import get_model_info, list_models, get_model
# Configuration
MODEL_NAMES = {str(mid): get_model_info(mid)["name"] for mid in list_models()}

st.set_page_config(page_title="VQA Benchmarking", layout="wide", page_icon="📈")

def set_design():
    st.markdown("""
        <style>
        .main { background-color: #f8f9fa; }
        .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .gain-positive { color: #28a745; font-weight: bold; }
        .gain-negative { color: #dc3545; font-weight: bold; }
        table { width: 100% !important; }
        </style>
    """, unsafe_allow_html=True)

set_design()

st.title("📈 VQA Benchmarking: Full vs Subset")
st.markdown("Detailed performance comparison between models trained on the **Subset** (25k) and **Full** GQA datasets.")

@st.cache_data
def load_all_metrics():
    """Load metrics from both results and results_subset folders."""
    all_data = []
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Mode configurations
    scan_configs = [
        {"mode": "FULL", "folder": "results"},
        {"mode": "SUBSET", "folder": "results_subset"}
    ]
    
    for cfg in scan_configs:
        folder_path = os.path.join(base_dir, cfg["folder"])
        if not os.path.exists(folder_path): continue
        
        # Scan subdirectories
        for entry in os.listdir(folder_path):
            entry_path = os.path.join(folder_path, entry)
            if os.path.isdir(entry_path):
                # Try to find model ID from folder name
                model_id = None
                if entry.startswith("model_"):
                    parts = entry.split("_")
                    if len(parts) >= 2 and parts[1].isdigit():
                        model_id = parts[1]
                
                if not model_id: continue
                
                metrics_path = os.path.join(entry_path, "metrics.json")
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path, "r", encoding="utf-8") as f:
                            m = json.load(f)
                            m["Model ID"] = model_id
                            m["Mode"] = cfg["mode"]
                            m["Display Name"] = f"M{model_id} ({cfg['mode']})"
                            all_data.append(m)
                    except (FileNotFoundError, json.JSONDecodeError):
                        pass
    return all_data

raw_data = load_all_metrics()

if not raw_data:
    st.error("No metrics found. Please run evaluate.py first for at least one model.")
    st.stop()

df = pd.DataFrame(raw_data)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Benchmark Settings")
    display_modes = st.multiselect("Dataset Modes", ["FULL", "SUBSET"], default=["FULL", "SUBSET"])
    
    available_model_ids = sorted(list(df["Model ID"].unique()), key=int)
    selected_model_ids = st.multiselect("Target Models", available_model_ids, default=available_model_ids)
    
    # Detect available metrics
    exclude_cols = ["Model ID", "Mode", "Display Name", "inference_time", "num_samples", "accuracy"]
    metric_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Sort metrics: Accuracy and Cider first
    priority = ["short_accuracy", "cider", "bleu_4"]
    metric_cols = priority + sorted([c for c in metric_cols if c not in priority])
    
    selected_metrics = st.multiselect("Metrics to Show", metric_cols, default=priority)

# Filtering
filtered_df = df[(df["Mode"].isin(display_modes)) & (df["Model ID"].isin(selected_model_ids))]

if filtered_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# ---------------------------------------------------------
# Comparison Dashboard
# ---------------------------------------------------------
st.subheader("📊 Performance Visualization")

# 1. Bar Chart Comparison
for metric in selected_metrics:
    fig = px.bar(
        filtered_df, 
        x="Model ID", 
        y=metric, 
        color="Mode",
        barmode="group",
        title=f"Comparison: {metric.replace('_', ' ').upper()}",
        color_discrete_map={"FULL": "#1f77b4", "SUBSET": "#ff7f0e"},
        text_auto='.4f'
    )
    fig.update_layout(height=350, plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

# 2. Performance Scoreboard (Pivot-like)
st.subheader("📋 Detailed Scoreboard")

pivot_rows = []
for mid in selected_model_ids:
    row = {"Model": MODEL_NAMES.get(mid, f"Model {mid}")}
    for mode in ["FULL", "SUBSET"]:
        subset_data = df[(df["Model ID"] == mid) & (df["Mode"] == mode)]
        if not subset_data.empty:
            for met in selected_metrics:
                row[f"{met} ({mode})"] = subset_data.iloc[0][met]
        else:
            for met in selected_metrics:
                row[f"{met} ({mode})"] = None
    
    # Calculate Gaps for Short Accuracy
    if "short_accuracy (FULL)" in row and "short_accuracy (SUBSET)" in row:
        val_f = row["short_accuracy (FULL)"]
        val_s = row["short_accuracy (SUBSET)"]
        if val_f is not None and val_s is not None:
            gap = val_f - val_s
            row["Acc. Delta"] = f"{gap*100:+.2f}%"
        else:
            row["Acc. Delta"] = "N/A"
            
    pivot_rows.append(row)

report_df = pd.DataFrame(pivot_rows).set_index("Model")
st.dataframe(report_df.style.highlight_max(axis=0, color="#d4edda"), width="stretch")

# 3. Trends & Analysis
st.subheader("🔍 Analysis")
col1, col2 = st.columns(2)

with col1:
    st.info("**Dataset Impact:** Generally, the FULL dataset improves the **Cider** score, which measures how 'natural' and descriptive the answers are compared to human ground truth.")

with col2:
    st.success("**Consistency:** All results above are derived from evaluation on the **GQA testdev** set (12,578 samples) for absolute fairness.")

if __name__ == "__main__":
    pass
