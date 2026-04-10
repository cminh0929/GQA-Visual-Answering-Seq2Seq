import streamlit as st
import json
import os
import sys
import pandas as pd
import plotly.express as px

# Add path to import config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    import vqa_config as config
except ImportError:
    st.error("Could not find vqa_config.py")
    st.stop()

# Configuration
MODEL_NAMES = {
    "1": "Model 1: Scratch + No Attention",
    "2": "Model 2: Pretrained + No Attention",
    "3": "Model 3: Scratch + Attention",
    "4": "Model 4: Pretrained + Attention",
    "5": "Model 5: Pretrained E2E + No Attention",
    "6": "Model 6: Pretrained E2E + Attention"
}

st.set_page_config(page_title="VQA Model Comparison", layout="wide", page_icon="📊")

st.title("📊 VQA Model Evaluation Comparison")
st.markdown("Dashboard for comparing performance metrics across selected Models.")

@st.cache_data
def load_data():
    data = {}
    for i in range(1, 7):
        model_str = f"model_{i}"
        dir_path = config.MODEL_DIRS.get(model_str)
        if dir_path:
            metrics_path = os.path.join(dir_path, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    data[str(i)] = json.load(f)
    return data

data = load_data()

if not data:
    st.error("⚠️ No `metrics.json` found in any model directory. Run `evaluate.py` first.")
    st.stop()

# ---------------------------------------------------------
# Prepare DataFrame
# ---------------------------------------------------------
rows = []
for model_id, metrics in data.items():
    row = {"Model ID": str(model_id), "Model": MODEL_NAMES.get(str(model_id), f"Model {model_id}")}
    for k, v in metrics.items():
        # Exclude "accuracy", "inference_time", "num_samples"
        if k not in ["accuracy", "inference_time", "num_samples"]:
            row[k] = v
    rows.append(row)

df = pd.DataFrame(rows)
available_metrics = [c for c in df.columns if c not in ["Model ID", "Model"]]

# ---------------------------------------------------------
# SIDEBAR - Settings
# ---------------------------------------------------------
st.sidebar.header("⚙️ Comparison Settings")

selected_models = st.sidebar.multiselect(
    "1. Select Models to compare:",
    options=df["Model ID"].tolist(),
    default=df["Model ID"].tolist(),
    format_func=lambda x: MODEL_NAMES.get(x, f"Model {x}")
)

selected_metrics = st.sidebar.multiselect(
    "2. Select Evaluation Metrics:",
    options=available_metrics,
    default=available_metrics
)

# ---------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------
if not selected_models:
    st.warning("⚠️ Please select at least one Model from the sidebar.")
    st.stop()

if not selected_metrics:
    st.warning("⚠️ Please select at least one Metric from the sidebar.")
    st.stop()

# Filter DataFrame by selected models
filtered_df = df[df["Model ID"].isin(selected_models)]

# Data table
st.subheader("📋 Detailed Comparison Table (Best values highlighted)")
display_df = filtered_df[["Model"] + selected_metrics].set_index("Model")
# Highlight best values with gradient coloring
cmap = "YlGn"
try:
    st.dataframe(display_df.style.background_gradient(cmap=cmap, axis=0))
except Exception:
    st.dataframe(display_df)

# Bar chart
st.subheader("📈 Visual Comparison Chart")
# Melt DataFrame for Plotly axis structure
melted_df = filtered_df.melt(id_vars=["Model"], value_vars=selected_metrics, var_name="Metric", value_name="Score")

# High-contrast, professional color palette for 6 Models
custom_colors = [
    "#E63946",  # Deep red
    "#F4A261",  # Orange
    "#E9C46A",  # Yellow
    "#2A9D8F",  # Teal green
    "#219EBC",  # Light blue
    "#023047"   # Dark navy
]

fig = px.bar(
    melted_df, 
    x="Metric", 
    y="Score", 
    color="Model", 
    barmode="group",        # Side-by-side bars for easy comparison
    text_auto=".4f",        # Show 4 decimal places
    color_discrete_sequence=custom_colors
)

fig.update_layout(
    xaxis_title="Evaluation Metric", 
    yaxis_title="Score", 
    legend_title="Models",
    xaxis={'categoryorder':'total descending'},
    plot_bgcolor="rgba(0,0,0,0)"  # Transparent background
)
st.plotly_chart(fig)
