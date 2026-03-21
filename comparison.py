import streamlit as st
import json
import os
import sys
import pandas as pd
import plotly.express as px

# Thêm path để có thể import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import config
except ImportError:
    st.error("Không tìm thấy file config.py")
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
st.markdown("Dashboard được dùng để so sánh các tiêu chí hiệu suất giữa các Model.")

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
    st.error("⚠️ Không tìm thấy file `metrics.json` trong bất kỳ thư mục model nào. Bạn cần chạy `evaluate.py` trước tiên.")
    st.stop()

# ---------------------------------------------------------
# Chuẩn bị DataFrame
# ---------------------------------------------------------
rows = []
for model_id, metrics in data.items():
    row = {"Model ID": str(model_id), "Model": MODEL_NAMES.get(str(model_id), f"Model {model_id}")}
    for k, v in metrics.items():
        # Lọc bỏ "accuracy", "inference_time", "num_samples"
        if k not in ["accuracy", "inference_time", "num_samples"]:
            row[k] = v
    rows.append(row)

df = pd.DataFrame(rows)
available_metrics = [c for c in df.columns if c not in ["Model ID", "Model"]]

# ---------------------------------------------------------
# SIDEBAR - Thiết lập
# ---------------------------------------------------------
st.sidebar.header("⚙️ Cấu Hình So Sánh")

selected_models = st.sidebar.multiselect(
    "1. Chọn các Model muốn so sánh:",
    options=df["Model ID"].tolist(),
    default=df["Model ID"].tolist(),
    format_func=lambda x: MODEL_NAMES.get(x, f"Model {x}")
)

selected_metrics = st.sidebar.multiselect(
    "2. Chọn các Tiêu chí đánh giá (Metrics):",
    options=available_metrics,
    default=available_metrics
)

# ---------------------------------------------------------
# HIỂN THỊ MAIN CONTENT
# ---------------------------------------------------------
if not selected_models:
    st.warning("⚠️ Vui lòng chọn ít nhất một Model bên cột trái.")
    st.stop()

if not selected_metrics:
    st.warning("⚠️ Vui lòng chọn ít nhất một Tiêu chí đánh giá bên cột trái.")
    st.stop()

# Lọc DataFrame theo model đã chọn
filtered_df = df[df["Model ID"].isin(selected_models)]

# Bảng dữ liệu
st.subheader("📋 Bảng So Sánh Chi Tiết (Bôi Xanh Cột Cao Nhất)")
display_df = filtered_df[["Model"] + selected_metrics].set_index("Model")
# Bôi sáng các giá trị lớn nhất theo từng cột để dễ nhìn (đỏ nhạt tới xanh đậm)
cmap = "YlGn"
try:
    # Streamlit dataframe có thể quăng warning use_container_width tuỳ phiên bản, ta dùng mặc định
    st.dataframe(display_df.style.background_gradient(cmap=cmap, axis=0))
except Exception:
    st.dataframe(display_df)

# Biểu đồ cột
st.subheader("📈 Trực Quan Hoá Bằng Biểu Đồ")
# Melt DataFrame để phù hợp cấu trúc trục của Plotly
melted_df = filtered_df.melt(id_vars=["Model"], value_vars=selected_metrics, var_name="Metric", value_name="Score")

# Sử dụng màu sắc tương phản cao, chuyên nghiệp cho 6 Model
custom_colors = [
    "#E63946", # Đỏ trầm
    "#F4A261", # Cam
    "#E9C46A", # Vàng
    "#2A9D8F", # Xanh lá
    "#219EBC", # Xanh biển sáng
    "#023047"  # Xanh dương đậm
]

fig = px.bar(
    melted_df, 
    x="Metric", 
    y="Score", 
    color="Model", 
    barmode="group",        # Các cột đứng cạnh nhau để dễ xem
    text_auto=".4f",        # Hiển thị 4 chữ số thập phân
    color_discrete_sequence=custom_colors
)

fig.update_layout(
    xaxis_title="Tiêu chí đánh giá", 
    yaxis_title="Điểm số", 
    legend_title="Danh sách Models",
    xaxis={'categoryorder':'total descending'},
    plot_bgcolor="rgba(0,0,0,0)" # Nền trong suốt
)
st.plotly_chart(fig)
