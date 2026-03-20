"""
app.py - Giao diện Web App tối giản cho VQA Seq2Seq
Cho phép chọn giữa 6 Model.
Chạy lệnh: py -3.10 -m streamlit run d:\Deeplearning\app.py
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import os
import sys
import json
import random

# Thêm path để import project modules
sys.path.insert(0, os.path.dirname(__file__))

import config
from data.dataset import load_vocab, get_image_transform
from models.model_1 import VQAModel1_ScratchNoAtt
from models.model_2 import VQAModel2_PretrainedNoAtt
from models.model_3 import VQAModel3_ScratchAtt
from models.model_4 import VQAModel4_PretrainedAtt
from models.model_5 import VQAModel5_PretrainedEndToEndNoAtt
from models.model_6 import VQAModel6_PretrainedEndToEndAtt
from utils.logger import TrainingLogger

# ============================================================
# CẤU HÌNH GIAO DIỆN
# ============================================================
st.set_page_config(page_title="VQA Demo", layout="centered")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAMES = {
    1: "Model 1: Scratch + No Attention",
    2: "Model 2: Pretrained + No Attention",
    3: "Model 3: Scratch + Attention",
    4: "Model 4: Pretrained + Attention",
    5: "Model 5: Pretrained End-to-End + No Attention",
    6: "Model 6: Pretrained End-to-End + Attention"
}

# ============================================================
# TẢI MODEL & DATA
# ============================================================
@st.cache_resource
def load_base_resources():
    """Tải từ vựng, ResNet (dùng chung cho Model 2 & 4), và dữ liệu Val."""
    vocab = load_vocab(config.VOCAB_PATH)
    
    # ResNet-50 Feature Extractor
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(resnet.children())[:-2]).eval().to(DEVICE)
    avgpool = nn.AdaptiveAvgPool2d((1, 1)).eval().to(DEVICE)
    
    # Validation data cho Quick Inference
    with open(config.VAL_JSON, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    val_keys = list(val_data.keys())
    
    return vocab, backbone, avgpool, val_data, val_keys

@st.cache_resource
def load_vqa_model(model_id, vocab_size):
    """Tải trọng số cho model được chọn. Hàm này được cache theo model_id."""
    if model_id == 1:
        model = VQAModel1_ScratchNoAtt(vocab_size)
    elif model_id == 2:
        model = VQAModel2_PretrainedNoAtt(vocab_size)
    elif model_id == 3:
        model = VQAModel3_ScratchAtt(vocab_size)
    elif model_id == 4:
        model = VQAModel4_PretrainedAtt(vocab_size)
    elif model_id == 5:
        model = VQAModel5_PretrainedEndToEndNoAtt(vocab_size)
    elif model_id == 6:
        model = VQAModel6_PretrainedEndToEndAtt(vocab_size)
        
    model = model.to(DEVICE)
    logger = TrainingLogger(config.MODEL_DIRS[f"model_{model_id}"])
    epoch = logger.load_checkpoint(model, load_best=True)
    model.eval()
    
    return model, epoch

# ============================================================
# HÀM SUY LUẬN
# ============================================================
def infer(image, question, model_id, model, vocab, backbone, avgpool):
    """Hàm xử lý dự đoán tùy theo model_id."""
    # 1. Số hóa câu hỏi
    q_indices = [vocab.sos_idx] + vocab.numericalize(question) + [vocab.eos_idx]
    q_tensor = torch.tensor(q_indices).unsqueeze(0).to(DEVICE)
    
    # Lựa chọn xử lý ảnh tùy Model
    with torch.no_grad():
        if model_id in [1, 3, 5, 6]:
            # Scratch models & E2E Pretrained models: Đưa ảnh vào thẳng model
            image_size = config.SCRATCH_IMAGE_SIZE if model_id in [1, 3] else config.PRETRAINED_IMAGE_SIZE
            transform = get_image_transform(image_size)
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            generated = model.generate(
                img_tensor, q_tensor,
                vocab.sos_idx, vocab.eos_idx, max_len=config.MAX_ANSWER_LENGTH
            )
            
        else:
            # Pretrained models: Trích xuất qua ResNet (224x224) trước
            transform = get_image_transform(config.PRETRAINED_IMAGE_SIZE)
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            spatial = backbone(img_tensor)
            if model_id == 2:
                # No Attention: Lấy vector 2048-dim
                feature = avgpool(spatial).view(1, -1)
                generated = model.generate(
                    feature, q_tensor,
                    vocab.sos_idx, vocab.eos_idx, max_len=config.MAX_ANSWER_LENGTH
                )
            elif model_id == 4:
                # Attention: Lấy spatial feature (49, 2048)
                B, C, H, W = spatial.size()
                spatial_flat = spatial.view(B, C, H * W).permute(0, 2, 1) # (1, 49, 2048)
                
                # Hàm generate qua class Attention có trả về thêm (generated, alphas)
                generated = model.generate(
                    spatial_flat, q_tensor,
                    vocab.sos_idx, vocab.eos_idx, max_len=config.MAX_ANSWER_LENGTH
                )
    
    # 3. Code ngược lại thành chữ
    # Chú ý Model 3, 4, 6 trả về Tuple do có Attention
    if model_id in [3, 4, 6]:
        ans_tensor = generated[0] # Model 3, 4, 6 trả về (generated, alphas)
    else:
        ans_tensor = generated
        
    return vocab.decode(ans_tensor[0].cpu())

# ============================================================
# GIAO DIỆN CHÍNH
# ============================================================
def main():
    st.title("🤖 Ứng dụng VQA Đa Mô Hình")
    
    # Load Base Resources
    vocab, backbone, avgpool, val_data, val_keys = load_base_resources()
    
    # TÙY CHỌN MÔ HÌNH
    selected_model_name = st.selectbox(
        "🧠 Chọn mô hình để dự đoán:", 
        list(MODEL_NAMES.values()),
        index=1  # Mặc định là Model 2 (Pretrained)
    )
    # Lấy model_id từ string select (vd lấy số 2 từ "Model 2: ...")
    model_id = int(selected_model_name.split(":")[0][-1])
    
    vqa_model, trained_epoch = load_vqa_model(model_id, len(vocab))
    if trained_epoch == 0:
        st.warning(f"⚠️ {selected_model_name} dường như chưa được huấn luyện (không thấy checkpoint). Kết quả có thể không chính xác.")

    # KHUNG INFERENCE NGẪU NHIÊN
    st.markdown("---")
    if st.button("🎲 Lấy ngẫu nhiên 1 mẫu (Quick Inference)", type="secondary"):
        random_id = random.choice(val_keys)
        item = val_data[random_id]
        
        img_id = item["imageId"]
        true_question = item["question"]
        true_answer = item["fullAnswer"]
        
        img_path = os.path.join(config.IMAGES_DIR, f"{img_id}.jpg")
        
        if os.path.exists(img_path):
            input_image = Image.open(img_path).convert("RGB")
            st.image(input_image, caption=f"Ảnh ID: {img_id}", width=400)
            st.info(f"**Câu hỏi:** {true_question}")
            
            with st.spinner(f"Đang phân tích với {MODEL_NAMES[model_id]}..."):
                try:
                    pred_answer = infer(input_image, true_question, model_id, vqa_model, vocab, backbone, avgpool)
                    st.success(f"**🤖 Dự đoán:** {pred_answer}")
                    st.write(f"*(✅ Đáp án đúng: {true_answer})*")
                except Exception as e:
                    st.error(f"Lỗi: {e}")
        else:
            st.error("Không tải được ảnh từ máy.")

    st.markdown("---")
    st.subheader("Hoặc tự tải ảnh lên:")

    # 1. Tải ảnh
    uploaded_file = st.file_uploader("📥 Chọn ảnh (jpg, png)", type=["jpg", "png", "jpeg"])
    input_image = None
    if uploaded_file:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption="Ảnh đầu vào", width=400)

    # 2. Câu hỏi
    question = st.text_input("❓ Nhập câu hỏi (Tiếng Anh):", "What is in the picture?")

    # 3. Kết quả tự chọn
    if st.button("Hỏi Mô Hình", type="primary"):
        if input_image and question:
            with st.spinner(f"Đang phân tích với {MODEL_NAMES[model_id]}..."):
                try:
                    pred_answer = infer(input_image, question, model_id, vqa_model, vocab, backbone, avgpool)
                    st.success(f"**🤖 Câu trả lời ({MODEL_NAMES[model_id]}):** {pred_answer}")
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi: {e}")
        else:
            st.warning("Vui lòng nhập đủ hình ảnh và câu hỏi.")

if __name__ == "__main__":
    main()
