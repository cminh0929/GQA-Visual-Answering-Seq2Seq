"""
app.py - Minimalist Web App Interface for VQA Seq2Seq
Allows selecting between 6 Models.
Run command: py -3.10 -m streamlit run d:\Deeplearning\app.py
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
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Add path to import project modules
sys.path.insert(0, os.path.dirname(__file__))

import config
from data.dataset import load_vocab, get_image_transform
from models import get_model, get_model_info, list_models
from utils.logger import TrainingLogger

# ============================================================
# HELPERS
# ============================================================
def plot_attention_overlay(image, alphas, words):
    """Plot attention heatmaps for each word in the predicted answer."""
    num_words = len(words)
    if num_words == 0: return None
    
    # Calculate grid size
    cols = min(3, num_words)
    rows = (num_words + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if num_words == 1: axes = [axes]
    else: axes = axes.flatten()
    
    img_array = np.array(image.resize((224, 224)))
    
    for i in range(num_words):
        # alphas[i] shape: (1, 1, 49) or similar
        att = alphas[i].cpu().detach().numpy().squeeze()
        if att.ndim == 1:
            side = int(np.sqrt(len(att)))
            att = att.reshape(side, side)
            
        # Upsample attention
        att_resized = transforms.ToPILImage()(torch.from_numpy(att)).resize((224, 224), resample=Image.BILINEAR)
        
        axes[i].imshow(img_array)
        axes[i].imshow(np.array(att_resized), cmap='jet', alpha=0.5)
        axes[i].set_title(f"Target: {words[i]}")
        axes[i].axis('off')
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    return fig

# ============================================================
# DESIGN & CSS
# ============================================================
def set_design():
    st.markdown("""
        <style>
        .main {
            background-color: #ffffff;
            color: #111111;
        }
        .stApp {
            background-color: #ffffff;
        }
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
            border-right: 1px solid #ececec;
        }
        .stButton>button {
            border-radius: 4px;
            font-weight: 500;
        }
        .prediction-result {
            padding: 15px;
            border-left: 4px solid #007bff;
            background-color: #fcfcfc;
            border-top: 1px solid #eee;
            border-right: 1px solid #eee;
            border-bottom: 1px solid #eee;
            margin: 10px 0;
        }
        h1, h2, h3, p, span, label {
            color: #111111 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAMES = {mid: get_model_info(mid)["name"] for mid in list_models()}

# ============================================================
# LOAD MODEL & DATA
# ============================================================
@st.cache_resource
def load_base_resources():
    """Load vocabulary, ResNet (shared by Model 2 & 4), and Val data."""
    vocab = load_vocab(config.VOCAB_PATH)
    
    # ResNet-50 Feature Extractor
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(resnet.children())[:-2]).eval().to(DEVICE)
    avgpool = nn.AdaptiveAvgPool2d((1, 1)).eval().to(DEVICE)
    
    # Validation data for Quick Inference
    with open(config.VAL_JSON, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    val_keys = list(val_data.keys())
    
    return vocab, backbone, avgpool, val_data, val_keys

@st.cache_resource
def load_vqa_model(model_id, vocab_size):
    """Load weights for the selected model using factory."""
    model = get_model(model_id, vocab_size, device=DEVICE)
    
    logger = TrainingLogger(config.MODEL_DIRS[f"model_{model_id}"])
    epoch = logger.load_checkpoint(model, load_best=True)
    model.eval()
    
    return model, epoch

# ============================================================
# INFERENCE FUNCTION
# ============================================================
def infer(image, question, model_id, model, vocab, backbone, avgpool):
    """Prediction handling function according to model_id."""
    # 1. Numericalize question
    q_indices = [vocab.sos_idx] + vocab.numericalize(question) + [vocab.eos_idx]
    q_tensor = torch.tensor(q_indices).unsqueeze(0).to(DEVICE)
    
    alphas = None
    with torch.no_grad():
        if model_id in [1, 3, 5, 6, 7]:
            image_size = config.SCRATCH_IMAGE_SIZE if model_id in [1, 3] else config.PRETRAINED_IMAGE_SIZE
            transform = get_image_transform(image_size)
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            output = model.generate(
                img_tensor, q_tensor,
                vocab.sos_idx, vocab.eos_idx, max_len=config.MAX_ANSWER_LENGTH
            )
            
            if isinstance(output, tuple):
                generated, alphas = output
            else:
                generated = output
            
        else:
            # Pretrained models: Extract through ResNet (224x224) first
            transform = get_image_transform(config.PRETRAINED_IMAGE_SIZE)
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            spatial = backbone(img_tensor)
            if model_id == 2:
                features = avgpool(spatial).view(1, -1)
                generated = model.generate(features, q_tensor, vocab.sos_idx, vocab.eos_idx)
            else:
                # Model 4
                B, C, H, W = spatial.size()
                features = spatial.view(B, C, -1).permute(0, 2, 1)
                output = model.generate(features, q_tensor, vocab.sos_idx, vocab.eos_idx)
                if isinstance(output, tuple):
                    generated, alphas = output
                else:
                    generated = output

    # Decode
    pred_words = []
    for idx in generated[0]:
        word = vocab.itos.get(idx.item(), "<UNK>")
        if word == "<EOS>": break
        if word not in ["<SOS>", "<PAD>"]:
            pred_words.append(word)
            
    return " ".join(pred_words), alphas

# ============================================================
# MAIN UI
# ============================================================
def main():
    st.set_page_config(page_title="VQA Demo", layout="wide")
    set_design()
    
    vocab, backbone, avgpool, val_data, val_keys = load_base_resources()

    with st.sidebar:
        st.title("VQA LAB")
        selected_model_id = st.selectbox(
            "Active Model", list_models(),
            format_func=lambda x: MODEL_NAMES[x], index=1
        )
        st.info(get_model_info(selected_model_id)["description"])
        st.caption(f"Device: {DEVICE}")

    tab1, tab2 = st.tabs(["Single Model", "Benchmark (All Models)"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🎲 Get Random Sample", use_container_width=True):
                random_id = random.choice(val_keys)
                st.session_state.single_sample = val_data[random_id]
                st.session_state.single_output = None # Reset
            
            if 'single_sample' in st.session_state:
                s = st.session_state.single_sample
                img_path = os.path.join(config.IMAGES_DIR, f"{s['imageId']}.jpg")
                img = Image.open(img_path).convert("RGB")
                st.image(img, use_container_width=True)
                st.markdown(f"**Question:** {s['question']}")
                
                # Auto-run prediction
                vqa_model, _ = load_vqa_model(selected_model_id, len(vocab))
                pred_answer, alphas = infer(img, s['question'], selected_model_id, vqa_model, vocab, backbone, avgpool)
                
                with col2:
                    st.subheader("Result")
                    st.markdown(f"""
                    <div class="prediction-result">
                        <p style="margin:0; font-size: 0.8rem; color: #666;">PREDICTION</p>
                        <h2 style="margin:0; color: #000;">{pred_answer}</h2>
                        <p style="margin-top: 10px; font-size: 0.8rem; border-top: 1px solid #eee; padding-top: 5px;">
                            Truth: {s['fullAnswer']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if alphas is not None:
                        with st.expander("Show Attention Map"):
                            fig = plot_attention_overlay(img, alphas, pred_answer.split())
                            if fig: st.pyplot(fig)

        st.markdown("---")
        uploaded_file = st.file_uploader("Or Upload Image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            u_img = Image.open(uploaded_file).convert("RGB")
            u_q = st.text_input("Question:", "What is this?")
            if st.button("Predict Uploaded"):
                m, _ = load_vqa_model(selected_model_id, len(vocab))
                ans, _ = infer(u_img, u_q, selected_model_id, m, vocab, backbone, avgpool)
                st.write(f"**Answer:** {ans}")

    with tab2:
        if st.button("🎲 New Benchmark Sample", use_container_width=True, key="btn_benchmark"):
            rid = random.choice(val_keys)
            st.session_state.bench_sample = val_data[rid]
        
        if 'bench_sample' in st.session_state:
            s = st.session_state.bench_sample
            col_img, col_res = st.columns([1, 2])
            
            with col_img:
                img = Image.open(os.path.join(config.IMAGES_DIR, f"{s['imageId']}.jpg")).convert("RGB")
                st.image(img, use_container_width=True)
                st.markdown(f"**Q:** {s['question']}")
                st.caption(f"Truth: {s['fullAnswer']}")
            
            with col_res:
                st.subheader("Results (All Models)")
                # Table-like results
                for mid in list_models():
                    m, _ = load_vqa_model(mid, len(vocab))
                    ans, _ = infer(img, s['question'], mid, m, vocab, backbone, avgpool)
                    st.markdown(f"**M{mid}**: {ans}")
                    st.divider()

if __name__ == "__main__":
    main()
