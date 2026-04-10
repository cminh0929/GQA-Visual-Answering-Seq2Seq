r"""
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

import vqa_config as config
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
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
        
        html, body, [data-testid="stSidebar"] {
            font-family: 'Outfit', sans-serif;
        }

        .main {
            background-color: #fdfdfd;
            color: #111111;
        }
        .stApp {
            background-color: #fdfdfd;
        }
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #f0f0f0;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            background-color: #000000;
            color: #ffffff;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #333333;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .prediction-result {
            padding: 25px;
            border-radius: 12px;
            border-left: 6px solid #000000;
            background-color: #ffffff;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            margin: 20px 0;
        }
        h1, h2, h3 {
            font-weight: 600 !important;
            letter-spacing: -0.5px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            font-weight: 600;
            font-size: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAMES = {mid: get_model_info(mid)["name"] for mid in list_models()}

# ============================================================
# LOAD MODEL & DATA
# ============================================================
@st.cache_resource
def load_base_resources(selected_mode):
    """Load vocabulary, ResNet, and Val data based on selected mode."""
    # Determine paths locally based on selection
    data_dir = os.path.join(os.path.dirname(__file__), "gqa_data")
    
    if selected_mode == 'FULL':
        vocab_path = os.path.join(data_dir, "vocab", "vocab_full_gqa.pkl")
        val_json = os.path.join(data_dir, "annotations", "val_subset_5k.json") # Fallback
        images_dir = os.path.join(data_dir, "images", "images_subset")
    else:
        vocab_path = os.path.join(data_dir, "vocab", "vocab.pkl")
        val_json = os.path.join(data_dir, "annotations", "val_subset_5k.json")
        images_dir = os.path.join(data_dir, "images", "images_subset")

    vocab = load_vocab(vocab_path)
    
    # ResNet-50 Feature Extractor
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(resnet.children())[:-2]).eval().to(DEVICE)
    avgpool = nn.AdaptiveAvgPool2d((1, 1)).eval().to(DEVICE)
    
    # Validation data
    with open(val_json, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    val_keys = list(val_data.keys())
    
    return vocab, backbone, avgpool, val_data, val_keys, images_dir

@st.cache_resource
def load_vqa_model(model_id, vocab_size, selected_mode):
    """Load weights for the selected model from the correct results folder."""
    model = get_model(model_id, vocab_size, device=DEVICE)
    
    # Use centralized resolution for model directory
    model_dir = config.get_model_dir(model_id, mode=selected_mode)
    
    logger = TrainingLogger(model_dir)
    epoch = logger.load_checkpoint(model, load_best=True)
    model.eval()
    
    return model, epoch

def check_model_ready(model_id, selected_mode):
    """Check if model checkpoint exists for a specific mode."""
    model_dir = config.get_model_dir(model_id, mode=selected_mode)
    checkpoint_path = os.path.join(model_dir, "checkpoints", "best_model.pth")
    return os.path.exists(checkpoint_path)

def load_metrics(model_id, selected_mode):
    """Load evaluation metrics if available."""
    model_dir = config.get_model_dir(model_id, mode=selected_mode)
    metrics_path = os.path.join(model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

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
    st.set_page_config(page_title="VQA Studio", layout="wide", page_icon="🚀")
    set_design()
    
    with st.sidebar:
        st.title("🚀 VQA STUDIO")
        
        # 1. Dataset Mode Selection
        st.subheader("⚙️ System Configuration")
        selected_mode = st.radio(
            "Target Dataset Mode",
            ["FULL", "SUBSET"],
            index=0 if config.MODE == 'FULL' else 1,
            help="Full: Use results/ folder | Subset: Use results_subset/ folder"
        )
        
        # Load resources based on selection
        vocab, backbone, avgpool, val_data, val_keys, current_images_dir = load_base_resources(selected_mode)

        st.divider()
        
        # 2. Model Selection with Readiness Check
        st.subheader("🤖 Model Selection")
        
        # Function to format model names with status
        def format_model_options(mid):
            is_ready = check_model_ready(mid, selected_mode)
            status = "✅" if is_ready else "❌"
            return f"{status} {MODEL_NAMES[mid]}"

        selected_model_id = st.selectbox(
            "Active Model", list_models(),
            format_func=format_model_options,
            index=0 
        )
        
        is_ready = check_model_ready(selected_model_id, selected_mode)
        if not is_ready:
            st.error("Model file not found in current mode!")
            st.warning("Please switch mode or train the model.")
        
        st.info(get_model_info(selected_model_id)["description"])
        st.caption(f"Device: {DEVICE} | Vocab Size: {len(vocab)}")

    tab1, tab2 = st.tabs(["Single Inference", "Model Benchmark"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🎲 Get Random Sample", width="stretch"):
                random_id = random.choice(val_keys)
                st.session_state.single_sample = val_data[random_id]
                st.session_state.single_output = None # Reset
            
            if 'single_sample' in st.session_state:
                s = st.session_state.single_sample
                img_path = os.path.join(current_images_dir, f"{s['imageId']}.jpg")
                img = Image.open(img_path).convert("RGB")
                st.image(img, width="stretch")
                st.markdown(f"**Question:** {s['question']}")
                
                # Inference Logic
                if is_ready:
                    with st.spinner("🧠 Computation in progress..."):
                        vqa_model, _ = load_vqa_model(selected_model_id, len(vocab), selected_mode)
                        pred_answer, alphas = infer(img, s['question'], selected_model_id, vqa_model, vocab, backbone, avgpool)
                    
                    # Metrics Section
                    metrics = load_metrics(selected_model_id, selected_mode)
                    if metrics:
                        st.subheader("📊 Performance Metrics")
                        m_col1, m_col2, m_col3 = st.columns(3)
                        with m_col1:
                            st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
                        with m_col2:
                            st.metric("BLEU-4", f"{metrics.get('bleu_4', 0):.4f}")
                        with m_col3:
                            st.metric("Cider", f"{metrics.get('cider', 0):.4f}")

                    with col2:
                        st.subheader("🎯 Prediction Result")
                        st.markdown(f"""
                        <div class="prediction-result">
                            <p style="margin:0; font-size: 0.8rem; color: #666;">PREDICTED ANSWER</p>
                            <h2 style="margin:0; color: #000;">{pred_answer}</h2>
                            <p style="margin-top: 10px; font-size: 0.8rem; border-top: 1px solid #eee; padding-top: 5px;">
                                Ground Truth: {s['fullAnswer']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if alphas is not None:
                            with st.expander("Show Attention Heatmap"):
                                fig = plot_attention_overlay(img, alphas, pred_answer.split())
                                if fig: st.pyplot(fig)
                else:
                    st.error("This model cannot answer because the weights (.pth) are missing for this mode.")

        st.markdown("---")
        uploaded_file = st.file_uploader("Or Upload Image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            u_img = Image.open(uploaded_file).convert("RGB")
            u_q = st.text_input("Question:", "What is this?")
            if st.button("Predict Uploaded"):
                m, _ = load_vqa_model(selected_model_id, len(vocab), selected_mode)
                ans, _ = infer(u_img, u_q, selected_model_id, m, vocab, backbone, avgpool)
                st.write(f"**Answer:** {ans}")

    with tab2:
        if st.button("🎲 New Benchmark Sample", width="stretch", key="btn_benchmark"):
            rid = random.choice(val_keys)
            st.session_state.bench_sample = val_data[rid]
        
        if 'bench_sample' in st.session_state:
            s = st.session_state.bench_sample
            col_img, col_res = st.columns([1, 2])
            
            with col_img:
                img = Image.open(os.path.join(config.IMAGES_DIR, f"{s['imageId']}.jpg")).convert("RGB")
                st.image(img, width="stretch")
                st.markdown(f"**Q:** {s['question']}")
                st.caption(f"Truth: {s['fullAnswer']}")
            
            with col_res:
                st.subheader("Results (All Models)")
                # Table-like results
                for mid in list_models():
                    ready = check_model_ready(mid, selected_mode)
                    if ready:
                        m, _ = load_vqa_model(mid, len(vocab), selected_mode)
                        ans, _ = infer(img, s['question'], mid, m, vocab, backbone, avgpool)
                        st.markdown(f"**M{mid}**: {ans}")
                    else:
                        st.markdown(f"**M{mid}**: ❌ *Missing weights*")
                    st.divider()

if __name__ == "__main__":
    main()
