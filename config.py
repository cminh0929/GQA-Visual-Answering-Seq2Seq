"""
config.py - Central configuration for the VQA Seq2Seq Project
All paths, training parameters, and constants are managed here.
"""

import os

# ============================================================
# DATA PATHS
# ============================================================
# Check environment (Kaggle or Local)
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    BASE_DIR = "/kaggle/working"
    # Subset data (JSON + vocab) - Based on actual Notebook path
    META_DATA_DIR = "/kaggle/input/datasets/minhngcng3/gqa-vqa-subset"
    # Original images (from the uploaded 3.9GB dataset)
    IMAGES_DIR = "/kaggle/input/datasets/minhngcng3/gqa-images-subset/images_subset"

    SUBSET_DIR = META_DATA_DIR
    TRAIN_JSON = os.path.join(SUBSET_DIR, "train_subset_25k.json")
    VAL_JSON = os.path.join(SUBSET_DIR, "val_subset_5k.json")
    VOCAB_PATH = os.path.join(SUBSET_DIR, "vocab.pkl")
    TEST_JSON = os.path.join(SUBSET_DIR, "testdev_balanced_questions.json")
    # Successfully uploaded h5 file in INPUT (Permanent, no need to rerun)
    FEATURES_H5 = os.path.join(META_DATA_DIR, "resnet50_features.h5")
else:
    BASE_DIR = r"d:\Deeplearning"
    DATA_DIR = os.path.join(BASE_DIR, "gqa_data")
    # Kaggle tự động giải nén tạo thêm thư mục con images_subset
    IMAGES_DIR = os.path.join(DATA_DIR, "images", "images_subset")
    
    ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
    FEATURES_DIR = os.path.join(DATA_DIR, "features")
    VOCAB_DIR = os.path.join(DATA_DIR, "vocab")

    TRAIN_JSON = os.path.join(ANNOTATIONS_DIR, "train_subset_25k.json")
    VAL_JSON = os.path.join(ANNOTATIONS_DIR, "val_subset_5k.json")
    TEST_JSON = os.path.join(ANNOTATIONS_DIR, "testdev_balanced_questions.json")
    VOCAB_PATH = os.path.join(VOCAB_DIR, "vocab.pkl")
    FEATURES_H5 = os.path.join(FEATURES_DIR, "resnet50_features.h5")

# ============================================================
# RESULTS PATHS
# ============================================================
RESULTS_DIR = os.path.join(BASE_DIR, "results")
if IS_KAGGLE and not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_DIRS = {
    "model_1": os.path.join(RESULTS_DIR, "model_1_scratch_no_att"),
    "model_2": os.path.join(RESULTS_DIR, "model_2_pretrained_no_att"),
    "model_3": os.path.join(RESULTS_DIR, "model_3_scratch_att"),
    "model_4": os.path.join(RESULTS_DIR, "model_4_pretrained_att"),
    "model_5": os.path.join(RESULTS_DIR, "model_5_pretrained_e2e_no_att"),
    "model_6": os.path.join(RESULTS_DIR, "model_6_pretrained_e2e_att"),
}

ATTENTION_MAPS_DIR = os.path.join(RESULTS_DIR, "attention_maps")

# ============================================================
# VOCABULARY PARAMETERS
# ============================================================
FREQ_THRESHOLD = 3          # Word must appear >= 3 times

# ============================================================
# IMAGE PARAMETERS
# ============================================================
SCRATCH_IMAGE_SIZE = 128    # Image size for Scratch Model
PRETRAINED_IMAGE_SIZE = 224 # Image size for Pretrained Model

# ============================================================
# MODEL PARAMETERS
# ============================================================
EMBED_SIZE = 256            # Word embedding size
HIDDEN_SIZE = 256           # LSTM hidden state size
NUM_LSTM_LAYERS = 2         # Number of LSTM layers
DROPOUT = 0.3               # Dropout rate

# CNN Scratch
SCRATCH_CNN_OUT = 512       # Number of CNN Scratch output channels

# ResNet-50 Pretrained
RESNET_FEATURE_DIM = 2048   # Pooled vector: 2048-d
RESNET_SPATIAL_SIZE = 7     # Feature map: 7x7x2048

# ============================================================
# TRAINING PARAMETERS
# ============================================================
# Scratch models
SCRATCH_LR = 5e-4
SCRATCH_BATCH_SIZE = 128
SCRATCH_EPOCHS = 12

# Pretrained models (pre-extracted features)
PRETRAINED_LR = 1e-4
PRETRAINED_BATCH_SIZE = 128
PRETRAINED_EPOCHS = 12

# End-to-End Pretrained models (Model 5 & 6)
E2E_PRETRAINED_BATCH_SIZE = 64  # Smaller because ResNet-50 runs forward pass each batch

# Common
TEACHER_FORCING_RATIO = 1.0     # Starts at 1.0
TEACHER_FORCING_DECAY = 0.05    # Decrease by 0.05 per epoch
GRADIENT_CLIP = 5.0             # Max norm for gradient clipping
EARLY_STOPPING_PATIENCE = 5    # Stop if Val Loss does not decrease after 5 epochs
NUM_WORKERS = 2                 # Workers for DataLoader

# ============================================================
# EVALUATION PARAMETERS
# ============================================================
BEAM_SIZE = 3                   # Beam search width
MAX_ANSWER_LENGTH = 30          # Maximum generated answer length

