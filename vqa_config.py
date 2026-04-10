"""
vqa_config.py - Central configuration for the VQA Seq2Seq Project
Supports SUBSET and FULL dataset modes for Kaggle and Local environments.
"""

import os

# ============================================================
# EXECUTION MODE
# ============================================================
# 'SUBSET': 25k samples (fast for testing)
# 'FULL': Original GQA dataset (best performance)
MODE = 'FULL' 

# ============================================================
# DATA PATHS
# ============================================================
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    BASE_DIR = "/kaggle/working"
    
    if MODE == 'FULL':
        # Paths for Full GQA Dataset on Kaggle
        IMAGES_INPUT = "/kaggle/input/datasets/lyte69/gqa-images"
        QUESTIONS_INPUT = "/kaggle/input/datasets/ammarmasselhy/gqa-questions"

        IMAGES_DIR = os.path.join(IMAGES_INPUT, "images")
        TRAIN_JSON = os.path.join(QUESTIONS_INPUT, "train_balanced_questions.json")
        VAL_JSON   = os.path.join(QUESTIONS_INPUT, "val_balanced_questions.json")
        TEST_JSON  = os.path.join(QUESTIONS_INPUT, "testdev_balanced_questions.json")
        VOCAB_PATH = os.path.join(BASE_DIR, "vocab_full_gqa.pkl")
        FEATURES_H5 = None # Full mode usually re-runs extraction or runs E2E
    else:
        # Paths for Subset dataset previously used
        SUBSET_IMAGES_DIR = "/kaggle/input/datasets/minhngcng3/gqa-images-subset/images_subset"
        SUBSET_META_DIR = "/kaggle/input/datasets/minhngcng3/gqa-vqa-subset"
        
        IMAGES_DIR = SUBSET_IMAGES_DIR
        TRAIN_JSON = os.path.join(SUBSET_META_DIR, "train_subset_25k.json")
        VAL_JSON   = os.path.join(SUBSET_META_DIR, "val_subset_5k.json")
        TEST_JSON  = os.path.join(SUBSET_META_DIR, "testdev_balanced_questions.json")
        VOCAB_PATH = os.path.join(SUBSET_META_DIR, "vocab.pkl")
        FEATURES_H5 = os.path.join(SUBSET_META_DIR, "resnet50_features.h5")
else:
    # Local Environment Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "gqa_data")
    
    if MODE == 'FULL':
        IMAGES_DIR = os.path.join(DATA_DIR, "images", "images_subset") # Use subset images for local testing
        TRAIN_JSON = os.path.join(DATA_DIR, "annotations", "train_subset_25k.json")
        VAL_JSON = os.path.join(DATA_DIR, "annotations", "val_subset_5k.json")
        TEST_JSON = os.path.join(DATA_DIR, "annotations", "testdev_balanced_questions.json")
        VOCAB_PATH = os.path.join(DATA_DIR, "vocab", "vocab_full_gqa.pkl")
        FEATURES_H5 = None
    else:
        IMAGES_DIR = os.path.join(DATA_DIR, "images", "images_subset")
        TRAIN_JSON = os.path.join(DATA_DIR, "annotations", "train_subset_25k.json")
        VAL_JSON = os.path.join(DATA_DIR, "annotations", "val_subset_5k.json")
        TEST_JSON = os.path.join(DATA_DIR, "annotations", "testdev_balanced_questions.json")
        VOCAB_PATH = os.path.join(DATA_DIR, "vocab", "vocab.pkl")
        FEATURES_H5 = os.path.join(DATA_DIR, "features", "resnet50_features.h5")

# ============================================================
# RESULTS PATHS
# ============================================================
# Default RESULTS_DIR based on MODE
RESULTS_DIR = os.path.join(BASE_DIR, "results" if MODE == 'FULL' else "results_subset")

# Sanity check: if preferred dir doesn't exist locally, fallback to the other one
if not IS_KAGGLE and not os.path.exists(RESULTS_DIR):
    alternative = os.path.join(BASE_DIR, "results_subset" if MODE == 'FULL' else "results")
    if os.path.exists(alternative):
        RESULTS_DIR = alternative

if IS_KAGGLE and not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, exist_ok=True)

# Model subdirectories mapping
SUBSET_MAP = {
    "model_1": "model_1_scratch_no_att",
    "model_2": "model_2_pretrained_no_att",
    "model_3": "model_3_scratch_att",
    "model_4": "model_4_pretrained_att",
    "model_5": "model_5_pretrained_e2e_no_att",
    "model_6": "model_6_pretrained_e2e_att",
    "model_7": "model_7_transformer"
}

def get_model_dir(model_id, mode=None):
    """
    Get the absolute path to a model's results directory.
    Handles SUBSET descriptive names and FULL simple names.
    """
    target_mode = mode if mode else MODE
    
    # Determine base results directory
    if target_mode == 'FULL':
        res_dir = os.path.join(BASE_DIR, "results")
    else:
        res_dir = os.path.join(BASE_DIR, "results_subset")
        
    # Check for descriptive name if in subset mode
    model_key = f"model_{model_id}"
    if target_mode == 'SUBSET' and model_key in SUBSET_MAP:
        descriptive_name = SUBSET_MAP[model_key]
        descriptive_dir = os.path.join(res_dir, descriptive_name)
        if os.path.exists(descriptive_dir):
            return descriptive_dir
            
    # Fallback to simple name (model_X)
    return os.path.join(res_dir, model_key)

# Initialize global MODEL_DIRS for backward compatibility
MODEL_DIRS = {f"model_{i}": get_model_dir(i) for i in range(1, 8)}
ATTENTION_MAPS_DIR = os.path.join(RESULTS_DIR, "attention_maps")

# ============================================================
# VOCABULARY PARAMETERS
# ============================================================
FREQ_THRESHOLD = 3 

# ============================================================
# IMAGE PARAMETERS
# ============================================================
SCRATCH_IMAGE_SIZE = 128
PRETRAINED_IMAGE_SIZE = 224

# ============================================================
# MODEL PARAMETERS
# ============================================================
EMBED_SIZE = 256
HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
DROPOUT = 0.3
RESNET_FEATURE_DIM = 2048
RESNET_SPATIAL_SIZE = 7

# CNN Scratch
SCRATCH_CNN_OUT = 512       # Number of CNN Scratch output channels

# ============================================================
# TRAINING PARAMETERS
# ============================================================
SCRATCH_LR = 5e-4
PRETRAINED_LR = 1e-4

SCRATCH_BATCH_SIZE = 128
PRETRAINED_BATCH_SIZE = 128
E2E_PRETRAINED_BATCH_SIZE = 128

SCRATCH_EPOCHS = 5
PRETRAINED_EPOCHS = 5

TEACHER_FORCING_RATIO = 1.0
TEACHER_FORCING_DECAY = 0.05
GRADIENT_CLIP = 5.0
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 4 if IS_KAGGLE else 2

# ============================================================
# EVALUATION PARAMETERS
# ============================================================
BEAM_SIZE = 3
MAX_ANSWER_LENGTH = 30
