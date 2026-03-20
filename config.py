"""
config.py - Cấu hình tập trung cho toàn bộ dự án VQA Seq2Seq
Tất cả đường dẫn, tham số huấn luyện, và hằng số được quản lý tại đây.
"""

import os

# ============================================================
# ĐƯỜNG DẪN DỮ LIỆU (DATA PATHS)
# ============================================================
# Kiểm tra môi trường (Kaggle hay Local)
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    BASE_DIR = "/kaggle/working"
    # Dữ liệu subset (JSON + vocab) bạn upload lên
    META_DATA_DIR = "/kaggle/input/gqa-vqa-subset"
    # Dữ liệu ảnh gốc (từ bộ dataset bạn upload dạng zip, Kaggle tự giải nén)
    IMAGES_DIR = "/kaggle/input/gqa-images-subset"

    SUBSET_DIR = META_DATA_DIR
    TRAIN_JSON = os.path.join(SUBSET_DIR, "train_subset_25k.json")
    VAL_JSON = os.path.join(SUBSET_DIR, "val_subset_5k.json")
    VOCAB_PATH = os.path.join(SUBSET_DIR, "vocab.pkl")
    TEST_JSON = os.path.join(SUBSET_DIR, "testdev_balanced_questions.json")
    # File h5 sẽ được tạo bởi extract_features.py trên Kaggle (thư mục ghi được)
    FEATURES_H5 = os.path.join(BASE_DIR, "resnet50_features.h5")
else:
    BASE_DIR = r"d:\Deeplearning"
    DATA_DIR = os.path.join(BASE_DIR, "gqa_data")
    IMAGES_DIR = os.path.join(DATA_DIR, "images")
    SUBSET_DIR = os.path.join(DATA_DIR, "subset")
    TRAIN_JSON = os.path.join(SUBSET_DIR, "train_subset_25k.json")
    VAL_JSON = os.path.join(SUBSET_DIR, "val_subset_5k.json")
    TEST_JSON = os.path.join(DATA_DIR, "questions", "testdev_balanced_questions.json")
    VOCAB_PATH = os.path.join(SUBSET_DIR, "vocab.pkl")
    FEATURES_H5 = os.path.join(SUBSET_DIR, "resnet50_features.h5")

# ============================================================
# ĐƯỜNG DẪN KẾT QUẢ (RESULTS PATHS)
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
# THAM SỐ VOCABULARY
# ============================================================
FREQ_THRESHOLD = 3          # Từ phải xuất hiện >= 3 lần

# ============================================================
# THAM SỐ ẢNH (IMAGE PARAMETERS)
# ============================================================
SCRATCH_IMAGE_SIZE = 128    # Kích thước ảnh cho Model Scratch
PRETRAINED_IMAGE_SIZE = 224 # Kích thước ảnh cho Model Pretrained

# ============================================================
# THAM SỐ MÔ HÌNH (MODEL PARAMETERS)
# ============================================================
EMBED_SIZE = 256            # Kích thước embedding cho từ
HIDDEN_SIZE = 256           # Kích thước hidden state của LSTM
NUM_LSTM_LAYERS = 2         # Số lớp LSTM
DROPOUT = 0.3               # Tỷ lệ dropout

# CNN Scratch
SCRATCH_CNN_OUT = 512       # Số kênh output của CNN Scratch

# ResNet-50 Pretrained
RESNET_FEATURE_DIM = 2048   # Vector pooled: 2048-d
RESNET_SPATIAL_SIZE = 7     # Feature map: 7x7x2048

# ============================================================
# THAM SỐ HUẤN LUYỆN (TRAINING PARAMETERS)
# ============================================================
# Scratch models
SCRATCH_LR = 5e-4
SCRATCH_BATCH_SIZE = 32
SCRATCH_EPOCHS = 12

# Pretrained models (pre-extracted features)
PRETRAINED_LR = 1e-4
PRETRAINED_BATCH_SIZE = 32
PRETRAINED_EPOCHS = 12

# End-to-End Pretrained models (Model 5 & 6)
E2E_PRETRAINED_BATCH_SIZE = 32  # Nhỏ hơn vì ResNet-50 chạy forward pass mỗi batch

# Chung
TEACHER_FORCING_RATIO = 1.0     # Bắt đầu = 1.0
TEACHER_FORCING_DECAY = 0.05    # Giảm 0.05 mỗi epoch
GRADIENT_CLIP = 5.0             # Max norm cho gradient clipping
EARLY_STOPPING_PATIENCE = 5    # Dừng nếu Val Loss không giảm sau 5 epoch
NUM_WORKERS = 2                 # Workers cho DataLoader

# ============================================================
# THAM SỐ ĐÁNH GIÁ (EVALUATION PARAMETERS)
# ============================================================
BEAM_SIZE = 3                   # Beam search width
MAX_ANSWER_LENGTH = 30          # Độ dài tối đa câu trả lời khi sinh
