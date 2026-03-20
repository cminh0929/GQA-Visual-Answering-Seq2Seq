"""
quick_inference.py - Chạy thử mô hình 2 trên vài mẫu tập Val
"""
import torch
import json
import random
import os
import sys

# Thêm path để import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import config
from data.dataset import load_vocab, GQAFeaturesDataset, get_dataloader
from models.model_2 import VQAModel2_PretrainedNoAtt
from utils.logger import TrainingLogger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run():
    print("="*60)
    print("DEMO INFERENCE - MODEL 2 (PRETRAINED NO ATT)")
    print("="*60)

    # 1. Load Vocab
    vocab = load_vocab(config.VOCAB_PATH)

    # 2. Build Model
    model = VQAModel2_PretrainedNoAtt(len(vocab)).to(DEVICE)
    
    # 3. Load Checkpoint
    logger = TrainingLogger(config.MODEL_DIRS["model_2"])
    epoch = logger.load_checkpoint(model, load_best=True)
    if epoch == 0:
        print("Model 2 chưa được huấn luyện hoặc không tìm thấy checkpoint!")
        return
    model.eval()

    # 4. Load Val Data
    with open(config.VAL_JSON, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    val_keys = list(val_data.keys())
    
    # 5. Dataset & Loader (để lấy features dễ dàng)
    dataset = GQAFeaturesDataset(config.VAL_JSON, config.FEATURES_H5, vocab, use_spatial=False)
    
    # Chọn ngẫu nhiên 5 mẫu
    indices = random.sample(range(len(dataset)), 5)
    
    print(f"\nChạy thử 5 mẫu ngẫu nhiên từ tập Validation:\n")

    for idx in indices:
        feature, question_idx, answer_idx = dataset[idx]
        feature = feature.unsqueeze(0).to(DEVICE)
        question_idx = question_idx.unsqueeze(0).to(DEVICE)

        # Thông tin mẫu
        item_id = val_keys[idx]
        item = val_data[item_id]
        
        # Generate dự đoán
        with torch.no_grad():
            generated = model.generate(
                feature, question_idx, 
                vocab.sos_idx, vocab.eos_idx, 
                max_len=config.MAX_ANSWER_LENGTH
            )
        
        # Giải mã kết quả
        pred_text = vocab.decode(generated[0].cpu())
        true_text = item["fullAnswer"]
        question_text = item["question"]
        image_id = item["imageId"]

        print(f"Ảnh: {image_id}.jpg")
        print(f"Q:    {question_text}")
        print(f"Pred: {pred_text}")
        print(f"True: {true_text}")
        print("-" * 40)

if __name__ == "__main__":
    run()
