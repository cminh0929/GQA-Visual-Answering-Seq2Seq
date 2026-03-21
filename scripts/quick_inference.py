"""
quick_inference.py - Test run model 2 on a few Val samples
"""
import torch
import json
import random
import os
import sys

# Add path to import project modules
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
        print("Model 2 is not trained or checkpoint not found!")
        return
    model.eval()

    # 4. Load Val Data
    with open(config.VAL_JSON, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    val_keys = list(val_data.keys())
    
    # 5. Dataset & Loader (to get features easily)
    dataset = GQAFeaturesDataset(config.VAL_JSON, config.FEATURES_H5, vocab, use_spatial=False)
    
    # Randomly select 5 samples
    indices = random.sample(range(len(dataset)), 5)
    
    print(f"\nRunning test on 5 random samples from Validation set:\n")

    for idx in indices:
        feature, question_idx, answer_idx = dataset[idx]
        feature = feature.unsqueeze(0).to(DEVICE)
        question_idx = question_idx.unsqueeze(0).to(DEVICE)

        # Sample info
        item_id = val_keys[idx]
        item = val_data[item_id]
        
        # Generate prediction
        with torch.no_grad():
            generated = model.generate(
                feature, question_idx, 
                vocab.sos_idx, vocab.eos_idx, 
                max_len=config.MAX_ANSWER_LENGTH
            )
        
        # Decode result
        pred_text = vocab.decode(generated[0].cpu())
        true_text = item["fullAnswer"]
        question_text = item["question"]
        image_id = item["imageId"]

        print(f"Image: {image_id}.jpg")
        print(f"Q:    {question_text}")
        print(f"Pred: {pred_text}")
        print(f"True: {true_text}")
        print("-" * 40)

if __name__ == "__main__":
    run()
