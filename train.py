"""
train.py - Script huấn luyện chính cho 6 mô hình VQA Seq2Seq

Usage:
    py -3.10 train.py --model 1    (Scratch + No Attention, End-to-End)
    py -3.10 train.py --model 2    (Pretrained + No Attention, Pre-extracted)
    py -3.10 train.py --model 3    (Scratch + Attention, End-to-End)
    py -3.10 train.py --model 4    (Pretrained + Attention, Pre-extracted)
    py -3.10 train.py --model 5    (Pretrained + No Attention, End-to-End)
    py -3.10 train.py --model 6    (Pretrained + Attention, End-to-End)
    py -3.10 train.py --model 2 --resume   (Tiếp tục train từ checkpoint)
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn

import config
from data.dataset import (
    GQADataset, GQAFeaturesDataset, load_vocab, get_dataloader
)
from models.model_1 import VQAModel1_ScratchNoAtt
from models.model_2 import VQAModel2_PretrainedNoAtt
from models.model_3 import VQAModel3_ScratchAtt
from models.model_4 import VQAModel4_PretrainedAtt
from models.model_5 import VQAModel5_PretrainedEndToEndNoAtt
from models.model_6 import VQAModel6_PretrainedEndToEndAtt
from utils.logger import TrainingLogger, EarlyStopping


# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_model_config(model_id):
    """Trả về cấu hình phù hợp cho từng model."""
    is_pretrained = model_id in [2, 4]       # Pre-extracted features
    is_end2end_pretrained = model_id in [5, 6]  # End-to-end pretrained
    has_attention = model_id in [3, 4, 6]
    uses_pretrained_cfg = is_pretrained or is_end2end_pretrained

    return {
        "model_id": model_id,
        "is_pretrained": is_pretrained,
        "is_end2end_pretrained": is_end2end_pretrained,
        "has_attention": has_attention,
        "lr": config.PRETRAINED_LR if uses_pretrained_cfg else config.SCRATCH_LR,
        "batch_size": config.E2E_PRETRAINED_BATCH_SIZE if is_end2end_pretrained else (
            config.PRETRAINED_BATCH_SIZE if is_pretrained else config.SCRATCH_BATCH_SIZE
        ),
        "epochs": config.PRETRAINED_EPOCHS if uses_pretrained_cfg else config.SCRATCH_EPOCHS,
        "image_size": config.PRETRAINED_IMAGE_SIZE if uses_pretrained_cfg else config.SCRATCH_IMAGE_SIZE,
        "model_dir": config.MODEL_DIRS[f"model_{model_id}"],
    }


def build_model(model_id, vocab_size):
    """Tạo model dựa trên ID."""
    models_map = {
        1: VQAModel1_ScratchNoAtt,
        2: VQAModel2_PretrainedNoAtt,
        3: VQAModel3_ScratchAtt,
        4: VQAModel4_PretrainedAtt,
        5: VQAModel5_PretrainedEndToEndNoAtt,
        6: VQAModel6_PretrainedEndToEndAtt,
    }
    model = models_map[model_id](vocab_size)
    return model.to(DEVICE)


def build_dataloaders(model_cfg, vocab):
    """Tạo DataLoader cho train và val."""
    if model_cfg["is_pretrained"]:
        # Model 2 & 4: Dùng pre-extracted features
        use_spatial = model_cfg["has_attention"]
        train_dataset = GQAFeaturesDataset(
            config.TRAIN_JSON, config.FEATURES_H5, vocab,
            use_spatial=use_spatial
        )
        val_dataset = GQAFeaturesDataset(
            config.VAL_JSON, config.FEATURES_H5, vocab,
            use_spatial=use_spatial
        )
    else:
        # Model 1, 3, 5, 6: Dùng raw images (End-to-End)
        train_dataset = GQADataset(
            config.TRAIN_JSON, config.IMAGES_DIR, vocab,
            image_size=model_cfg["image_size"]
        )
        val_dataset = GQADataset(
            config.VAL_JSON, config.IMAGES_DIR, vocab,
            image_size=model_cfg["image_size"]
        )

    train_loader = get_dataloader(
        train_dataset, model_cfg["batch_size"],
        shuffle=True, num_workers=config.NUM_WORKERS,
        pad_idx=vocab.pad_idx
    )
    val_loader = get_dataloader(
        val_dataset, model_cfg["batch_size"],
        shuffle=False, num_workers=config.NUM_WORKERS,
        pad_idx=vocab.pad_idx
    )

    return train_loader, val_loader


# ============================================================
# TRAIN & VALIDATE
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, tf_ratio,
                    has_attention, device):
    """Huấn luyện 1 epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (inputs, questions, answers) in enumerate(loader):
        inputs = inputs.to(device)
        questions = questions.to(device)
        answers = answers.to(device)

        optimizer.zero_grad()

        # Forward
        if has_attention:
            outputs, _ = model(inputs, questions, answers, tf_ratio)
        else:
            outputs = model(inputs, questions, answers, tf_ratio)

        # Loss: bỏ token đầu tiên (<SOS>) của target
        # outputs: (B, seq_len, vocab_size), answers: (B, seq_len)
        output_dim = outputs.shape[-1]
        outputs = outputs[:, 1:, :].contiguous().view(-1, output_dim)
        targets = answers[:, 1:].contiguous().view(-1)

        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # In tiến độ mỗi 100 batch
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / num_batches
            print(f"    Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {avg_loss:.4f}")

    return total_loss / max(num_batches, 1)


def validate(model, loader, criterion, vocab, has_attention, device):
    """Đánh giá trên tập validation."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for inputs, questions, answers in loader:
            inputs = inputs.to(device)
            questions = questions.to(device)
            answers = answers.to(device)

            # Forward (teacher forcing = 0 khi validate)
            if has_attention:
                outputs, _ = model(inputs, questions, answers, 0.0)
            else:
                outputs = model(inputs, questions, answers, 0.0)

            # Loss
            output_dim = outputs.shape[-1]
            outputs_flat = outputs[:, 1:, :].contiguous().view(-1, output_dim)
            targets_flat = answers[:, 1:].contiguous().view(-1)
            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item()
            num_batches += 1

            # Accuracy: so sánh từ dự đoán với từ đúng (bỏ qua PAD)
            predictions = outputs[:, 1:, :].argmax(dim=-1)  # (B, seq_len-1)
            targets = answers[:, 1:]  # (B, seq_len-1)

            mask = targets != vocab.pad_idx
            correct += ((predictions == targets) & mask).sum().item()
            total += mask.sum().item()

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train VQA Seq2Seq Model")
    parser.add_argument("--model", type=int, required=True, choices=[1, 2, 3, 4, 5, 6],
                        help="Model ID (1-6)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    args = parser.parse_args()

    # Cấu hình
    model_cfg = get_model_config(args.model)
    model_names = {
        1: "Scratch + No Attention (End-to-End)",
        2: "Pretrained + No Attention (Pre-extracted)",
        3: "Scratch + Attention (End-to-End)",
        4: "Pretrained + Attention (Pre-extracted)",
        5: "Pretrained + No Attention (End-to-End)",
        6: "Pretrained + Attention (End-to-End)",
    }

    print("=" * 60)
    print(f"TRAINING MODEL {args.model}: {model_names[args.model]}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # 1. Load vocabulary
    print("\n[1/4] Loading vocabulary...")
    vocab = load_vocab(config.VOCAB_PATH)

    # 2. Build model
    print(f"\n[2/4] Building Model {args.model}...")
    model = build_model(args.model, len(vocab))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # 3. Build dataloaders
    print(f"\n[3/4] Building DataLoaders...")
    print(f"  Batch size: {model_cfg['batch_size']}")
    train_loader, val_loader = build_dataloaders(model_cfg, vocab)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # 4. Setup training
    print(f"\n[4/4] Setting up training...")
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["lr"])
    logger = TrainingLogger(model_cfg["model_dir"])
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

    start_epoch = 0
    if args.resume:
        start_epoch = logger.load_checkpoint(model, optimizer, load_best=False)

    # Training loop
    print(f"\n{'='*60}")
    print(f"START TRAINING")
    print(f"  Epochs: {model_cfg['epochs']}")
    print(f"  LR: {model_cfg['lr']}")
    print(f"  Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch + 1, model_cfg["epochs"] + 1):
        epoch_start = time.time()

        # Teacher forcing ratio (giảm dần)
        tf_ratio = max(
            0.0,
            config.TEACHER_FORCING_RATIO - config.TEACHER_FORCING_DECAY * (epoch - 1)
        )

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            tf_ratio, model_cfg["has_attention"], DEVICE
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, vocab,
            model_cfg["has_attention"], DEVICE
        )

        epoch_time = time.time() - epoch_start

        # Log
        is_best = logger.log_epoch(
            epoch, train_loss, val_loss, val_acc,
            model_cfg["lr"], tf_ratio,
            extra_metrics={"epoch_time": epoch_time}
        )

        # Save checkpoint
        logger.save_checkpoint(model, optimizer, epoch, is_best)
        logger.save_history()

        # Early stopping
        if early_stopping.should_stop(val_loss):
            break

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"  Best epoch: {logger.best_epoch}")
    print(f"  Best val_loss: {logger.best_val_loss:.4f}")
    print(f"  Results saved to: {model_cfg['model_dir']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
