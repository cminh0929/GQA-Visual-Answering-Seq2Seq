"""
train.py - Main training script for 6 VQA Seq2Seq models

Usage:
    py -3.10 train.py --model 1    (Scratch + No Attention, End-to-End)
    py -3.10 train.py --model 2    (Pretrained + No Attention, Pre-extracted)
    py -3.10 train.py --model 3    (Scratch + Attention, End-to-End)
    py -3.10 train.py --model 4    (Pretrained + Attention, Pre-extracted)
    py -3.10 train.py --model 5    (Pretrained + No Attention, End-to-End)
    py -3.10 train.py --model 6    (Pretrained + Attention, End-to-End)
    py -3.10 train.py --model 2 --resume   (Resume training from checkpoint)
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn

import vqa_config as config
from data.dataset import (
    GQADataset, GQAFeaturesDataset, load_vocab, get_dataloader
)
from models import get_model, get_model_info
from utils.logger import TrainingLogger, EarlyStopping


# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_model_config(model_id):
    """Return appropriate config for each model."""
    info = get_model_info(model_id)
    is_pretrained = info["strategy"] == "pre-extracted"
    is_end2end_pretrained = model_id in [5, 6]  # Specific E2E variants
    has_attention = info["has_attention"]
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
    """Create model using factory."""
    return get_model(model_id, vocab_size, device=DEVICE)


def build_dataloaders(model_cfg, vocab):
    """Create DataLoaders for train and val."""
    if model_cfg["is_pretrained"]:
        # Model 2 & 4: Use pre-extracted features
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
        # Model 1, 3, 5, 6: Use raw images (End-to-End)
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
    """Train 1 epoch."""
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

        # Loss: discard the first token (<SOS>) of the target
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

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / num_batches
            print(f"    Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {avg_loss:.4f}")

    return total_loss / max(num_batches, 1)


def validate(model, loader, criterion, vocab, has_attention, device):
    """Evaluate on the validation set."""
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

            # Forward (teacher forcing = 0 when validating)
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

            # Accuracy: compare predicted words with target words (ignore PAD)
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
    parser.add_argument("--model", type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7],
                        help="Model ID (1-7)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    args = parser.parse_args()

    # Configuration
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
    
    # Enable Multi-GPU support (e.g., Kaggle T4 x2)
    if torch.cuda.device_count() > 1:
        print(f"  ★ Multi-GPU detected: Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # 3. Build dataloaders
    print(f"\n[3/4] Building DataLoaders...")
    print(f"  Batch size: {model_cfg['batch_size']}")
    print(f"  Num workers: {config.NUM_WORKERS}")
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

        # Teacher forcing ratio (decaying)
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
