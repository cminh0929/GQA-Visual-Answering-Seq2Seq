"""
visualize.py - Visualize training results and compare models
Step 6 in VQA_Seq2Seq_Project_Plan.md

Features:
1. Plot Learning Curves (Train/Val Loss) for each model
2. Plot metrics comparison table between models
3. Show prediction samples: Image + Question + Answer
4. Visualize Attention Heatmap (for Model 3 & 4)

Usage:
    py -3.10 visualize.py --curves           (Learning curves)
    py -3.10 visualize.py --compare          (Metrics comparison)
    py -3.10 visualize.py --samples 4        (Prediction samples Model 4)
    py -3.10 visualize.py --attention 4      (Attention maps Model 4)
    py -3.10 visualize.py --all              (All visualisations)
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import config


# ============================================================
# STYLE
# ============================================================
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    1: "#e74c3c",  # Red - Scratch No Att
    2: "#3498db",  # Blue - Pretrained No Att
    3: "#e67e22",  # Orange - Scratch Att
    4: "#2ecc71",  # Green - Pretrained Att
    5: "#9b59b6",  # Purple - Pretrained E2E
    6: "#1abc9c",  # Cyan - Pretrained E2E Att
}
MODEL_LABELS = {
    1: "M1: Scratch",
    2: "M2: Pretrained",
    3: "M3: Scratch+Att",
    4: "M4: Pretrained+Att",
    5: "M5: E2E Pretrained",
    6: "M6: E2E + Att",
}


# ============================================================
# 1. LEARNING CURVES
# ============================================================
def plot_learning_curves():
    """Plot Train Loss and Val Loss charts for all trained models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Learning Curves - VQA Seq2Seq Models", fontsize=16, fontweight="bold")

    has_data = False

    for model_id in range(1, 7):
        history_path = os.path.join(
            config.MODEL_DIRS[f"model_{model_id}"], "logs", "history.json"
        )
        if not os.path.exists(history_path):
            continue

        with open(history_path, "r") as f:
            history = json.load(f)

        epochs = range(1, len(history["train_loss"]) + 1)
        color = COLORS[model_id]
        label = MODEL_LABELS[model_id]

        # Train Loss
        axes[0].plot(epochs, history["train_loss"],
                     color=color, label=label, linewidth=2, marker="o", markersize=4)

        # Val Loss
        axes[1].plot(epochs, history["val_loss"],
                     color=color, label=label, linewidth=2, marker="o", markersize=4)

        has_data = True

    if not has_data:
        print("  No training history found. Train a model first.")
        plt.close()
        return

    axes[0].set_title("Training Loss", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("Validation Loss", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, "learning_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    # Val Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Validation Accuracy", fontsize=16, fontweight="bold")

    for model_id in range(1, 7):
        history_path = os.path.join(
            config.MODEL_DIRS[f"model_{model_id}"], "logs", "history.json"
        )
        if not os.path.exists(history_path):
            continue

        with open(history_path, "r") as f:
            history = json.load(f)

        if "val_accuracy" in history:
            epochs = range(1, len(history["val_accuracy"]) + 1)
            ax.plot(epochs, history["val_accuracy"],
                    color=COLORS[model_id], label=MODEL_LABELS[model_id],
                    linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, "val_accuracy.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# 2. METRICS COMPARISON
# ============================================================
def plot_comparison():
    """Plot bar chart comparing metrics across models."""
    comparison_path = os.path.join(config.RESULTS_DIR, "comparison.json")

    if not os.path.exists(comparison_path):
        # Default to reading from each model individually
        all_metrics = {}
        for model_id in range(1, 7):
            metrics_path = os.path.join(
                config.MODEL_DIRS[f"model_{model_id}"], "metrics.json"
            )
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    all_metrics[str(model_id)] = json.load(f)

        if not all_metrics:
            print("  No metrics found. Evaluate models first.")
            return
    else:
        with open(comparison_path, "r") as f:
            all_metrics = json.load(f)

    # Required Metrics
    metric_keys = ["accuracy", "bleu_1", "bleu_4", "meteor", "rouge_l", "cider"]
    metric_labels = ["Accuracy", "BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]

    model_ids = sorted(all_metrics.keys())
    x = np.arange(len(metric_keys))
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle("Model Comparison - Evaluation Metrics",
                 fontsize=16, fontweight="bold")

    for i, mid in enumerate(model_ids):
        mid_int = int(mid)
        values = [all_metrics[mid].get(k, 0) for k in metric_keys]
        bars = ax.bar(x + i * width, values, width,
                      label=MODEL_LABELS.get(mid_int, f"Model {mid}"),
                      color=COLORS.get(mid_int, "#95a5a6"),
                      edgecolor="white", linewidth=0.5)
        # Show values on top of bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x + width * (len(model_ids) - 1) / 2)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, "comparison_chart.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# 3. SAMPLE PREDICTIONS
# ============================================================
def plot_samples(model_id):
    """Show prediction samples: Image + Question + Answer."""
    predictions_path = os.path.join(
        config.MODEL_DIRS[f"model_{model_id}"], "predictions.json"
    )
    if not os.path.exists(predictions_path):
        print(f"  No predictions found for Model {model_id}.")
        return

    with open(predictions_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    num_samples = min(6, len(samples))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Sample Predictions - {MODEL_LABELS[model_id]}",
                 fontsize=16, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i >= num_samples:
            ax.axis("off")
            continue

        s = samples[i]
        img_path = os.path.join(config.IMAGES_DIR, f"{s['image_id']}.jpg")

        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Image\nNot Found", ha="center", va="center",
                    fontsize=14, transform=ax.transAxes)

        ax.set_title(
            f"Q: {s['question']}\n"
            f"Pred: {s['prediction']}\n"
            f"True: {s['reference_full']}",
            fontsize=9, wrap=True
        )
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(
        config.MODEL_DIRS[f"model_{model_id}"], "sample_predictions.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# 4. ATTENTION HEATMAPS
# ============================================================
def plot_attention(model_id):
    """
    Visualize Attention Heatmap for Model 3 & 4.
    Requires running inference separately to get attention weights.
    """
    if model_id not in [3, 4, 6]:
        print(f"  Attention visualization only for Model 3, 4 & 6.")
        return

    import torch
    from data.dataset import load_vocab, GQAFeaturesDataset, GQADataset

    vocab = load_vocab(config.VOCAB_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if model_id in [4, 6]:
        if model_id == 4:
            from models.model_4 import VQAModel4_PretrainedAtt
            model = VQAModel4_PretrainedAtt(len(vocab)).to(device)
            dataset = GQAFeaturesDataset(
                config.TEST_JSON, config.FEATURES_H5, vocab, use_spatial=True
            )
        else:
            from models.model_6 import VQAModel6_PretrainedEndToEndAtt
            model = VQAModel6_PretrainedEndToEndAtt(len(vocab)).to(device)
            dataset = GQADataset(
                config.TEST_JSON, config.IMAGES_DIR, vocab,
                image_size=config.PRETRAINED_IMAGE_SIZE
            )
        spatial_size = 7  # ResNet spatial: 7x7
    else:
        from models.model_3 import VQAModel3_ScratchAtt
        model = VQAModel3_ScratchAtt(len(vocab)).to(device)
        dataset = GQADataset(
            config.TEST_JSON, config.IMAGES_DIR, vocab,
            image_size=config.SCRATCH_IMAGE_SIZE
        )
        spatial_size = 8  # Scratch CNN spatial: 8x8

    # Load checkpoint
    logger = TrainingLogger(config.MODEL_DIRS[f"model_{model_id}"])
    epoch = logger.load_checkpoint(model, load_best=True)
    if epoch == 0:
        print(f"  No checkpoint for Model {model_id}.")
        return

    model.eval()

    # Load raw data
    with open(config.TEST_JSON, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    raw_items = list(raw_data.values())

    # Generate for a few samples
    num_vis = 4
    fig, axes = plt.subplots(num_vis, 6, figsize=(24, num_vis * 4))
    fig.suptitle(f"Attention Maps - {MODEL_LABELS[model_id]}",
                 fontsize=16, fontweight="bold")

    with torch.no_grad():
        for i in range(min(num_vis, len(dataset))):
            inputs, questions, answers = dataset[i]
            inputs = inputs.unsqueeze(0).to(device)
            questions = questions.unsqueeze(0).to(device)

            if model_id in [4, 6]:
                generated, alphas = model.generate(
                    inputs, questions, vocab.sos_idx, vocab.eos_idx, max_len=15
                )
            else:
                generated, alphas = model.generate(
                    inputs, questions, vocab.sos_idx, vocab.eos_idx, max_len=15
                )

            # Decode
            pred_words = []
            for idx in generated[0]:
                word = vocab.itos.get(idx.item(), "<UNK>")
                if word == "<EOS>":
                    break
                if word not in ["<SOS>", "<PAD>"]:
                    pred_words.append(word)

            # Show original image
            img_id = raw_items[i]["imageId"]
            img_path = os.path.join(config.IMAGES_DIR, f"{img_id}.jpg")

            for j in range(6):
                ax = axes[i][j] if num_vis > 1 else axes[j]

                if j == 0:
                    # First column: Original image + Question
                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert("RGB")
                        ax.imshow(img)
                    ax.set_title(f"Q: {raw_items[i]['question'][:50]}", fontsize=8)
                elif j - 1 < len(alphas) and j - 1 < len(pred_words):
                    # Next columns: attention map for each word
                    alpha = alphas[j-1][0].cpu().numpy()
                    alpha = alpha.reshape(spatial_size, spatial_size)

                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert("RGB")
                        img = img.resize((spatial_size * 32, spatial_size * 32))
                        ax.imshow(img, alpha=0.5)

                    ax.imshow(
                        np.array(Image.fromarray(
                            (alpha * 255).astype(np.uint8)
                        ).resize((spatial_size * 32, spatial_size * 32))),
                        alpha=0.6, cmap="jet"
                    )
                    ax.set_title(f'"{pred_words[j-1]}"', fontsize=10, fontweight="bold")
                else:
                    ax.axis("off")
                    continue

                ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(config.ATTENTION_MAPS_DIR,
                             f"attention_model_{model_id}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Visualize VQA Results")
    parser.add_argument("--curves", action="store_true", help="Plot learning curves")
    parser.add_argument("--compare", action="store_true", help="Compare metrics")
    parser.add_argument("--samples", type=int, help="Show sample predictions for model")
    parser.add_argument("--attention", type=int, help="Show attention maps for model")
    parser.add_argument("--all", action="store_true", help="Generate all visualizations")
    args = parser.parse_args()

    print("=" * 60)
    print("VISUALIZATION")
    print("=" * 60)

    if args.all or args.curves:
        print("\n[1] Learning Curves...")
        plot_learning_curves()

    if args.all or args.compare:
        print("\n[2] Metrics Comparison...")
        plot_comparison()

    if args.samples or args.all:
        model_ids = [args.samples] if args.samples else range(1, 7)
        for mid in model_ids:
            print(f"\n[3] Sample Predictions (Model {mid})...")
            plot_samples(mid)

    if args.attention or args.all:
        model_ids = [args.attention] if args.attention else [3, 4, 6]
        for mid in model_ids:
            print(f"\n[4] Attention Maps (Model {mid})...")
            plot_attention(mid)

    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
