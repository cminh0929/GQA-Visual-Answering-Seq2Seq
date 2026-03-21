"""
evaluate.py - Evaluate VQA Seq2Seq models on Test set
Step 5 in VQA_Seq2Seq_Project_Plan.md

Run inference on testdev, compute all metrics,
and save results to predictions.json + metrics.json.

Usage:
    py -3.10 evaluate.py --model 2
    py -3.10 evaluate.py --model 4
    py -3.10 evaluate.py --model 1 2 3 4    (evaluate all)
"""

import argparse
import json
import os
import sys
import time
import torch
from tqdm import tqdm

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
from utils.logger import TrainingLogger
from utils.metrics import compute_all_metrics


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAMES = {
    1: "Scratch + No Attention (End-to-End)",
    2: "Pretrained + No Attention (Pre-extracted)",
    3: "Scratch + Attention (End-to-End)",
    4: "Pretrained + Attention (Pre-extracted)",
    5: "Pretrained + No Attention (End-to-End)",
    6: "Pretrained + Attention (End-to-End)",
}


def build_model(model_id, vocab_size):
    """Create model."""
    models_map = {
        1: VQAModel1_ScratchNoAtt,
        2: VQAModel2_PretrainedNoAtt,
        3: VQAModel3_ScratchAtt,
        4: VQAModel4_PretrainedAtt,
        5: VQAModel5_PretrainedEndToEndNoAtt,
        6: VQAModel6_PretrainedEndToEndAtt,
    }
    return models_map[model_id](vocab_size).to(DEVICE)


def build_test_loader(model_id, vocab):
    """Create DataLoader for test set."""
    is_pretrained = model_id in [2, 4]       # Pre-extracted features
    is_end2end_pretrained = model_id in [5, 6]  # End-to-end pretrained
    has_attention = model_id in [3, 4, 6]

    if is_pretrained:
        dataset = GQAFeaturesDataset(
            config.TEST_JSON, config.FEATURES_H5, vocab,
            use_spatial=has_attention
        )
        batch_size = config.PRETRAINED_BATCH_SIZE
    else:
        # Model 1, 3, 5, 6: raw images
        uses_pretrained_cfg = is_pretrained or is_end2end_pretrained
        image_size = config.PRETRAINED_IMAGE_SIZE if uses_pretrained_cfg else config.SCRATCH_IMAGE_SIZE
        dataset = GQADataset(
            config.TEST_JSON, config.IMAGES_DIR, vocab,
            image_size=image_size
        )
        batch_size = config.E2E_PRETRAINED_BATCH_SIZE if is_end2end_pretrained else config.SCRATCH_BATCH_SIZE

    return get_dataloader(
        dataset, batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pad_idx=vocab.pad_idx
    ), dataset


def evaluate_model(model_id, vocab):
    """
    Evaluate a model on test set.

    Returns:
        metrics: dict containing all metrics
        predictions_list: list of tuples (prediction, reference)
    """
    model_dir = config.MODEL_DIRS[f"model_{model_id}"]
    has_attention = model_id in [3, 4, 6]

    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL {model_id}: {MODEL_NAMES[model_id]}")
    print(f"{'='*60}")

    # 1. Build model & load best checkpoint
    model = build_model(model_id, len(vocab))
    logger = TrainingLogger(model_dir)
    epoch = logger.load_checkpoint(model, load_best=True)

    if epoch == 0:
        print(f"  ⚠ No checkpoint found! Skipping Model {model_id}.")
        return None, None

    model.eval()

    # 2. Build test loader
    test_loader, test_dataset = build_test_loader(model_id, vocab)
    print(f"  Test samples: {len(test_dataset)}")

    # 3. Load raw data for ground truth
    with open(config.TEST_JSON, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    raw_keys = list(raw_data.keys())

    # 4. Inference
    print("  Running inference...")
    all_predictions = []
    all_references = []
    all_short_preds = []
    all_short_refs = []
    sample_outputs = []

    idx_counter = 0
    start_time = time.time()

    with torch.no_grad():
        for inputs, questions, answers in tqdm(test_loader, desc="  Evaluating"):
            inputs = inputs.to(DEVICE)
            questions = questions.to(DEVICE)
            batch_size = inputs.size(0)

            # Generate answers
            if has_attention:
                generated, _ = model.generate(
                    inputs, questions,
                    vocab.sos_idx, vocab.eos_idx,
                    max_len=config.MAX_ANSWER_LENGTH
                )
            else:
                generated = model.generate(
                    inputs, questions,
                    vocab.sos_idx, vocab.eos_idx,
                    max_len=config.MAX_ANSWER_LENGTH
                )

            # Decode predictions and references
            for i in range(batch_size):
                pred_text = vocab.decode(generated[i].cpu())
                ref_item = raw_data[raw_keys[idx_counter]]
                ref_full = ref_item["fullAnswer"]
                ref_short = ref_item.get("answer", "")

                all_predictions.append(pred_text)
                all_references.append(ref_full)
                all_short_preds.append(pred_text.split(".")[-1].strip() if "." in pred_text else pred_text)
                all_short_refs.append(ref_short)

                # Save sample outputs for review
                if len(sample_outputs) < 10:
                    sample_outputs.append({
                        "question": ref_item["question"],
                        "prediction": pred_text,
                        "reference_full": ref_full,
                        "reference_short": ref_short,
                        "image_id": ref_item["imageId"],
                    })

                idx_counter += 1

    elapsed = time.time() - start_time

    # 5. Compute metrics
    print("  Computing metrics...")
    metrics = compute_all_metrics(all_predictions, all_references)
    metrics["short_accuracy"] = sum(
        1 for p, r in zip(all_predictions, all_short_refs)
        if r.lower() in p.lower()
    ) / max(len(all_predictions), 1)
    metrics["inference_time"] = elapsed
    metrics["num_samples"] = len(all_predictions)

    # 6. Print results
    print(f"\n  --- RESULTS (Model {model_id}) ---")
    print(f"  {'Metric':<20} {'Score':>10}")
    print(f"  {'-'*30}")
    for key, value in metrics.items():
        if key not in ["inference_time", "num_samples"]:
            print(f"  {key:<20} {value:>10.4f}")
    print(f"  {'inference_time':<20} {elapsed:>10.2f}s")

    # 7. Print samples
    print(f"\n  --- SAMPLE PREDICTIONS ---")
    for s in sample_outputs[:5]:
        print(f"  Q: {s['question']}")
        print(f"  P: {s['prediction']}")
        print(f"  R: {s['reference_full']}")
        print()

    # 8. Save results
    predictions_path = os.path.join(model_dir, "predictions.json")
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(sample_outputs, f, indent=2, ensure_ascii=False)

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved: {predictions_path}")
    print(f"  Saved: {metrics_path}")

    return metrics, sample_outputs


def compare_models(all_metrics):
    """Print model comparison table."""
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")

    # Header
    model_ids = sorted(all_metrics.keys())
    header = f"{'Metric':<20}"
    for mid in model_ids:
        header += f"{'Model '+str(mid):>15}"
    print(header)
    print("-" * (20 + 15 * len(model_ids)))

    # Rows
    metric_keys = ["accuracy", "short_accuracy", "bleu_1", "bleu_2",
                   "bleu_3", "bleu_4", "meteor", "rouge_l", "cider"]
    for key in metric_keys:
        row = f"  {key:<18}"
        for mid in model_ids:
            val = all_metrics[mid].get(key, 0)
            row += f"{val:>15.4f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA Seq2Seq Model")
    parser.add_argument("--model", type=int, nargs="+", required=True,
                        choices=[1, 2, 3, 4, 5, 6],
                        help="Model ID(s) to evaluate (e.g. 1 2 3 4 5 6)")
    args = parser.parse_args()

    # Load vocab
    print("Loading vocabulary...")
    vocab = load_vocab(config.VOCAB_PATH)

    # Evaluate
    all_metrics = {}
    for model_id in args.model:
        metrics, _ = evaluate_model(model_id, vocab)
        if metrics is not None:
            all_metrics[model_id] = metrics

    # Compare models if multiple models
    if len(all_metrics) > 1:
        compare_models(all_metrics)

        # Save comparison
        comparison_path = os.path.join(config.RESULTS_DIR, "comparison.json")
        serializable = {str(k): v for k, v in all_metrics.items()}
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nComparison saved to: {comparison_path}")

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
