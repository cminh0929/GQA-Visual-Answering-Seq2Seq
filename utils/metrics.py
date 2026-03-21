"""
metrics.py - Evaluation metrics for VQA Seq2Seq
Step 5 in VQA_Seq2Seq_Project_Plan.md

Supports: Accuracy, BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr-D
"""

import re
from collections import Counter
import math


def tokenize(text):
    """Split sentence into list of words (lowercase)."""
    text = text.lower().strip()
    for p in ["?", ".", ",", "!", ";", ":"]:
        text = text.replace(p, f" {p}")
    return text.split()


# ============================================================
# ACCURACY
# ============================================================
def compute_accuracy(predictions, references):
    """
    Accuracy: 100% exact match comparison (lowercase, strip).

    Args:
        predictions: List[str] - predicted answers
        references: List[str] - ground truth answers
    Returns:
        float
    """
    correct = sum(
        1 for p, r in zip(predictions, references)
        if p.strip().lower() == r.strip().lower()
    )
    return correct / max(len(predictions), 1)


# ============================================================
# BLEU (1, 2, 3, 4)
# ============================================================
def _ngrams(tokens, n):
    """Extract n-grams."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _bleu_sentence(prediction_tokens, reference_tokens, max_n=4):
    """Compute BLEU for 1 pair of sentences."""
    scores = []
    for n in range(1, max_n + 1):
        pred_ngrams = _ngrams(prediction_tokens, n)
        ref_ngrams = _ngrams(reference_tokens, n)

        if not pred_ngrams:
            scores.append(0.0)
            continue

        ref_counts = Counter(ref_ngrams)
        clipped = 0
        for ng in pred_ngrams:
            if ref_counts[ng] > 0:
                clipped += 1
                ref_counts[ng] -= 1

        scores.append(clipped / len(pred_ngrams))
    return scores


def compute_bleu(predictions, references, max_n=4):
    """
    Corpus-level BLEU-1 to BLEU-4.

    Args:
        predictions: List[str]
        references: List[str]
    Returns:
        dict: {"bleu_1": float, "bleu_2": float, ...}
    """
    all_scores = [[] for _ in range(max_n)]
    total_bp = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)

        scores = _bleu_sentence(pred_tokens, ref_tokens, max_n)
        for i, s in enumerate(scores):
            all_scores[i].append(s)

        # Brevity penalty
        if len(pred_tokens) < len(ref_tokens):
            total_bp += 1

    results = {}
    for n in range(max_n):
        avg_precision = sum(all_scores[n]) / max(len(all_scores[n]), 1)

        # Brevity penalty
        bp_ratio = total_bp / max(len(predictions), 1)
        bp = math.exp(1 - 1 / max(1 - bp_ratio, 0.01)) if bp_ratio > 0.5 else 1.0

        results[f"bleu_{n+1}"] = avg_precision * bp

    return results


# ============================================================
# METEOR (Simplified)
# ============================================================
def compute_meteor(predictions, references):
    """
    Simplified METEOR (unigram matching + penalty for gaps).

    Args:
        predictions: List[str]
        references: List[str]
    Returns:
        float
    """
    total_score = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)

        if not pred_tokens or not ref_tokens:
            continue

        # Unigram matching
        pred_set = Counter(pred_tokens)
        ref_set = Counter(ref_tokens)
        matches = sum((pred_set & ref_set).values())

        precision = matches / max(len(pred_tokens), 1)
        recall = matches / max(len(ref_tokens), 1)

        if precision + recall == 0:
            continue

        # F-mean (METEOR uses harmonic mean with higher recall weight)
        alpha = 0.9  # Weight for recall
        f_score = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        total_score += f_score

    return total_score / max(len(predictions), 1)


# ============================================================
# ROUGE-L
# ============================================================
def _lcs_length(x, y):
    """Compute Longest Common Subsequence length."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def compute_rouge_l(predictions, references):
    """
    ROUGE-L based on Longest Common Subsequence.

    Args:
        predictions: List[str]
        references: List[str]
    Returns:
        float
    """
    total_score = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)

        if not pred_tokens or not ref_tokens:
            continue

        lcs = _lcs_length(pred_tokens, ref_tokens)
        precision = lcs / max(len(pred_tokens), 1)
        recall = lcs / max(len(ref_tokens), 1)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        total_score += f1

    return total_score / max(len(predictions), 1)


# ============================================================
# CIDEr-D (Simplified)
# ============================================================
def compute_cider(predictions, references):
    """
    Simplified CIDEr-D (TF-IDF weighted n-gram matching).

    Args:
        predictions: List[str]
        references: List[str]
    Returns:
        float
    """
    # Build document frequency
    doc_freq = Counter()
    for ref in references:
        tokens = set(tokenize(ref))
        for t in tokens:
            doc_freq[t] += 1

    num_docs = len(references)
    total_score = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)

        if not pred_tokens or not ref_tokens:
            continue

        # TF-IDF for prediction
        pred_tf = Counter(pred_tokens)
        pred_tfidf = {}
        for t, count in pred_tf.items():
            tf = count / len(pred_tokens)
            idf = math.log(max(num_docs, 1) / max(doc_freq.get(t, 0), 1))
            pred_tfidf[t] = tf * idf

        # TF-IDF for reference
        ref_tf = Counter(ref_tokens)
        ref_tfidf = {}
        for t, count in ref_tf.items():
            tf = count / len(ref_tokens)
            idf = math.log(max(num_docs, 1) / max(doc_freq.get(t, 0), 1))
            ref_tfidf[t] = tf * idf

        # Cosine similarity
        all_tokens = set(list(pred_tfidf.keys()) + list(ref_tfidf.keys()))
        dot = sum(pred_tfidf.get(t, 0) * ref_tfidf.get(t, 0) for t in all_tokens)
        norm_p = math.sqrt(sum(v**2 for v in pred_tfidf.values())) or 1
        norm_r = math.sqrt(sum(v**2 for v in ref_tfidf.values())) or 1
        cosine = dot / (norm_p * norm_r)

        total_score += cosine

    return (total_score / max(len(predictions), 1)) * 10  # CIDEr usually scales x10


# ============================================================
# ALL METRICS
# ============================================================
def compute_all_metrics(predictions, references):
    """
    Compute all metrics.

    Args:
        predictions: List[str] - predicted answers
        references: List[str] - ground truth answers
    Returns:
        dict: Contains all metrics
    """
    metrics = {}

    # Accuracy
    metrics["accuracy"] = compute_accuracy(predictions, references)

    # BLEU
    bleu_scores = compute_bleu(predictions, references)
    metrics.update(bleu_scores)

    # METEOR
    metrics["meteor"] = compute_meteor(predictions, references)

    # ROUGE-L
    metrics["rouge_l"] = compute_rouge_l(predictions, references)

    # CIDEr
    metrics["cider"] = compute_cider(predictions, references)

    return metrics
