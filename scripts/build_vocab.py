"""
build_vocab.py - Build vocabulary from the train subset.
Step 1 in VQA_Seq2Seq_Project_Plan.md

Scans all questions and fullAnswers in the train set,
builds the vocabulary and saves it to vocab.pkl.
"""

import json
import os
import sys

# Add root directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vqa_config import TRAIN_JSON, VOCAB_PATH, FREQ_THRESHOLD
from data.dataset import Vocabulary, save_vocab


def build():
    print("=" * 60)
    print("BUILDING VOCABULARY")
    print("=" * 60)

    # 1. Load training data
    print(f"\n[1/3] Loading training data: {TRAIN_JSON}")
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"      Total questions: {len(data)}")

    # 2. Collect all sentences (question + fullAnswer)
    print(f"\n[2/3] Collecting sentences...")
    all_sentences = []
    for item in data.values():
        all_sentences.append(item["question"])
        all_sentences.append(item["fullAnswer"])
    print(f"      Total sentences: {len(all_sentences)} "
          f"({len(all_sentences)//2} questions + {len(all_sentences)//2} answers)")

    # 3. Build vocab
    print(f"\n[3/3] Building vocabulary (freq_threshold={FREQ_THRESHOLD})...")
    vocab = Vocabulary(freq_threshold=FREQ_THRESHOLD)
    vocab.build_vocabulary(all_sentences)

    # Statistics
    print(f"\n--- VOCABULARY STATISTICS ---")
    print(f"Total words in vocab: {len(vocab)}")
    print(f"Special tokens: <PAD>={vocab.pad_idx}, <SOS>={vocab.sos_idx}, "
          f"<EOS>={vocab.eos_idx}, <UNK>={vocab.unk_idx}")

    # Demo: convert one sample
    sample = list(data.values())[0]
    q = sample["question"]
    a = sample["fullAnswer"]
    print(f"\n--- DEMO ---")
    print(f"Question:  {q}")
    print(f"Tokens:    {vocab.tokenize(q)}")
    print(f"Indices:   {vocab.numericalize(q)}")
    print(f"Answer:    {a}")
    print(f"Tokens:    {vocab.tokenize(a)}")
    print(f"Indices:   {vocab.numericalize(a)}")

    # Decode back
    q_decoded = vocab.decode(vocab.numericalize(q))
    print(f"Decoded Q: {q_decoded}")

    # 4. Save vocab
    print(f"\n--- SAVING ---")
    save_vocab(vocab, VOCAB_PATH)
    print(f"\nDone! Vocab saved to: {VOCAB_PATH}")


if __name__ == "__main__":
    build()
