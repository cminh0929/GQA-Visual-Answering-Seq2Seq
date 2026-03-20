"""
build_vocab.py - Xây dựng bộ từ điển (Vocabulary) từ tập train subset.
Bước 1 trong VQA_Seq2Seq_Project_Plan.md

Quét toàn bộ câu hỏi (question) và câu trả lời (fullAnswer) trong tập train,
xây dựng bộ từ điển và lưu vào file vocab.pkl.
"""

import json
import os
import sys

# Thêm thư mục gốc vào path để import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import TRAIN_JSON, VOCAB_PATH, FREQ_THRESHOLD
from data.dataset import Vocabulary, save_vocab


def build():
    print("=" * 60)
    print("BUILDING VOCABULARY")
    print("=" * 60)

    # 1. Load dữ liệu train
    print(f"\n[1/3] Loading training data: {TRAIN_JSON}")
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"      Total questions: {len(data)}")

    # 2. Thu thập tất cả các câu (question + fullAnswer)
    print(f"\n[2/3] Collecting sentences...")
    all_sentences = []
    for item in data.values():
        all_sentences.append(item["question"])
        all_sentences.append(item["fullAnswer"])
    print(f"      Total sentences: {len(all_sentences)} "
          f"({len(all_sentences)//2} questions + {len(all_sentences)//2} answers)")

    # 3. Xây dựng vocab
    print(f"\n[3/3] Building vocabulary (freq_threshold={FREQ_THRESHOLD})...")
    vocab = Vocabulary(freq_threshold=FREQ_THRESHOLD)
    vocab.build_vocabulary(all_sentences)

    # Thống kê
    print(f"\n--- VOCABULARY STATISTICS ---")
    print(f"Total words in vocab: {len(vocab)}")
    print(f"Special tokens: <PAD>={vocab.pad_idx}, <SOS>={vocab.sos_idx}, "
          f"<EOS>={vocab.eos_idx}, <UNK>={vocab.unk_idx}")

    # Demo: chuyển đổi một mẫu
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

    # Decode ngược lại
    q_decoded = vocab.decode(vocab.numericalize(q))
    print(f"Decoded Q: {q_decoded}")

    # 4. Lưu vocab
    print(f"\n--- SAVING ---")
    save_vocab(vocab, VOCAB_PATH)
    print(f"\nDone! Vocab saved to: {VOCAB_PATH}")


if __name__ == "__main__":
    build()
