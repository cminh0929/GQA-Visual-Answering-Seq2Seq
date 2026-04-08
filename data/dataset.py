"""
dataset.py - Vocabulary, Dataset and DataLoader for VQA Seq2Seq
Step 1 in VQA_Seq2Seq_Project_Plan.md
"""

import os
import json
import torch
import pickle
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from collections import Counter
from torchvision import transforms


# ============================================================
# 1. VOCABULARY CLASS
# ============================================================
class Vocabulary:
    """
    Vocabulary dictionary converting between words and indices.
    Supports special tokens: <PAD>, <SOS>, <EOS>, <UNK>.
    """

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, freq_threshold=3):
        """
        Args:
            freq_threshold: Minimum frequency for a word to be added to vocab.
        """
        self.freq_threshold = freq_threshold

        # Index → Word
        self.itos = {
            0: self.PAD_TOKEN,
            1: self.SOS_TOKEN,
            2: self.EOS_TOKEN,
            3: self.UNK_TOKEN,
        }
        # Word → Index
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @property
    def pad_idx(self):
        return self.stoi[self.PAD_TOKEN]

    @property
    def sos_idx(self):
        return self.stoi[self.SOS_TOKEN]

    @property
    def eos_idx(self):
        return self.stoi[self.EOS_TOKEN]

    @property
    def unk_idx(self):
        return self.stoi[self.UNK_TOKEN]

    @staticmethod
    def tokenize(text):
        """
        Split sentence into list of tokens.
        Handles basic punctuation: splits ? . , ! from words.
        """
        text = text.lower().strip()
        # Split punctuation into individual tokens
        for punct in ["?", ".", ",", "!", ";", ":"]:
            text = text.replace(punct, f" {punct}")
        return text.split()

    def build_vocabulary(self, sentence_list):
        """
        Build vocabulary from a list of sentences.
        Only add words with frequency >= freq_threshold.

        Args:
            sentence_list: List[str] - list of sentences (question + fullAnswer)
        """
        frequencies = Counter()
        idx = len(self.itos)  # Start from index 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                # When word frequency reaches threshold, add to vocab
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

        print(f"Vocabulary built: {len(self.itos)} words "
              f"(freq_threshold={self.freq_threshold})")

    def numericalize(self, text):
        """
        Convert text sentence to a list of indices.
        Words not in vocab will be replaced with <UNK>.
        """
        return [
            self.stoi.get(token, self.unk_idx)
            for token in self.tokenize(text)
        ]

    def decode(self, indices):
        """
        Convert list of indices back to a text sentence.
        Stop when encountering <EOS> or <PAD>.
        """
        words = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            word = self.itos.get(idx, self.UNK_TOKEN)
            if word == self.EOS_TOKEN or word == self.PAD_TOKEN:
                break
            if word != self.SOS_TOKEN:
                words.append(word)
        return " ".join(words)


def save_vocab(vocab, path):
    """Save Vocabulary to pickle file."""
    with open(path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to {path}")


def load_vocab(path):
    """Load Vocabulary from pickle file."""
    with open(path, "rb") as f:
        vocab = pickle.load(f)
    print(f"Vocabulary loaded: {len(vocab)} words")
    return vocab


# ============================================================
# 2. DATASET CLASSES
# ============================================================

def get_image_transform(image_size=224):
    """
    Standard transform for input image.
    Used for both Scratch Model (128x128) and Pretrained Model (224x224).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


class GQADataset(Dataset):
    """
    Dataset for Model 1 & 3 (Scratch - End-to-End).
    Returns: (image_tensor, question_indices, answer_indices)
    Images are loaded directly from disk and transformed each call.
    """

    def __init__(self, json_file, images_dir, vocab, image_size=128):
        """
        Args:
            json_file: Path to JSON file (train/val/test subset)
            images_dir: Path to directory containing images
            vocab: Built Vocabulary object
            image_size: Output image size (128 for Scratch)
        """
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.keys = list(self.data.keys())
        self.images_dir = images_dir
        self.vocab = vocab
        self.transform = get_image_transform(image_size)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        item = self.data[self.keys[index]]
        img_id = item["imageId"]
        question = item["question"]
        answer = item["fullAnswer"]

        # Load and transform image
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Convert text → tensor indices (add SOS and EOS)
        q_indices = (
            [self.vocab.sos_idx]
            + self.vocab.numericalize(question)
            + [self.vocab.eos_idx]
        )
        a_indices = (
            [self.vocab.sos_idx]
            + self.vocab.numericalize(answer)
            + [self.vocab.eos_idx]
        )

        return img, torch.tensor(q_indices), torch.tensor(a_indices)


class GQAFeaturesDataset(Dataset):
    """
    Dataset for Model 2 & 4 (Pretrained - Pre-extracted Features).
    Returns: (feature_vector, question_indices, answer_indices)
    Features are loaded from HDF5 file (pre-extracted using ResNet-50).
    """

    def __init__(self, json_file, features_h5_path, vocab, use_spatial=False):
        """
        Args:
            json_file: Path to JSON file (train/val/test subset)
            features_h5_path: Path to .h5 file containing features
            vocab: Built Vocabulary object
            use_spatial: True → extract 7x7x2048 feature map (for Attention)
                         False → extract 2048-d vector (for No Attention)
        """
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.keys = list(self.data.keys())
        self.features_h5_path = features_h5_path
        self.vocab = vocab
        self.use_spatial = use_spatial

        # Open HDF5 file (lazy loading)
        self.h5_file = None

    def _open_h5(self):
        """Open HDF5 file (needs to be reopened for each worker in DataLoader)."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.features_h5_path, "r")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        self._open_h5()

        item = self.data[self.keys[index]]
        img_id = item["imageId"]
        question = item["question"]
        answer = item["fullAnswer"]

        # Load feature from HDF5
        if self.use_spatial:
            # 7x7x2048 feature map for Attention (Model 4)
            feature = torch.tensor(
                self.h5_file[f"spatial/{img_id}"][()], dtype=torch.float32
            )
        else:
            # 2048-d vector for No Attention (Model 2)
            feature = torch.tensor(
                self.h5_file[f"pooled/{img_id}"][()], dtype=torch.float32
            )

        # Convert text → tensor indices
        q_indices = (
            [self.vocab.sos_idx]
            + self.vocab.numericalize(question)
            + [self.vocab.eos_idx]
        )
        a_indices = (
            [self.vocab.sos_idx]
            + self.vocab.numericalize(answer)
            + [self.vocab.eos_idx]
        )

        return feature, torch.tensor(q_indices), torch.tensor(a_indices)



# ============================================================
# 3. COLLATE FUNCTION & DATALOADER
# ============================================================

class VQACollate:
    """
    Custom collate function to handle padding for batches
    with questions and answers of varying lengths.
    """

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Split batch into 3 lists
        imgs_or_features, questions, answers = zip(*batch)

        # Stack images/features to tensor
        imgs_or_features = torch.stack(imgs_or_features, dim=0)

        # Pad questions and answers
        questions = pad_sequence(
            questions, batch_first=True, padding_value=self.pad_idx
        )
        answers = pad_sequence(
            answers, batch_first=True, padding_value=self.pad_idx
        )

        return imgs_or_features, questions, answers


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=2, pad_idx=0):
    """
    Create DataLoader from Dataset with custom collate function.

    Args:
        dataset: GQADataset or GQAFeaturesDataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers for DataLoader
        pad_idx: Index of <PAD> token

    Returns:
        DataLoader
    """
    collate_fn = VQACollate(pad_idx=pad_idx)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
