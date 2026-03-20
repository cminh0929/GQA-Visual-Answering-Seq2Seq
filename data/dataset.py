"""
data_utils.py - Vocabulary, Dataset và DataLoader cho VQA Seq2Seq
Bước 1 trong VQA_Seq2Seq_Project_Plan.md
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
    Bộ từ điển chuyển đổi giữa từ (word) và chỉ số (index).
    Hỗ trợ các token đặc biệt: <PAD>, <SOS>, <EOS>, <UNK>.
    """

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, freq_threshold=3):
        """
        Args:
            freq_threshold: Số lần xuất hiện tối thiểu để một từ được thêm vào vocab.
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
        Tách câu thành danh sách các từ (tokens).
        Xử lý dấu câu cơ bản: tách dấu ? . , ! ra khỏi từ.
        """
        text = text.lower().strip()
        # Tách dấu câu ra thành token riêng
        for punct in ["?", ".", ",", "!", ";", ":"]:
            text = text.replace(punct, f" {punct}")
        return text.split()

    def build_vocabulary(self, sentence_list):
        """
        Xây dựng bộ từ điển từ danh sách câu.
        Chỉ thêm từ có tần suất >= freq_threshold.

        Args:
            sentence_list: List[str] - danh sách các câu (question + fullAnswer)
        """
        frequencies = Counter()
        idx = len(self.itos)  # Bắt đầu từ index 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                # Khi từ đạt ngưỡng tần suất, thêm vào vocab
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

        print(f"Vocabulary built: {len(self.itos)} words "
              f"(freq_threshold={self.freq_threshold})")

    def numericalize(self, text):
        """
        Chuyển câu văn bản thành danh sách các index.
        Từ không có trong vocab sẽ được thay bằng <UNK>.
        """
        return [
            self.stoi.get(token, self.unk_idx)
            for token in self.tokenize(text)
        ]

    def decode(self, indices):
        """
        Chuyển danh sách index ngược lại thành câu văn bản.
        Dừng khi gặp <EOS> hoặc <PAD>.
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
    """Lưu Vocabulary vào file pickle."""
    with open(path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to {path}")


def load_vocab(path):
    """Load Vocabulary từ file pickle."""
    with open(path, "rb") as f:
        vocab = pickle.load(f)
    print(f"Vocabulary loaded: {len(vocab)} words")
    return vocab


# ============================================================
# 2. DATASET CLASSES
# ============================================================

def get_image_transform(image_size=224):
    """
    Transform chuẩn cho ảnh đầu vào.
    Dùng cho cả Model Scratch (128x128) và Pretrained (224x224).
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
    Dataset cho Model 1 & 3 (Scratch - End-to-End).
    Trả về: (image_tensor, question_indices, answer_indices)
    Ảnh được load trực tiếp từ đĩa và transform mỗi lần gọi.
    """

    def __init__(self, json_file, images_dir, vocab, image_size=128):
        """
        Args:
            json_file: Đường dẫn tới tệp JSON (train/val/test subset)
            images_dir: Đường dẫn thư mục chứa ảnh
            vocab: Đối tượng Vocabulary đã xây dựng
            image_size: Kích thước ảnh đầu ra (128 cho Scratch)
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

        # Load và transform ảnh
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Chuyển văn bản → tensor indices (thêm SOS và EOS)
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
    Dataset cho Model 2 & 4 (Pretrained - Pre-extracted Features).
    Trả về: (feature_vector, question_indices, answer_indices)
    Features được load từ file HDF5 (đã trích xuất trước bằng ResNet-50).
    """

    def __init__(self, json_file, features_h5_path, vocab, use_spatial=False):
        """
        Args:
            json_file: Đường dẫn tới tệp JSON (train/val/test subset)
            features_h5_path: Đường dẫn tới file .h5 chứa features
            vocab: Đối tượng Vocabulary đã xây dựng
            use_spatial: True → lấy feature map 7x7x2048 (cho Attention)
                         False → lấy vector 2048-d (cho No Attention)
        """
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.keys = list(self.data.keys())
        self.features_h5_path = features_h5_path
        self.vocab = vocab
        self.use_spatial = use_spatial

        # Mở file HDF5 (lazy loading)
        self.h5_file = None

    def _open_h5(self):
        """Mở file HDF5 (cần mở lại cho mỗi worker trong DataLoader)."""
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

        # Load feature từ HDF5
        if self.use_spatial:
            # Feature map 7x7x2048 cho Attention (Model 4)
            feature = torch.tensor(
                self.h5_file[f"spatial/{img_id}"][()], dtype=torch.float32
            )
        else:
            # Vector 2048-d cho No Attention (Model 2)
            feature = torch.tensor(
                self.h5_file[f"pooled/{img_id}"][()], dtype=torch.float32
            )

        # Chuyển văn bản → tensor indices
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

    def __del__(self):
        try:
            if self.h5_file is not None:
                self.h5_file.close()
        except Exception:
            pass


# ============================================================
# 3. COLLATE FUNCTION & DATALOADER
# ============================================================

class VQACollate:
    """
    Custom collate function để xử lý padding cho các batch
    có câu hỏi và câu trả lời dài ngắn khác nhau.
    """

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Tách batch thành 3 danh sách
        imgs_or_features, questions, answers = zip(*batch)

        # Stack ảnh/features thành tensor
        imgs_or_features = torch.stack(imgs_or_features, dim=0)

        # Pad câu hỏi và câu trả lời
        questions = pad_sequence(
            questions, batch_first=True, padding_value=self.pad_idx
        )
        answers = pad_sequence(
            answers, batch_first=True, padding_value=self.pad_idx
        )

        return imgs_or_features, questions, answers


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=2, pad_idx=0):
    """
    Tạo DataLoader từ Dataset với custom collate function.

    Args:
        dataset: GQADataset hoặc GQAFeaturesDataset
        batch_size: Kích thước batch
        shuffle: Có xáo trộn dữ liệu không
        num_workers: Số worker cho DataLoader
        pad_idx: Index của token <PAD>

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
