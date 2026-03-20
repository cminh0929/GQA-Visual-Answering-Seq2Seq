"""
extract_features.py - Trích xuất đặc trưng ảnh bằng ResNet-50 Pretrained
Bước 2 trong VQA_Seq2Seq_Project_Plan.md

Chạy tất cả ảnh (Train + Val + Test) qua ResNet-50 một lần duy nhất,
lưu kết quả vào file HDF5:
  - pooled/{img_id}:  vector 2048-d       (cho Model 2 - No Attention)
  - spatial/{img_id}: feature map 49x2048  (cho Model 4 - Attention)

Usage:
    py -3.10 scripts/extract_features.py
"""

import json
import os
import sys
import h5py
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ============================================================
# CẤU HÌNH
# ============================================================
BATCH_SIZE = 32  # Số ảnh xử lý mỗi lần (giảm nếu hết VRAM)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_all_image_ids():
    """
    Thu thập tất cả Image ID cần trích xuất từ Train + Val + Test.
    Trả về set các ID duy nhất.
    """
    all_ids = set()

    for json_path in [config.TRAIN_JSON, config.VAL_JSON, config.TEST_JSON]:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for v in data.values():
                all_ids.add(v["imageId"])
            print(f"  Loaded {len(data)} questions from {os.path.basename(json_path)}")

    print(f"  Total unique images to extract: {len(all_ids)}")
    return list(all_ids)


def build_resnet50():
    """
    Load ResNet-50 Pretrained, bỏ lớp FC cuối cùng.
    Trả về model ở chế độ eval (không train).
    """
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Bỏ lớp avgpool và fc cuối cùng
    # Lấy output từ layer4 → feature map (B, 2048, 7, 7)
    modules = list(resnet.children())[:-2]  # Bỏ avgpool + fc
    backbone = torch.nn.Sequential(*modules)
    backbone.eval()
    backbone.to(DEVICE)

    # Lớp Global Average Pooling riêng (cho vector 2048-d)
    avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    return backbone, avgpool


def get_transform():
    """Transform chuẩn cho ResNet-50."""
    return transforms.Compose([
        transforms.Resize((config.PRETRAINED_IMAGE_SIZE,
                           config.PRETRAINED_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def extract():
    print("=" * 60)
    print("EXTRACTING RESNET-50 FEATURES")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # 1. Thu thập Image IDs
    print("\n[1/3] Collecting image IDs...")
    image_ids = get_all_image_ids()

    # 2. Load model
    print("\n[2/3] Loading ResNet-50 Pretrained...")
    backbone, avgpool = build_resnet50()
    transform = get_transform()
    print("  ResNet-50 loaded successfully.")

    # 3. Trích xuất features
    print(f"\n[3/3] Extracting features for {len(image_ids)} images...")
    print(f"  Output: {config.FEATURES_H5}")
    print(f"  Batch size: {BATCH_SIZE}")

    with h5py.File(config.FEATURES_H5, "w") as h5f:
        # Tạo 2 group: pooled và spatial
        pooled_group = h5f.create_group("pooled")
        spatial_group = h5f.create_group("spatial")

        # Xử lý theo batch
        num_batches = (len(image_ids) + BATCH_SIZE - 1) // BATCH_SIZE
        extracted = 0
        skipped = 0

        for batch_idx in tqdm(range(num_batches), desc="Extracting"):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(image_ids))
            batch_ids = image_ids[start:end]

            # Load và transform ảnh trong batch
            batch_tensors = []
            valid_ids = []

            for img_id in batch_ids:
                img_path = os.path.join(config.IMAGES_DIR, f"{img_id}.jpg")
                if not os.path.exists(img_path):
                    skipped += 1
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform(img)
                    batch_tensors.append(img_tensor)
                    valid_ids.append(img_id)
                except Exception as e:
                    skipped += 1
                    continue

            if not batch_tensors:
                continue

            # Stack thành batch tensor
            batch_input = torch.stack(batch_tensors).to(DEVICE)  # (B, 3, 224, 224)

            # Forward pass (không tính gradient)
            with torch.no_grad():
                spatial_features = backbone(batch_input)  # (B, 2048, 7, 7)
                pooled_features = avgpool(spatial_features)  # (B, 2048, 1, 1)
                pooled_features = pooled_features.view(
                    pooled_features.size(0), -1
                )  # (B, 2048)

            # Chuyển spatial: (B, 2048, 7, 7) → (B, 49, 2048)
            B, C, H, W = spatial_features.size()
            spatial_flat = spatial_features.view(B, C, H * W).permute(
                0, 2, 1
            )  # (B, 49, 2048)

            # Lưu vào HDF5
            for i, img_id in enumerate(valid_ids):
                pooled_group.create_dataset(
                    img_id,
                    data=pooled_features[i].cpu().numpy(),
                    dtype="float32",
                )
                spatial_group.create_dataset(
                    img_id,
                    data=spatial_flat[i].cpu().numpy(),
                    dtype="float32",
                )

            extracted += len(valid_ids)

    # Thống kê
    file_size = os.path.getsize(config.FEATURES_H5) / (1024 ** 3)
    print(f"\n--- EXTRACTION COMPLETE ---")
    print(f"  Images extracted: {extracted}")
    print(f"  Images skipped:   {skipped}")
    print(f"  Output file:      {config.FEATURES_H5}")
    print(f"  File size:        {file_size:.2f} GB")
    print(f"  Pooled shape:     (2048,) per image")
    print(f"  Spatial shape:    (49, 2048) per image")


if __name__ == "__main__":
    extract()
