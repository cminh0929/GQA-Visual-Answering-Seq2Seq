# VQA Visual Reasoning với Seq2Seq (LSTM & CNN) - GQA Subset

Dự án xây dựng và so sánh 6 kiến trúc mô hình VQA Seq2Seq trên bộ dữ liệu GQA thu gọn.

## Thành phần dữ liệu

Toàn bộ dữ liệu của dự án đã được tinh gọn và xuất bản trên Kaggle để tiết kiệm thời gian và dễ dàng tái tạo kết quả:

1. **[GQA VQA Subset](https://www.kaggle.com/datasets/minhngcng3/gqa-vqa-subset)**:
   - Chứa các file `json` (Train 25k, Val 5k, Testdev).
   - Đặc biệt: **Đã bao gồm sẵn file trích xuất đặc trưng `resnet50_features.h5` và từ điển `vocab.pkl`**. Phục vụ huấn luyện ngay các mô hình Pre-extracted (Model 2, 4) mà không cần tải ảnh gốc và không mất phần cứng/thời gian để chạy trích xuất lại đặc trưng.
2. **[GQA Images Subset](https://www.kaggle.com/datasets/minhngcng3/gqa-images-subset)**:
   - Chứa tập ảnh gốc cần thiết nếu bạn muốn huấn luyện các mô hình End-to-End (Model 1, 3, 5, 6) từ đầu bằng ảnh RAW.

| Tập | Số lượng Ảnh | Số câu hỏi | Mục đích |
|:---|:---|:---|:---|
| **Train** | 25,000 | 326,574 | Huấn luyện mô hình |
| **Val** | 5,000 | 64,525 | Kiểm định trong lúc huấn luyện |
| **Test** | 398 | 12,578 | Chạy đánh giá (Inference) báo cáo |

## Hướng dẫn cài đặt Dataset (Dành cho người mới)

Để ứng dụng / mô hình chạy mượt mà ngay trên máy tính cá nhân, bạn phải cung cấp thư mục `gqa_data` theo đúng cấu trúc mà file `config.py` yêu cầu. Bạn thực hiện theo thứ tự sau (lưu ý tất cả file đã nằm trong `.gitignore`, sẽ không bị commit lên kho này):

**Bước 1:** Tải từ Kaggle bằng API (hoặc tải thuỷ công bằng trình duyệt rồi giải nén)
```bash
# Tải subset JSON, Từ điển & file H5 (Bắt buộc)
kaggle datasets download minhngcng3/gqa-vqa-subset

# Tải bộ ảnh gốc (Chỉ cần tải nếu bạn muốn chạy Model 1, 3, 5, 6)
kaggle datasets download minhngcng3/gqa-images-subset
```

**Bước 2:** Tổ chức thư mục
Khởi tạo và xả nén các thư mục tải về đưa vào gốc dự án sao cho kết quả cuối cùng phải giống hệt sơ đồ cây thư mục đính kèm:
```text
Deeplearning/
└── gqa_data/
    ├── subset/
    │   ├── train_subset_25k.json
    │   ├── val_subset_5k.json
    │   ├── resnet50_features.h5
    │   └── vocab.pkl
    ├── questions/
    │   └── testdev_balanced_questions.json
    └── images/
        ├── 2374353.jpg
        ├── ... (hàng nghìn file ảnh khác)
```
*Gợi ý: Nếu bạn chỉ hứng thú nghiệm chứng độ chính xác mà không có GPU khủng, hãy thiết lập Model 2 (tải json + file h5, bỏ qua phần images) là đã có thể chạy huấn luyện trong môi trường CPU một cách hoàn hảo.*

## Kiến trúc mô hình

| Mô hình | CNN | Attention | Training |
|:---|:---|:---|:---|
| **Model 1** | Scratch 4-layer | Không | End-to-End |
| **Model 2** | ResNet-50 Pretrained | Không | Pre-extract features |
| **Model 3** | Scratch 4-layer | Spatial Attention | End-to-End |
| **Model 4** | ResNet-50 Pretrained | Spatial Attention | Pre-extract features |
| **Model 5** | ResNet-50 Pretrained | Không | End-to-End |
| **Model 6** | ResNet-50 Pretrained | Spatial Attention | End-to-End |

## Độ đo đánh giá
- **Textual:** Accuracy, BLEU-1/2/3/4, ROUGE-L
- **Semantic:** METEOR, CIDEr-D, BERTScore

## Lộ trình
1. Xây dựng Vocabulary & Data Pipeline
2. Trích xuất đặc trưng ảnh (ResNet-50) → lưu `.h5`
3. Xây dựng kiến trúc mô hình (6 biến thể)
4. Huấn luyện (ưu tiên Pretrained trước, Scratch sau)
5. Đánh giá trên tập Test (testdev)
6. So sánh & Trực quan hóa kết quả

> Chi tiết đầy đủ: xem `VQA_Seq2Seq_Project_Plan.md`
