# VQA Visual Reasoning với Seq2Seq (LSTM & CNN) - GQA Subset

Dự án xây dựng và so sánh 6 kiến trúc mô hình VQA Seq2Seq trên bộ dữ liệu GQA thu gọn.

## Thành phần dữ liệu

| Tập | Tệp JSON | Ảnh | Câu hỏi | Mục đích |
|:---|:---|:---|:---|:---|
| **Train** | `gqa_data/subset/train_subset_25k.json` | 25,000 | 326,574 | Huấn luyện |
| **Val** | `gqa_data/subset/val_subset_5k.json` | 5,000 | 64,525 | Kiểm định |
| **Test** | `gqa_data/questions/testdev_balanced_questions.json` | 398 | 12,578 | Đánh giá cuối |

- Toàn bộ 148,854 ảnh được giữ nguyên trong `gqa_data/images/`, truy cập qua `imageId`.
- Các tệp `.zip` gốc được bảo toàn để có thể giải nén lại khi cần.

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
