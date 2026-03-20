# VQA Seq2Seq Project Plan
# Visual Question Answering trên bộ dữ liệu GQA (Subset)

---

## 1. Mục tiêu dự án (Objective)
Xây dựng hệ thống VQA có khả năng **sinh câu trả lời đầy đủ** (fullAnswer) từ một hình ảnh và một câu hỏi đầu vào, sử dụng kiến trúc **Encoder-Decoder (Seq2Seq)** dựa trên CNN + LSTM.

So sánh hiệu năng giữa 6 kiến trúc khác nhau dựa trên 3 đặc điểm:
1. Không có vs Có cơ chế **Attention**
2. Train từ đầu (Scratch) vs Sử dụng **Pretrained** model
3. Trích xuất đặc trưng trước (Pre-extracted) vs Học đồng thời (End-to-End)

---

## 2. Dữ liệu sử dụng (Dataset)

### 2.1. Nguồn gốc
- Bộ dữ liệu **GQA (Visual Reasoning)** - Stanford
- Đã lọc subset từ bộ `balanced` gốc để tối ưu thời gian huấn luyện

### 2.2. Phân chia dữ liệu

| Tập dữ liệu | Tệp JSON | Số ảnh | Số câu hỏi | Mục đích |
|:---|:---|:---|:---|:---|
| **Train** | `gqa_data/subset/train_subset_25k.json` | 25,000 | 326,574 | Huấn luyện chính |
| **Validation** | `gqa_data/subset/val_subset_5k.json` | 5,000 | 64,525 | Theo dõi overfitting khi train |
| **Test** | `gqa_data/questions/testdev_balanced_questions.json` | 398 | 12,578 | Đánh giá cuối cùng (benchmark) |

### 2.3. Cấu trúc thư mục dữ liệu
```
D:\Deeplearning\gqa_data\
├── images\                  (148,854 ảnh - giữ nguyên toàn bộ, truy cập qua imageId)
├── subset\
│   ├── train_subset_25k.json
│   └── val_subset_5k.json
├── questions\
│   └── testdev_balanced_questions.json
├── images.zip               (Backup gốc)
├── questions1.2.zip          (Backup gốc)
└── sceneGraphs.zip           (Backup gốc - dùng cho nâng cao)
```

### 2.4. Cấu trúc mỗi mẫu dữ liệu (Sample)
Mỗi câu hỏi trong JSON chứa các trường chính:
- `imageId`: ID của ảnh trong thư mục `images/`
- `question`: Câu hỏi (Input cho Encoder)
- `fullAnswer`: Câu trả lời đầy đủ (Target cho Decoder)
- `answer`: Câu trả lời ngắn (Dùng để tính Accuracy)

**Ví dụ:**
```json
{
    "imageId": "2405722",
    "question": "What is this bird called?",
    "fullAnswer": "This is a parrot.",
    "answer": "parrot"
}
```

---

## 3. Ma trận thí nghiệm (Experiment Matrix) - 6 Mô hình

| Mô hình | CNN (Image Encoder) | Attention | Training Strategy | Ghi chú |
|:---|:---|:---|:---|:---|
| **Model 1** | Simple CNN 4 layers (Scratch) | Không | End-to-End (CNN + LSTM học song song) | Baseline thấp nhất |
| **Model 2** | ResNet-50 (Pretrained, Frozen) | Không | Pre-extract features → chỉ train LSTM | Nhanh nhất |
| **Model 3** | Simple CNN 4 layers (Scratch) | Có (Spatial) | End-to-End (CNN + LSTM học song song) | So sánh hiệu quả Attention |
| **Model 4** | ResNet-50 (Pretrained, Frozen) | Có (Spatial) | Pre-extract features → chỉ train LSTM + Attention | Kỳ vọng tốt nhất |
| **Model 5** | ResNet-50 (Pretrained, Unfrozen) | Không | End-to-End (Cập nhật cả ResNet + LSTM) | Đánh giá E2E Pretrained |
| **Model 6** | ResNet-50 (Pretrained, Unfrozen) | Có (Spatial) | End-to-End (Cập nhật cả ResNet + Attention) | Nặng nhất, kỳ vọng E2E |

### 3.1. Chiến lược huấn luyện hỗn hợp (Hybrid Training)
- **Model 1 & 3 (Scratch):** CNN và LSTM chạy **song song (End-to-End)** vì CNN cần cập nhật trọng số liên tục.
- **Model 2 & 4 (Pretrained):** **Trích xuất đặc trưng ảnh trước** bằng ResNet-50, lưu vào `.h5`, sau đó chỉ train LSTM/Attention. Tăng tốc ~20x.
- **Model 5 & 6 (Pretrained E2E):** ResNet-50 chạy forward pass ở mỗi epoch để fine-tune cùng mạng LSTM. Tốn chi phí bộ nhớ lớn.

---

## 4. Độ đo đánh giá (Evaluation Metrics)

### 4.1. Độ đo văn bản (Textual Metrics)
| Độ đo | Mô tả | Vai trò |
|:---|:---|:---|
| **Accuracy** | So khớp chính xác câu trả lời ngắn (`answer`) | Đo lường cơ bản |
| **BLEU-1,2,3,4** | Đo trùng lặp n-gram giữa câu sinh ra và đáp án | Đo độ chính xác từ ngữ |
| **ROUGE-L** | Dựa trên chuỗi con chung dài nhất (LCS) | Đo cấu trúc câu |

### 4.2. Độ đo ngữ nghĩa (Semantic Metrics)
| Độ đo | Mô tả | Vai trò |
|:---|:---|:---|
| **METEOR** | Xét cả từ đồng nghĩa và hình thái từ (stemming) | Đo độ linh hoạt ngôn ngữ |
| **CIDEr-D** | Ưu tiên từ mang thông tin cao (TF-IDF weighted) | Đo chất lượng mô tả ảnh |
| **BERTScore** | Sử dụng BERT embedding để tính Cosine Similarity | Đo hiểu ngữ nghĩa sâu |

---

## 5. Lộ trình thực hiện chi tiết (Step-by-Step Roadmap)

### Bước 1: Xây dựng Vocabulary & Data Pipeline
**Mục tiêu:** Chuyển đổi văn bản (câu hỏi + câu trả lời) thành dạng số để LSTM xử lý được.

**Công việc cụ thể:**
- [ ] Tạo lớp `Vocabulary` với các token đặc biệt: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- [ ] Quét toàn bộ 326k câu hỏi + câu trả lời trong tập train để xây dựng bộ từ điển
- [ ] Thiết lập ngưỡng tần suất tối thiểu (freq_threshold = 3) để loại bỏ từ hiếm
- [ ] Tạo lớp `GQADataset` kế thừa `torch.utils.data.Dataset`
- [ ] Viết hàm `collate_fn` để xử lý padding cho các batch có độ dài khác nhau
- [ ] Lưu Vocabulary vào tệp `vocab.pkl` để tái sử dụng

**Tệp đầu ra:**
- `d:\Deeplearning\data_utils.py` (Vocabulary + Dataset class)
- `d:\Deeplearning\gqa_data\subset\vocab.pkl` (Bộ từ điển đã xây dựng)

---

### Bước 2: Trích xuất đặc trưng ảnh (Feature Extraction)
**Mục tiêu:** Chạy ảnh qua CNN một lần duy nhất và lưu kết quả để tăng tốc huấn luyện.

**Công việc cụ thể:**
- [ ] Load ResNet-50 Pretrained (bỏ lớp FC cuối cùng)
- [ ] Với mỗi ảnh: resize → normalize → forward pass → lấy feature map
- [ ] **Cho Model 2 (No Attention):** Lưu vector 2048-d (sau Global Average Pooling)
- [ ] **Cho Model 4 (With Attention):** Lưu feature map 7x7x2048 (giữ nguyên spatial dimensions)
- [ ] Lưu toàn bộ features vào tệp HDF5 (`.h5`) để truy cập nhanh khi train
- [ ] Chạy extraction cho tất cả ảnh trong Train + Val + Test

**Tệp đầu ra:**
- `d:\Deeplearning\extract_features.py` (Script trích xuất)
- `d:\Deeplearning\gqa_data\subset\resnet50_features.h5` (Features đã trích xuất)

---

### Bước 3: Xây dựng kiến trúc mô hình (Model Architecture)
**Mục tiêu:** Cài đặt mã nguồn cho 6 kiến trúc mô hình.

**Công việc cụ thể:**

#### 3a. Các thành phần chung (Shared Components)
- [ ] `QuestionEncoder`: Embedding + Bi-LSTM để mã hóa câu hỏi
- [ ] `AnswerDecoder`: LSTM Decoder sinh câu trả lời từng từ (autoregressive)
- [ ] `AttentionModule`: Cơ chế Soft-Attention (dùng cho Model 3 & 4)

#### 3b. Các thành phần riêng biệt
- [ ] `ScratchCNN`: Mạng CNN 4 lớp đơn giản (cho Model 1 & 3)
- [ ] `PretrainedCNN`: ResNet-50 wrapper (cho Model 2 & 4 - chỉ dùng khi train End-to-End)
- [ ] `FusionModule`: Kết hợp đặc trưng ảnh + câu hỏi (Concatenation hoặc Element-wise Multiply)

#### 3c. Tổ hợp 6 mô hình hoàn chỉnh
- [ ] `VQAModel1_ScratchNoAtt`: ScratchCNN + QuestionEncoder + AnswerDecoder
- [ ] `VQAModel2_PretrainedNoAtt`: PreExtracted Features + QuestionEncoder + AnswerDecoder
- [ ] `VQAModel3_ScratchAtt`: ScratchCNN + QuestionEncoder + AttentionModule + AnswerDecoder
- [ ] `VQAModel4_PretrainedAtt`: PreExtracted Features + QuestionEncoder + AttentionModule + AnswerDecoder
- [ ] `VQAModel5_PretrainedEndToEndNoAtt`: PretrainedCNN + QuestionEncoder + AnswerDecoder
- [ ] `VQAModel6_PretrainedEndToEndAtt`: PretrainedCNN + QuestionEncoder + AttentionModule + AnswerDecoder

**Tệp đầu ra:**
- `d:\Deeplearning\models.py` (Tất cả kiến trúc mô hình)

---

### Bước 4: Huấn luyện (Training)
**Mục tiêu:** Huấn luyện tuần tự 6 mô hình và lưu kết quả.

**Tham số huấn luyện (Hyperparameters):**
- Loss Function: `CrossEntropyLoss` (ignore_index = `<PAD>`)
- Optimizer: `Adam` (lr = 5e-4 cho Scratch, lr = 1e-4 cho Pretrained)
- Batch Size: 64 (Scratch), 128-256 (Pretrained vì nhẹ hơn)
- Epochs: 15-25
- Teacher Forcing Ratio: Bắt đầu từ 1.0, giảm dần 0.05/epoch
- Early Stopping: Patience = 5 epochs (dừng nếu Val Loss không giảm)
- Gradient Clipping: max_norm = 5.0

**Thứ tự huấn luyện (ưu tiên mô hình nhanh trước):**
1. Model 2 (Pretrained, No Attention) → Nhanh nhất, dùng làm baseline
2. Model 4 (Pretrained, Attention) → So sánh hiệu quả Attention
3. Model 1 (Scratch, No Attention) → Chậm hơn, chạy sau
4. Model 3 (Scratch, Attention) → Chậm hơn
5. Model 5 (E2E Pretrained, No Attention) → Chậm nhất do ResNet forward
6. Model 6 (E2E Pretrained, Attention) → Chậm nhất do ResNet forward

**Cơ chế Logging & Checkpointing:**
- [ ] Sau mỗi epoch: Ghi `train_loss`, `val_loss`, `val_accuracy` vào `history.json`
- [ ] Lưu model checkpoint (`.pth`) khi `val_loss` đạt giá trị tốt nhất (best model)
- [ ] Lưu checkpoint cuối cùng (last model) để có thể resume training
- [ ] Tùy chọn: Tích hợp TensorBoard để theo dõi trực quan

**Tệp đầu ra:**
- `d:\Deeplearning\train.py` (Script huấn luyện chính)
- `d:\Deeplearning\results\model_X\checkpoints\best_model.pth`
- `d:\Deeplearning\results\model_X\logs\history.json`

---

### Bước 5: Đánh giá & Sinh câu trả lời (Evaluation & Inference)
**Mục tiêu:** Chạy mô hình trên tập Test và tính toán các độ đo.

**Công việc cụ thể:**
- [ ] Load best checkpoint cho từng mô hình
- [ ] Chạy inference trên tập `testdev_balanced_questions.json`
- [ ] Sử dụng **Greedy Decoding** (hoặc Beam Search k=3) để sinh câu trả lời
- [ ] Tính toán các độ đo: Accuracy, BLEU-1/2/3/4, METEOR, CIDEr-D, BERTScore
- [ ] Lưu kết quả predictions vào tệp JSON để kiểm tra thủ công

**Tệp đầu ra:**
- `d:\Deeplearning\evaluate.py` (Script đánh giá)
- `d:\Deeplearning\results\model_X\predictions.json`
- `d:\Deeplearning\results\model_X\metrics.json`

---

### Bước 6: So sánh & Trực quan hóa (Comparison & Visualization)
**Mục tiêu:** Tổng hợp kết quả và tạo báo cáo trực quan.

**Công việc cụ thể:**
- [ ] Vẽ biểu đồ Learning Curves (Train Loss vs Val Loss) cho 6 mô hình
- [ ] Tạo bảng so sánh tổng hợp tất cả độ đo
- [ ] Hiển thị mẫu dự đoán: Ảnh + Câu hỏi + Câu trả lời gốc vs Câu trả lời mô hình
- [ ] Với Model 3 & 4: Trực quan hóa Attention Map (vùng ảnh mà mô hình "nhìn" vào)

**Tệp đầu ra:**
- `d:\Deeplearning\visualize.py` (Script vẽ đồ thị)
- `d:\Deeplearning\results\comparison_chart.png`
- `d:\Deeplearning\results\attention_maps\` (Các ảnh attention heatmap)

---

## 6. Cấu trúc thư mục dự án cuối cùng (Final Project Structure)

```
D:\Deeplearning\
├── README.md                          (Tổng quan dự án)
├── VQA_Seq2Seq_Project_Plan.md        (Kế hoạch chi tiết - file này)
│
├── data_utils.py                      (Vocabulary + Dataset + DataLoader)
├── extract_features.py                (Trích xuất đặc trưng ảnh bằng ResNet-50)
├── models.py                          (Kiến trúc 6 mô hình)
├── train.py                           (Script huấn luyện chính)
├── evaluate.py                        (Script đánh giá trên tập Test)
├── visualize.py                       (Script vẽ đồ thị và so sánh)
│
├── gqa_data\
│   ├── images\                        (148,854 ảnh gốc)
│   ├── subset\
│   │   ├── train_subset_25k.json      (Train: 25k ảnh, 326k câu hỏi)
│   │   ├── val_subset_5k.json         (Val: 5k ảnh, 64k câu hỏi)
│   │   ├── vocab.pkl                  (Bộ từ điển)
│   │   └── resnet50_features.h5       (Đặc trưng ảnh đã trích xuất)
│   ├── questions\
│   │   └── testdev_balanced_questions.json  (Test: 398 ảnh, 12k câu hỏi)
│   ├── images.zip                     (Backup)
│   ├── questions1.2.zip               (Backup)
│   └── sceneGraphs.zip                (Backup - dùng cho mở rộng)
│
└── results\
    ├── model_1_scratch_no_att\
    │   ├── checkpoints\               (best_model.pth, last_model.pth)
    │   ├── logs\                      (history.json)
    │   ├── predictions.json
    │   └── metrics.json
    ├── model_2_pretrained_no_att\
    │   └── ...
    ├── model_3_scratch_att\
    │   └── ...
    ├── model_4_pretrained_att\
    │   └── ...
    ├── model_5_pretrained_e2e_no_att\
    │   └── ...
    ├── model_6_pretrained_e2e_att\
    │   └── ...
    ├── comparison_chart.png
    └── attention_maps\
```

---

## 7. Ghi chú bổ sung (Additional Notes)

### 7.1. Scene Graphs (Nâng cao - Tùy chọn)
- Tệp `sceneGraphs.zip` chứa thông tin cấu trúc vật thể và quan hệ trong ảnh
- Có thể tích hợp vào **Bước 2** (Object-level Feature Extraction) hoặc **Bước 3** (Graph Neural Network)
- Đề xuất: Thêm **Model 5 (Scene-Graph-Enhanced)** nếu có thời gian

### 7.2. Kỹ thuật tối ưu thời gian huấn luyện
- Model Scratch: Dùng ảnh 128x128 (thay vì 224x224) để giảm tải CNN
- Model Pretrained: Trích xuất features trước, chỉ train LSTM → nhanh hơn 20x
- Sử dụng Early Stopping (patience=5) để tránh lãng phí thời gian
- Lưu checkpoint để có thể resume training bất cứ lúc nào

### 7.3. Logging & Checkpoint
- Mỗi epoch: Ghi lại train_loss, val_loss, val_accuracy, val_bleu vào history.json
- Lưu best_model.pth khi val_loss đạt giá trị nhỏ nhất
- Lưu last_model.pth để có thể tiếp tục huấn luyện nếu bị gián đoạn
- Tùy chọn: Sử dụng TensorBoard để theo dõi trực quan real-time
