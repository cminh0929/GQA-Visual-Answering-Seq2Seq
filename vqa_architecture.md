# Kiến trúc tổng quan mô hình VQA (Visual Question Answering)

Mô hình được chia làm 3 khối chính: Xử lý hình ảnh (Vision Encoder), Nhúng câu hỏi (Question Encoder) và Giải mã câu trả lời (Answer Decoder). Sơ đồ dưới đây trình bày luồng đi của dữ liệu bao gồm cả cơ chế giải mã thông thường lẫn có Attention.

```mermaid
graph LR
    %% --- Khối 1: Vision Encoder ---
    subgraph Vision_Encoder ["1. Vision Encoder"]
        direction TB
        
        subgraph Model_1_3 ["Model 1/3 (Scratch CNN)"]
            direction TB
            C1["Conv(3 → 64)<br>BatchNorm + ReLU + MaxPool"] -->
            C2["Conv(64 → 128)<br>BatchNorm + ReLU + MaxPool"] -->
            C3["Conv(128 → 256)<br>BatchNorm + ReLU + MaxPool"] -->
            C4["Conv(256 → 512)<br>BatchNorm + ReLU + MaxPool"]
        end
        
        subgraph Model_5_6 ["Model 5/6 (ResNet-50)"]
            direction TB
            R1["ResNet-50 Backbone<br>4 Bottleneck Stages<br>↓<br>Output: 2048-dim feature map"]
        end
        
        No_Att["Nhánh No-Att<br>AdaptiveAvgPool2d(1x1)<br>↓<br>Vector (2048)"]
        Att["Nhánh Att<br>Keep Grid (7x7)<br>↓<br>Tensor (7x7x2048)"]
        
        C4 -.-> No_Att
        C4 -.-> Att
        R1 -.-> No_Att
        R1 -.-> Att
    end
    
    %% --- Khối 2: Question Encoder ---
    subgraph Question_Encoder ["2. Question Encoder"]
        direction TB
        Q1["Input Question"] -->
        Q2["Embedding (2589 → 256)"] -->
        Q3["Bi-LSTM (2 Layers, Hidden: 256)"] -->
        Q4["Dropout (0.3)"] -->
        Q5["Question Vector (1024-dim)"]
    end
    
    %% --- Khối 3: Answer Decoder ---
    subgraph Answer_Decoder ["3. Answer Decoder"]
        direction TB
        D1["Input: Nhận Visual Vector<br>& Question Vector"] -->
        D2["Projection: Linear(1024 → 512)<br>↓<br>ReLU"] -->
        D3["Core: LSTM (2 Layers, Hidden: 256)"] -->
        D4["Output: Linear(256 → 2589)<br>↓<br>Softmax"]
    end
    
    %% --- Luồng Dữ Liệu ---
    No_Att ==>|Visual Vector| D1
    Q5 ==>|Question Vector| D1
    
    Att -.->|Grid Tensor 7x7| Attention_Mechanism(("Attention Hook"))
    D3 -.->|LSTM Hidden State| Attention_Mechanism
    Attention_Mechanism -.->|Context Vector| D3

    %% Định dạng màu sắc
    style Question_Encoder fill:#f4f4f9,stroke:#333
    style Answer_Decoder fill:#fff9f9,stroke:#333
    style Vision_Encoder fill:#f9fff9,stroke:#333
    style Attention_Mechanism fill:#ffcc99,stroke:#e65c00,stroke-width:2px
```
