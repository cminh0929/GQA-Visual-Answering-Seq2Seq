import base64
import urllib.request
import json

mermaid_code = """graph LR
    subgraph Vision_Encoder ["1. Vision Encoder"]
        direction TB
        subgraph Model_1_3 ["Model 1/3 (Scratch CNN)"]
            direction TB
            C1["Conv(3 → 64)<br>BatchNorm + ReLU + MaxPool"] --> C2["Conv(64 → 128)<br>BatchNorm + ReLU + MaxPool"] --> C3["Conv(128 → 256)<br>BatchNorm + ReLU + MaxPool"] --> C4["Conv(256 → 512)<br>BatchNorm + ReLU + MaxPool"]
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
    subgraph Question_Encoder ["2. Question Encoder"]
        direction TB
        Q1["Input Question"] --> Q2["Embedding (2589 → 256)"] --> Q3["Bi-LSTM (2 Layers, Hidden: 256)"] --> Q4["Dropout (0.3)"] --> Q5["Question Vector (1024-dim)"]
    end
    subgraph Answer_Decoder ["3. Answer Decoder"]
        direction TB
        D1["Input: Nhận Visual Vector<br>& Question Vector"] --> D2["Projection: Linear(1024 → 512)<br>↓<br>ReLU"] --> D3["Core: LSTM (2 Layers, Hidden: 256)"] --> D4["Output: Linear(256 → 2589)<br>↓<br>Softmax"]
    end
    No_Att ==>|Visual Vector| D1
    Q5 ==>|Question Vector| D1
    Att -.->|Grid Tensor 7x7| Attention_Mechanism(("Attention Hook"))
    D3 -.->|LSTM Hidden State| Attention_Mechanism
    Attention_Mechanism -.->|Context Vector| D3
    style Question_Encoder fill:#f4f4f9,stroke:#333
    style Answer_Decoder fill:#fff9f9,stroke:#333
    style Vision_Encoder fill:#f9fff9,stroke:#333
    style Attention_Mechanism fill:#ffcc99,stroke:#e65c00,stroke-width:2px
"""

state = {
  "code": mermaid_code,
  "mermaid": {"theme": "default"}
}

encoded = base64.urlsafe_b64encode(json.dumps(state).encode('utf-8')).decode('utf-8')
url = f"https://mermaid.ink/img/{encoded}"

print(f"Downloading from {url}")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
with urllib.request.urlopen(req) as response:
    with open('vqa_architecture.png', 'wb') as f:
        f.write(response.read())

print("Download completed! Saved to vqa_architecture.png")
