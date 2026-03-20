"""
answer_decoder.py - LSTM Decoder sinh câu trả lời (No Attention version)
Dùng cho Model 1 & Model 2
"""

import torch
import torch.nn as nn


class AnswerDecoder(nn.Module):
    """
    LSTM Decoder sinh câu trả lời từng từ một.
    Nhận context vector (từ CNN + Question) làm hidden state ban đầu,
    rồi autoregressive sinh ra từng token.

    Input:  context_vector (B, context_dim)
    Output: (B, seq_len, vocab_size) - xác suất cho từng từ tại mỗi bước
    """

    def __init__(self, vocab_size, embed_size, hidden_size,
                 context_dim, num_layers=2, dropout=0.3):
        """
        Args:
            vocab_size: Kích thước bộ từ điển
            embed_size: Kích thước embedding
            hidden_size: Kích thước hidden state LSTM
            context_dim: Kích thước context vector đầu vào
            num_layers: Số lớp LSTM
            dropout: Tỷ lệ dropout
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # Chiếu context vector → hidden state kích thước phù hợp
        self.context_to_hidden = nn.Linear(context_dim, hidden_size * num_layers)
        self.context_to_cell = nn.Linear(context_dim, hidden_size * num_layers)

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _init_hidden(self, context):
        """
        Khởi tạo hidden state từ context vector.
        Args:
            context: (B, context_dim)
        Returns:
            hidden: (num_layers, B, hidden_size)
            cell:   (num_layers, B, hidden_size)
        """
        batch_size = context.size(0)

        hidden = self.context_to_hidden(context)  # (B, H * num_layers)
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_size)
        hidden = hidden.permute(1, 0, 2).contiguous()  # (num_layers, B, H)

        cell = self.context_to_cell(context)  # (B, H * num_layers)
        cell = cell.view(batch_size, self.num_layers, self.hidden_size)
        cell = cell.permute(1, 0, 2).contiguous()  # (num_layers, B, H)

        return hidden, cell

    def forward(self, context, targets, teacher_forcing_ratio=1.0):
        """
        Forward pass khi huấn luyện (với Teacher Forcing).

        Args:
            context: (B, context_dim) - vector ngữ cảnh tổng hợp
            targets: (B, max_len) - câu trả lời đúng (ground truth)
            teacher_forcing_ratio: Xác suất sử dụng từ đúng thay vì từ dự đoán

        Returns:
            outputs: (B, max_len, vocab_size) - log-probabilities cho từng từ
        """
        batch_size = context.size(0)
        max_len = targets.size(1)
        vocab_size = self.fc_out.out_features

        # Khởi tạo hidden state từ context
        hidden, cell = self._init_hidden(context)

        # Tensor lưu kết quả
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(context.device)

        # Token đầu tiên là <SOS> (lấy từ targets[:, 0])
        input_token = targets[:, 0].unsqueeze(1)  # (B, 1)

        for t in range(1, max_len):
            embedded = self.dropout(self.embedding(input_token))  # (B, 1, E)
            output, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # (B, 1, H)
            prediction = self.fc_out(output.squeeze(1))  # (B, vocab_size)
            outputs[:, t, :] = prediction

            # Teacher Forcing: dùng từ đúng hoặc từ dự đoán
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = targets[:, t].unsqueeze(1)  # Từ đúng
            else:
                input_token = prediction.argmax(dim=1).unsqueeze(1)  # Từ dự đoán

        return outputs

    def generate(self, context, sos_idx, eos_idx, max_len=30):
        """
        Sinh câu trả lời khi inference (không có Teacher Forcing).

        Args:
            context: (B, context_dim)
            sos_idx: Index của token <SOS>
            eos_idx: Index của token <EOS>
            max_len: Độ dài tối đa câu trả lời

        Returns:
            generated: (B, max_len) - indices của các từ được sinh ra
        """
        batch_size = context.size(0)
        hidden, cell = self._init_hidden(context)

        generated = []
        input_token = torch.full((batch_size, 1), sos_idx,
                                 dtype=torch.long, device=context.device)

        for _ in range(max_len):
            embedded = self.embedding(input_token)  # (B, 1, E)
            output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
            prediction = self.fc_out(output.squeeze(1))  # (B, vocab_size)
            predicted_token = prediction.argmax(dim=1)  # (B,)
            generated.append(predicted_token.unsqueeze(1))
            input_token = predicted_token.unsqueeze(1)

        return torch.cat(generated, dim=1)  # (B, max_len)
