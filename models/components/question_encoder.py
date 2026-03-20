"""
question_encoder.py - LSTM Encoder cho câu hỏi (dùng chung cho cả 4 model)
"""

import torch
import torch.nn as nn


class QuestionEncoder(nn.Module):
    """
    Mã hóa câu hỏi bằng Word Embedding + Bi-LSTM.
    Input:  (B, seq_len) - indices của câu hỏi
    Output: (B, hidden_size * 2) - context vector (nối 2 chiều LSTM)
            hoặc full outputs cho Attention
    """

    def __init__(self, vocab_size, embed_size, hidden_size,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, seq_len) - câu hỏi đã numericalize
        Returns:
            outputs: (B, seq_len, hidden_size*2) - output tại mỗi bước
            hidden:  (B, hidden_size*2) - context vector cuối cùng
        """
        embedded = self.dropout(self.embedding(x))    # (B, seq_len, embed_size)
        outputs, (hidden, cell) = self.lstm(embedded)  # outputs: (B, seq_len, H*2)

        # Nối hidden state cuối cùng của 2 chiều (forward + backward)
        # hidden shape: (num_layers*2, B, hidden_size)
        hidden_fwd = hidden[-2]  # (B, hidden_size)
        hidden_bwd = hidden[-1]  # (B, hidden_size)
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)  # (B, H*2)

        return outputs, hidden_cat
