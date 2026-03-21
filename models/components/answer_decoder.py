"""
answer_decoder.py - LSTM Decoder for answer generation (No Attention version)
Used for Model 1 & Model 2
"""

import torch
import torch.nn as nn


class AnswerDecoder(nn.Module):
    """
    LSTM Decoder generates answers word by word.
    Takes context vector (from CNN + Question) as initial hidden state,
    then autoregressively generates each token.

    Input:  context_vector (B, context_dim)
    Output: (B, seq_len, vocab_size) - probability for each word at each step
    """

    def __init__(self, vocab_size, embed_size, hidden_size,
                 context_dim, num_layers=2, dropout=0.3):
        """
        Args:
            vocab_size: Vocabulary size
            embed_size: Embedding size
            hidden_size: LSTM hidden state size
            context_dim: Input context vector size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # Project context vector → hidden state of appropriate size
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
        Initialize hidden state from context vector.
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
        Forward pass during training (with Teacher Forcing).

        Args:
            context: (B, context_dim) - combined context vector
            targets: (B, max_len) - true answer (ground truth)
            teacher_forcing_ratio: Probability of using true word instead of predicted word

        Returns:
            outputs: (B, max_len, vocab_size) - log-probabilities for each word
        """
        batch_size = context.size(0)
        max_len = targets.size(1)
        vocab_size = self.fc_out.out_features

        # Initialize hidden state from context
        hidden, cell = self._init_hidden(context)

        # Tensor to store results
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(context.device)

        # First token is <SOS> (from targets[:, 0])
        input_token = targets[:, 0].unsqueeze(1)  # (B, 1)

        for t in range(1, max_len):
            embedded = self.dropout(self.embedding(input_token))  # (B, 1, E)
            output, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # (B, 1, H)
            prediction = self.fc_out(output.squeeze(1))  # (B, vocab_size)
            outputs[:, t, :] = prediction

            # Teacher Forcing: use true word or predicted word
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = targets[:, t].unsqueeze(1)  # True word
            else:
                input_token = prediction.argmax(dim=1).unsqueeze(1)  # Predicted word

        return outputs

    def generate(self, context, sos_idx, eos_idx, max_len=30):
        """
        Generate answer during inference (no Teacher Forcing).

        Args:
            context: (B, context_dim)
            sos_idx: Index of <SOS> token
            eos_idx: Index of <EOS> token
            max_len: Maximum answer length

        Returns:
            generated: (B, max_len) - indices of generated words
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
