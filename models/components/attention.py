"""
attention.py - Spatial Attention mechanism (for Model 3 & 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    Soft Attention on CNN spatial feature map.
    Allows Decoder to "look" at different image regions at each decoding step.

    Input:  feature_map (B, num_pixels, feature_dim) + decoder_hidden (B, hidden_size)
    Output: context (B, feature_dim) + attention_weights (B, num_pixels)
    """

    def __init__(self, feature_dim, hidden_size, attention_dim):
        """
        Args:
            feature_dim: Image feature size (512 for Scratch, 2048 for ResNet)
            hidden_size: Decoder LSTM hidden state size
            attention_dim: Hidden layer size in attention network
        """
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, attention_dim)
        self.hidden_proj = nn.Linear(hidden_size, attention_dim)
        self.attention = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden):
        """
        Args:
            features: (B, num_pixels, feature_dim) - flattened feature map
            hidden:   (B, hidden_size) - current Decoder hidden state

        Returns:
            context: (B, feature_dim) - context vector with attention weights
            alpha:   (B, num_pixels) - attention weights (used for visualization)
        """
        # Project features and hidden into attention space
        feat_proj = self.feature_proj(features)          # (B, num_pixels, att_dim)
        hidden_proj = self.hidden_proj(hidden).unsqueeze(1)  # (B, 1, att_dim)

        # Compute attention scores
        energy = torch.tanh(feat_proj + hidden_proj)     # (B, num_pixels, att_dim)
        scores = self.attention(energy).squeeze(2)        # (B, num_pixels)

        # Softmax to get weights
        alpha = F.softmax(scores, dim=1)                  # (B, num_pixels)

        # Multiply weights with features
        context = (features * alpha.unsqueeze(2)).sum(dim=1)  # (B, feature_dim)

        return context, alpha


class AttentionDecoder(nn.Module):
    """
    LSTM Decoder with Spatial Attention mechanism.
    At each decoding step, Decoder "looks" at the most relevant image region.

    Used for Model 3 & Model 4.
    """

    def __init__(self, vocab_size, embed_size, hidden_size,
                 feature_dim, question_dim, attention_dim=256,
                 num_layers=1, dropout=0.3):
        """
        Args:
            vocab_size: Vocabulary size
            embed_size: Embedding size
            hidden_size: LSTM hidden state size
            feature_dim: Image feature size (512 or 2048)
            question_dim: Question context vector size
            attention_dim: Hidden layer size in attention
            num_layers: Number of LSTM layers (default 1 for Attention Decoder)
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.attention = SpatialAttention(feature_dim, hidden_size, attention_dim)

        # LSTM input = embedding + attention_context + question_context
        self.lstm = nn.LSTM(
            input_size=embed_size + feature_dim + question_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Project question context → initial hidden state
        self.q_to_hidden = nn.Linear(question_dim, hidden_size * num_layers)
        self.q_to_cell = nn.Linear(question_dim, hidden_size * num_layers)

        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _init_hidden(self, question_context):
        """Initialize hidden state from question context."""
        batch_size = question_context.size(0)

        hidden = self.q_to_hidden(question_context)
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_size)
        hidden = hidden.permute(1, 0, 2).contiguous()

        cell = self.q_to_cell(question_context)
        cell = cell.view(batch_size, self.num_layers, self.hidden_size)
        cell = cell.permute(1, 0, 2).contiguous()

        return hidden, cell

    def forward(self, spatial_features, question_context, targets,
                teacher_forcing_ratio=1.0):
        """
        Forward pass during training.

        Args:
            spatial_features: (B, num_pixels, feature_dim)
            question_context: (B, question_dim)
            targets: (B, max_len)
            teacher_forcing_ratio: Teacher Forcing ratio

        Returns:
            outputs: (B, max_len, vocab_size)
            alphas:  List[(B, num_pixels)] - attention weights for visualization
        """
        batch_size = targets.size(0)
        max_len = targets.size(1)
        vocab_size = self.fc_out.out_features

        hidden, cell = self._init_hidden(question_context)
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(targets.device)
        alphas = []

        input_token = targets[:, 0]  # <SOS>

        for t in range(1, max_len):
            embedded = self.dropout(self.embedding(input_token))  # (B, E)

            # Attention: "look" at relevant region
            att_context, alpha = self.attention(
                spatial_features, hidden[-1]
            )  # att_context: (B, feat_dim), alpha: (B, num_pixels)
            alphas.append(alpha)

            # Concatenate embedding + attention context + question context
            lstm_input = torch.cat(
                [embedded, att_context, question_context], dim=1
            ).unsqueeze(1)  # (B, 1, E + feat_dim + q_dim)

            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            prediction = self.fc_out(output.squeeze(1))  # (B, vocab_size)
            outputs[:, t, :] = prediction

            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = targets[:, t]
            else:
                input_token = prediction.argmax(dim=1)

        return outputs, alphas

    def generate(self, spatial_features, question_context,
                 sos_idx, eos_idx, max_len=30):
        """
        Generate answer during inference.

        Returns:
            generated: (B, max_len)
            alphas: List[(B, num_pixels)]
        """
        batch_size = spatial_features.size(0)
        hidden, cell = self._init_hidden(question_context)

        generated = []
        alphas = []
        input_token = torch.full((batch_size,), sos_idx,
                                 dtype=torch.long, device=spatial_features.device)

        for _ in range(max_len):
            embedded = self.embedding(input_token)
            att_context, alpha = self.attention(spatial_features, hidden[-1])
            alphas.append(alpha)

            lstm_input = torch.cat(
                [embedded, att_context, question_context], dim=1
            ).unsqueeze(1)

            output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            prediction = self.fc_out(output.squeeze(1))
            predicted_token = prediction.argmax(dim=1)
            generated.append(predicted_token.unsqueeze(1))
            input_token = predicted_token

        return torch.cat(generated, dim=1), alphas
