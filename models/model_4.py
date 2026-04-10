"""
Model 4: Pretrained ResNet-50 + LSTM Seq2Seq (With Spatial Attention)
- CNN: ResNet-50 Pretrained (Frozen) → spatial feature map 7x7x2048
- Question Encoder: Bi-LSTM
- Decoder: Attention LSTM (Spatial Attention on feature map)
- Training: Only train LSTM + Attention (features pre-extracted)
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.components.question_encoder import QuestionEncoder
from models.components.attention import AttentionDecoder
import vqa_config as config


class VQAModel4_PretrainedAtt(nn.Module):
    """
    Model 4: Pretrained ResNet-50 + LSTM Seq2Seq (With Spatial Attention)
    Input: Pre-extracted spatial features (B, 49, 2048) + Question (B, seq_len)
    Output: Answer (B, max_len, vocab_size) + attention weights

    This model is expected to provide the best results.
    """

    def __init__(self, vocab_size):
        super().__init__()

        # Question Encoder: Bi-LSTM
        self.question_encoder = QuestionEncoder(
            vocab_size=vocab_size,
            embed_size=config.EMBED_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LSTM_LAYERS,
            dropout=config.DROPOUT,
        )

        # Attention Decoder
        self.decoder = AttentionDecoder(
            vocab_size=vocab_size,
            embed_size=config.EMBED_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            feature_dim=config.RESNET_FEATURE_DIM,  # 2048
            question_dim=config.HIDDEN_SIZE * 2,     # Bi-LSTM output
            attention_dim=256,
            num_layers=1,
            dropout=config.DROPOUT,
        )

    def forward(self, spatial_features, questions, answers,
                teacher_forcing_ratio=1.0):
        """
        Args:
            spatial_features: (B, 49, 2048) - pre-extracted ResNet-50 spatial
            questions: (B, q_len)
            answers: (B, a_len)
        Returns:
            outputs: (B, a_len, vocab_size)
            alphas: List of attention weights (used for visualization)
        """
        # Encode question
        _, q_context = self.question_encoder(questions)  # (B, H*2)

        # Decode with attention
        outputs, alphas = self.decoder(
            spatial_features, q_context, answers, teacher_forcing_ratio
        )
        return outputs, alphas

    def generate(self, spatial_features, questions,
                 sos_idx, eos_idx, max_len=30):
        """Answer generation for inference."""
        _, q_context = self.question_encoder(questions)
        return self.decoder.generate(
            spatial_features, q_context, sos_idx, eos_idx, max_len
        )
