"""
Model 2: Pretrained ResNet-50 + LSTM Seq2Seq (No Attention)
- CNN: ResNet-50 Pretrained (Frozen) → Pre-extracted features
- Question Encoder: Bi-LSTM
- Decoder: LSTM (No Attention)
- Training: Only train LSTM (features pre-extracted into .h5)
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.components.question_encoder import QuestionEncoder
from models.components.answer_decoder import AnswerDecoder
import vqa_config as config


class VQAModel2_PretrainedNoAtt(nn.Module):
    """
    Model 2: Pretrained ResNet-50 + LSTM Seq2Seq (No Attention)
    Input: Pre-extracted feature (B, 2048) + Question (B, seq_len)
    Output: Answer (B, max_len, vocab_size)
    """

    def __init__(self, vocab_size):
        super().__init__()

        # Project 2048-d feature to embed_size dimensions to reduce size
        self.image_proj = nn.Sequential(
            nn.Linear(config.RESNET_FEATURE_DIM, config.HIDDEN_SIZE * 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
        )

        # Question Encoder: Bi-LSTM
        self.question_encoder = QuestionEncoder(
            vocab_size=vocab_size,
            embed_size=config.EMBED_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LSTM_LAYERS,
            dropout=config.DROPOUT,
        )

        # Fusion: image_projected + question_context
        context_dim = config.HIDDEN_SIZE * 2 + config.HIDDEN_SIZE * 2

        # Answer Decoder
        self.decoder = AnswerDecoder(
            vocab_size=vocab_size,
            embed_size=config.EMBED_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            context_dim=context_dim,
            num_layers=config.NUM_LSTM_LAYERS,
            dropout=config.DROPOUT,
        )

    def forward(self, features, questions, answers, teacher_forcing_ratio=1.0):
        """
        Args:
            features: (B, 2048) - pre-extracted ResNet-50 pooled features
            questions: (B, q_len)
            answers: (B, a_len)
        Returns:
            outputs: (B, a_len, vocab_size)
        """
        img_proj = self.image_proj(features)                 # (B, H*2)
        _, q_context = self.question_encoder(questions)       # (B, H*2)
        context = torch.cat([img_proj, q_context], dim=1)     # (B, H*4)
        outputs = self.decoder(context, answers, teacher_forcing_ratio)
        return outputs

    def generate(self, features, questions, sos_idx, eos_idx, max_len=30):
        """Answer generation for inference."""
        img_proj = self.image_proj(features)
        _, q_context = self.question_encoder(questions)
        context = torch.cat([img_proj, q_context], dim=1)
        return self.decoder.generate(context, sos_idx, eos_idx, max_len)
