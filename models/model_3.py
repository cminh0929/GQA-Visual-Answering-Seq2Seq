"""
Model 3: Scratch CNN + LSTM Seq2Seq (With Spatial Attention)
- CNN: 4-layer CNN train từ đầu → spatial feature map
- Question Encoder: Bi-LSTM
- Decoder: Attention LSTM (Spatial Attention trên feature map)
- Training: End-to-End (CNN + LSTM song song)
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.components.scratch_cnn import ScratchCNN
from models.components.question_encoder import QuestionEncoder
from models.components.attention import AttentionDecoder
import config


class VQAModel3_ScratchAtt(nn.Module):
    """
    Model 3: Scratch CNN + LSTM Seq2Seq (With Spatial Attention)
    Input:  ảnh (B, 3, 128, 128) + câu hỏi (B, seq_len)
    Output: câu trả lời (B, max_len, vocab_size) + attention weights
    """

    def __init__(self, vocab_size):
        super().__init__()

        # Image Encoder: CNN Scratch → spatial feature map (B, 512, 8, 8)
        self.image_encoder = ScratchCNN(
            out_channels=config.SCRATCH_CNN_OUT,
            return_spatial=True
        )

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
            feature_dim=config.SCRATCH_CNN_OUT,   # 512
            question_dim=config.HIDDEN_SIZE * 2,   # Bi-LSTM output
            attention_dim=256,
            num_layers=1,
            dropout=config.DROPOUT,
        )

    def forward(self, images, questions, answers, teacher_forcing_ratio=1.0):
        """
        Args:
            images: (B, 3, 128, 128)
            questions: (B, q_len)
            answers: (B, a_len)
        Returns:
            outputs: (B, a_len, vocab_size)
            alphas: List of attention weights
        """
        # Encode image → spatial features
        spatial = self.image_encoder(images)  # (B, 512, 8, 8)
        B, C, H, W = spatial.size()
        spatial = spatial.view(B, C, H * W).permute(0, 2, 1)  # (B, 64, 512)

        # Encode question
        _, q_context = self.question_encoder(questions)  # (B, H*2)

        # Decode with attention
        outputs, alphas = self.decoder(
            spatial, q_context, answers, teacher_forcing_ratio
        )
        return outputs, alphas

    def generate(self, images, questions, sos_idx, eos_idx, max_len=30):
        """Sinh câu trả lời khi inference."""
        spatial = self.image_encoder(images)
        B, C, H, W = spatial.size()
        spatial = spatial.view(B, C, H * W).permute(0, 2, 1)
        _, q_context = self.question_encoder(questions)
        return self.decoder.generate(spatial, q_context, sos_idx, eos_idx, max_len)
