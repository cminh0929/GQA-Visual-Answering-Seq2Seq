"""
Model 6: Pretrained ResNet-50 (Frozen, End-to-End) + LSTM Seq2Seq (With Spatial Attention)
- CNN: ResNet-50 Pretrained (Frozen) → chạy end-to-end mỗi batch → spatial feature map
- Question Encoder: Bi-LSTM
- Decoder: Attention LSTM (Spatial Attention trên feature map 7x7)
- Training: End-to-End pipeline (CNN frozen, chỉ train LSTM + Attention)

So sánh với Model 4: cùng CNN frozen, cùng attention decoder, nhưng end-to-end thay vì pre-extracted
So sánh với Model 3: cùng pipeline end-to-end, nhưng pretrained CNN thay vì scratch
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.components.pretrained_cnn import PretrainedCNN
from models.components.question_encoder import QuestionEncoder
from models.components.attention import AttentionDecoder
import config


class VQAModel6_PretrainedEndToEndAtt(nn.Module):
    """
    Model 6: Pretrained ResNet-50 (Frozen, End-to-End) + LSTM (With Spatial Attention)
    Input:  ảnh (B, 3, 224, 224) + câu hỏi (B, seq_len)
    Output: câu trả lời (B, max_len, vocab_size) + attention weights

    So sánh với Model 4 (pre-extracted) để đánh giá ảnh hưởng training strategy.
    """

    def __init__(self, vocab_size):
        super().__init__()

        # Image Encoder: ResNet-50 Pretrained (Frozen) → spatial feature map
        self.image_encoder = PretrainedCNN(return_spatial=True)

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

    def forward(self, images, questions, answers, teacher_forcing_ratio=1.0):
        """
        Args:
            images: (B, 3, 224, 224) - ảnh gốc (end-to-end)
            questions: (B, q_len)
            answers: (B, a_len)
        Returns:
            outputs: (B, a_len, vocab_size)
            alphas: List of attention weights
        """
        # CNN forward (frozen, no grad) → spatial features
        spatial = self.image_encoder(images)  # (B, 2048, 7, 7)
        B, C, H, W = spatial.size()
        spatial = spatial.view(B, C, H * W).permute(0, 2, 1)  # (B, 49, 2048)

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
        return self.decoder.generate(
            spatial, q_context, sos_idx, eos_idx, max_len
        )
