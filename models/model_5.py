"""
Model 5: Pretrained ResNet-50 (Frozen, End-to-End) + LSTM Seq2Seq (No Attention)
- CNN: ResNet-50 Pretrained (Frozen) → runs end-to-end each batch
- Question Encoder: Bi-LSTM
- Decoder: LSTM (No Attention)
- Training: End-to-End pipeline (CNN frozen, only train LSTM)

Compare with Model 2: same frozen CNN, same decoder, but end-to-end instead of pre-extracted
Compare with Model 1: same end-to-end pipeline, but pretrained CNN instead of scratch
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.components.pretrained_cnn import PretrainedCNN
from models.components.question_encoder import QuestionEncoder
from models.components.answer_decoder import AnswerDecoder
import config


class VQAModel5_PretrainedEndToEndNoAtt(nn.Module):
    """
    Model 5: Pretrained ResNet-50 (Frozen, End-to-End) + LSTM (No Attention)
    Input: Image (B, 3, 224, 224) + Question (B, seq_len)
    Output: Answer (B, max_len, vocab_size)
    """

    def __init__(self, vocab_size):
        super().__init__()

        # Image Encoder: ResNet-50 Pretrained (Frozen) → 2048-d vector
        self.image_encoder = PretrainedCNN(return_spatial=False)

        # Project 2048-d feature for fusion
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

    def _encode_context(self, images, questions):
        """Unified encoding logic for forward and generate."""
        img_features = self.image_encoder(images)            # (B, 2048)
        img_proj = self.image_proj(img_features)             # (B, H*2)
        _, q_context = self.question_encoder(questions)      # (B, H*2)
        return torch.cat([img_proj, q_context], dim=1)        # (B, H*4)

    def forward(self, images, questions, answers, teacher_forcing_ratio=1.0):
        """
        Args:
            images: (B, 3, 224, 224) - original image (end-to-end)
            questions: (B, q_len)
            answers: (B, a_len)
        Returns:
            outputs: (B, a_len, vocab_size)
        """
        context = self._encode_context(images, questions)
        outputs = self.decoder(context, answers, teacher_forcing_ratio)
        return outputs

    def generate(self, images, questions, sos_idx, eos_idx, max_len=30):
        """Answer generation for inference."""
        context = self._encode_context(images, questions)
        return self.decoder.generate(context, sos_idx, eos_idx, max_len)
