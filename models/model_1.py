"""
Model 1: Scratch CNN + LSTM Seq2Seq (No Attention)
- CNN: 4-layer Scratch CNN
- Question Encoder: Bi-LSTM
- Decoder: LSTM (No Attention)
- Training: End-to-End (CNN + LSTM in parallel)
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.components.scratch_cnn import ScratchCNN
from models.components.question_encoder import QuestionEncoder
from models.components.answer_decoder import AnswerDecoder
import vqa_config as config


class VQAModel1_ScratchNoAtt(nn.Module):
    """
    Model 1: Scratch CNN + LSTM Seq2Seq (No Attention)
    Input: Image (B, 3, 128, 128) + Question (B, seq_len)
    Output: Answer (B, max_len, vocab_size)
    """

    def __init__(self, vocab_size):
        super().__init__()

        # Image Encoder: Scratch CNN → 512-d vector
        self.image_encoder = ScratchCNN(
            out_channels=config.SCRATCH_CNN_OUT,
            return_spatial=False
        )

        # Question Encoder: Bi-LSTM → (hidden_size * 2)-d vector
        self.question_encoder = QuestionEncoder(
            vocab_size=vocab_size,
            embed_size=config.EMBED_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LSTM_LAYERS,
            dropout=config.DROPOUT,
        )

        # Fusion: Concatenate image_vector + question_vector
        context_dim = config.SCRATCH_CNN_OUT + config.HIDDEN_SIZE * 2

        # Answer Decoder: LSTM for answer generation
        self.decoder = AnswerDecoder(
            vocab_size=vocab_size,
            embed_size=config.EMBED_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            context_dim=context_dim,
            num_layers=config.NUM_LSTM_LAYERS,
            dropout=config.DROPOUT,
        )

    def forward(self, images, questions, answers, teacher_forcing_ratio=1.0):
        """
        Args:
            images: (B, 3, 128, 128)
            questions: (B, q_len)
            answers: (B, a_len)
            teacher_forcing_ratio: float
        Returns:
            outputs: (B, a_len, vocab_size)
        """
        # Encode
        img_features = self.image_encoder(images)           # (B, 512)
        _, q_context = self.question_encoder(questions)      # (B, H*2)

        # Fusion: concatenate
        context = torch.cat([img_features, q_context], dim=1)  # (B, 512 + H*2)

        # Decode
        outputs = self.decoder(context, answers, teacher_forcing_ratio)
        return outputs

    def generate(self, images, questions, sos_idx, eos_idx, max_len=30):
        """Answer generation for inference."""
        img_features = self.image_encoder(images)
        _, q_context = self.question_encoder(questions)
        context = torch.cat([img_features, q_context], dim=1)
        return self.decoder.generate(context, sos_idx, eos_idx, max_len)
