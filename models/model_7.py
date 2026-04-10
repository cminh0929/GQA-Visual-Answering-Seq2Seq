import torch
import torch.nn as nn
import os
import sys

# Add root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import vqa_config as config
from models.components.pretrained_cnn import PretrainedCNN
from models.components.transformer_utils import PositionalEncoding

class VQAModel7_Transformer(nn.Module):
    """
    Model 7: Transformer-based VQA
    - Encoder: ResNet-50 (Image) + Transformer Encoder (Question)
    - Decoder: Transformer Decoder (Generates Answer)
    - Strategy: End-to-End
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1. Image Encoder (Frozen ResNet backbone)
        self.image_encoder = PretrainedCNN(return_spatial=True)
        self.visual_projection = nn.Linear(config.RESNET_FEATURE_DIM, d_model)
        
        # 2. Question/Answer Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # 3. Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        # 4. Final Projection
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _encode_visual(self, images):
        """Extract and project visual features."""
        # spatial: (B, 2048, 7, 7)
        spatial = self.image_encoder(images)
        B, C, H, W = spatial.size()
        # Flatten spatial: (B, 49, 2048)
        spatial = spatial.view(B, C, H * W).permute(0, 2, 1)
        # Project: (B, 49, d_model)
        return self.visual_projection(spatial)

    def forward(self, images, questions, answers):
        """
        Standard training forward pass.
        answers input used for teacher forcing (shifted).
        """
        # 1. Encode Image
        visual_src = self._encode_visual(images) # (B, 49, d_model)
        
        # 2. Encode Question
        question_src = self.embedding(questions) # (B, q_len, d_model)
        question_src = self.pos_encoding(question_src)
        
        # 3. Combine Visual and Question as Encoder Input
        # (B, 49 + q_len, d_model)
        src = torch.cat([visual_src, question_src], dim=1)
        
        # 4. Prepare Answer (Target for Decoder)
        # Shift target for causal training
        tgt = self.embedding(answers) # (B, a_len, d_model)
        tgt = self.pos_encoding(tgt)
        
        # 5. Causal Mask for Decoder
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # 6. Transformer Forward
        # src: encoder input, tgt: decoder input
        out = self.transformer(src, tgt, tgt_mask=tgt_mask) # (B, a_len, d_model)
        
        return self.fc_out(out)

    def generate(self, images, questions, sos_idx, eos_idx, max_len=30):
        """Autoregressive generation."""
        device = images.device
        B = images.size(0)
        
        # 1. Encode Context (Memory)
        visual_src = self._encode_visual(images)
        question_src = self.embedding(questions)
        question_src = self.pos_encoding(question_src)
        src = torch.cat([visual_src, question_src], dim=1)
        
        # Pre-calculate memory (encoder output)
        memory = self.transformer.encoder(src)
        
        # 2. Start generation
        generated = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            # Target for decoder
            tgt = self.embedding(generated)
            tgt = self.pos_encoding(tgt)
            
            # Causal mask
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            # Decode
            out = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
            
            # Get last token prediction
            next_token_logits = self.fc_out(out[:, -1, :]) # (B, vocab_size)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check if all batches hit EOS
            if (next_token == eos_idx).all():
                break
                
        return generated
