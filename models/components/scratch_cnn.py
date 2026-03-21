"""
scratch_cnn.py - Simple CNN trained from scratch (for Model 1 & 3)
"""

import torch
import torch.nn as nn


class ScratchCNN(nn.Module):
    """
    Simple 4-layer CNN to extract image features from scratch.
    Input:  (B, 3, 128, 128)
    Output: (B, 512, 8, 8) if return_spatial=True  (for Attention)
            (B, 512)       if return_spatial=False (for No Attention)
    """

    def __init__(self, out_channels=512, return_spatial=False):
        super().__init__()
        self.return_spatial = return_spatial

        self.features = nn.Sequential(
            # Layer 1: 3 → 64, 128x128 → 64x64
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Layer 2: 64 → 128, 64x64 → 32x32
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Layer 3: 128 → 256, 32x32 → 16x16
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Layer 4: 256 → 512, 16x16 → 8x8
            nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Output: (B, 512, 8, 8)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Args:
            x: (B, 3, 128, 128) - input image
        Returns:
            If return_spatial: (B, 512, 8, 8) - feature map for Attention
            If not:            (B, 512)        - vector for No Attention
        """
        features = self.features(x)  # (B, 512, 8, 8)

        if self.return_spatial:
            return features
        else:
            pooled = self.pool(features)       # (B, 512, 1, 1)
            return pooled.view(pooled.size(0), -1)  # (B, 512)
