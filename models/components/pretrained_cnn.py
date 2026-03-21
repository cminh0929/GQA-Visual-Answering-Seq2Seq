"""
pretrained_cnn.py - ResNet-50 Pretrained (Frozen) used end-to-end
Used for Model 5 & 6 (end-to-end pipeline with pretrained CNN).

ResNet-50 is loaded with pretrained weights but totally FROZEN (no training).
Forward pass still runs through CNN each batch (end-to-end pipeline),
but gradients do not flow back to CNN.
"""

import torch
import torch.nn as nn
from torchvision import models


class PretrainedCNN(nn.Module):
    """
    ResNet-50 pretrained (frozen) for end-to-end VQA pipeline.

    return_spatial=False:
        Input:  (B, 3, 224, 224)
        Output: (B, 2048)  - pooled feature vector (for No Attention)

    return_spatial=True:
        Input:  (B, 3, 224, 224)
        Output: (B, 2048, 7, 7) - spatial feature map (for Attention)
    """

    def __init__(self, return_spatial=False):
        super().__init__()
        self.return_spatial = return_spatial

        # Load ResNet-50 pretrained
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Get all layers except avgpool and fc
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        # Output of backbone: (B, 2048, 7, 7) with input 224x224

        # Average pooling (used when return_spatial=False)
        self.avgpool = resnet.avgpool  # AdaptiveAvgPool2d((1, 1))

        # FREEZE entire CNN - no training
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) - normalized image

        Returns:
            If return_spatial: (B, 2048, 7, 7) - feature map for Attention
            If not:            (B, 2048)        - vector for No Attention
        """
        features = self.backbone(x)  # (B, 2048, 7, 7)

        if self.return_spatial:
            return features
        else:
            pooled = self.avgpool(features)  # (B, 2048, 1, 1)
            return pooled.view(pooled.size(0), -1)  # (B, 2048)
