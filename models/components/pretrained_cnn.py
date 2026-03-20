"""
pretrained_cnn.py - ResNet-50 Pretrained (Frozen) dùng end-to-end
Dùng cho Model 5 & 6 (end-to-end pipeline với pretrained CNN).

ResNet-50 được load pretrained weights nhưng FREEZE toàn bộ (không train).
Forward pass vẫn chạy qua CNN mỗi batch (end-to-end pipeline),
nhưng gradient không chảy về CNN.
"""

import torch
import torch.nn as nn
from torchvision import models


class PretrainedCNN(nn.Module):
    """
    ResNet-50 pretrained (frozen) cho end-to-end VQA pipeline.

    return_spatial=False:
        Input:  (B, 3, 224, 224)
        Output: (B, 2048)  - pooled feature vector (cho No Attention)

    return_spatial=True:
        Input:  (B, 3, 224, 224)
        Output: (B, 2048, 7, 7) - spatial feature map (cho Attention)
    """

    def __init__(self, return_spatial=False):
        super().__init__()
        self.return_spatial = return_spatial

        # Load ResNet-50 pretrained
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Lấy tất cả layers trừ avgpool và fc
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
        # Output của backbone: (B, 2048, 7, 7) với input 224x224

        # Average pooling (dùng khi return_spatial=False)
        self.avgpool = resnet.avgpool  # AdaptiveAvgPool2d((1, 1))

        # FREEZE toàn bộ CNN - không train
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) - ảnh đã normalize

        Returns:
            Nếu return_spatial: (B, 2048, 7, 7) - feature map cho Attention
            Nếu không:          (B, 2048)        - vector cho No Attention
        """
        features = self.backbone(x)  # (B, 2048, 7, 7)

        if self.return_spatial:
            return features
        else:
            pooled = self.avgpool(features)  # (B, 2048, 1, 1)
            return pooled.view(pooled.size(0), -1)  # (B, 2048)
