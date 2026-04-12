"""Localization modules
"""

import torch
import torch.nn as nn

class CustomDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"p value is not between 0 and 1, got {p}")
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        if self.p == 1.0:
            return torch.zeros_like(x)
        keep = (torch.rand_like(x) > self.p).float()
        inv_keep = 1.0 / (1.0 - self.p)
        return x * keep * inv_keep

class VGG11Encoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.part1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.drop1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.drop2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.drop3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.drop4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.part5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.drop5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, return_features: bool = False):
        p1 = self.part1(x)
        t = self.drop1(p1)
        p2 = self.part2(t)
        t = self.drop2(p2)
        p3 = self.part3(t)
        t = self.drop3(p3)
        p4 = self.part4(t)
        t = self.drop4(p4)
        p5 = self.part5(t)
        bottleneck = self.drop5(p5)
        if return_features:
            return bottleneck, {'f1': p1, 'f2': p2, 'f3': p3, 'f4': p4, 'f5': p5}
        return bottleneck

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        self.seer = VGG11Encoder(in_channels=in_channels)
        self.squash = nn.AdaptiveAvgPool2d((7, 7))
        self.finder = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        pooled = self.squash(self.seer(x, return_features=False))
        flat = torch.flatten(pooled, 1)
        return self.finder(flat)
