"""Segmentation model
"""

import torch
import torch.nn as nn

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

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.rise = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_x):
        r = self.rise(x)
        merged = torch.cat([r, skip_x], dim=1)
        return self.fuse(merged)

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """
    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.seer = VGG11Encoder(in_channels=in_channels)
        self.climb1 = DecoderBlock(in_channels=512, skip_channels=512, out_channels=512)
        self.climb2 = DecoderBlock(in_channels=512, skip_channels=512, out_channels=256)
        self.climb3 = DecoderBlock(in_channels=256, skip_channels=256, out_channels=128)
        self.climb4 = DecoderBlock(in_channels=128, skip_channels=128, out_channels=64)
        self.climb5 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)
        self.painter = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        bottleneck, features = self.seer(x, return_features=True)
        c1 = self.climb1(bottleneck, features['f5'])
        c2 = self.climb2(c1, features['f4'])
        c3 = self.climb3(c2, features['f3'])
        c4 = self.climb4(c3, features['f2'])
        c5 = self.climb5(c4, features['f1'])
        return self.painter(c5)
