"""Unified multi-task model
"""

import torch
import torch.nn as nn


def _extract_state_dict(ckpt_obj):
    """Support both raw state_dict and wrapped checkpoint dict formats."""
    if isinstance(ckpt_obj, dict) and 'state_dict' in ckpt_obj:
        return ckpt_obj['state_dict']
    return ckpt_obj

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "checkpoints/classifier.pth", localizer_path: str = "checkpoints/localizer.pth", unet_path: str = "checkpoints/unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()
        import os
        from .vgg11 import VGG11Encoder
        from .classification import VGG11Classifier
        from .localization import VGG11Localizer
        from .segmentation import VGG11UNet

        # Auto-download checkpoints from Google Drive when files are missing.
        if not (os.path.exists(classifier_path) and os.path.exists(localizer_path) and os.path.exists(unet_path)):
            try:
                import gdown
                os.makedirs(os.path.dirname(classifier_path) or ".", exist_ok=True)
                os.makedirs(os.path.dirname(localizer_path) or ".", exist_ok=True)
                os.makedirs(os.path.dirname(unet_path) or ".", exist_ok=True)

                if not os.path.exists(classifier_path):
                    gdown.download(id="1c-3v_lRaMJiS28rK1WQOuP31qgBwtMUl", output=classifier_path, quiet=False)
                if not os.path.exists(localizer_path):
                    gdown.download(id="150uYxzErRgr9KOy3Z94MwG-0YExOlJZE", output=localizer_path, quiet=False)
                if not os.path.exists(unet_path):
                    gdown.download(id="1KZ8LQEh9twkyzmwPeKbatnR2__KF_NbZ", output=unet_path, quiet=False)
            except Exception as e:
                raise RuntimeError(f"Checkpoint download failed. Install gdown and verify Drive IDs. Error: {e}")
        
        self.enc = VGG11Encoder(in_channels)
        
        c_mod = VGG11Classifier(num_breeds, in_channels)
        if os.path.exists(classifier_path):
            c_mod.load_state_dict(_extract_state_dict(torch.load(classifier_path, map_location='cpu')))
        self.cls_head = nn.Sequential(c_mod.avg, nn.Flatten(), c_mod.hd)
        
        l_mod = VGG11Localizer(in_channels)
        if os.path.exists(localizer_path):
            l_mod.load_state_dict(_extract_state_dict(torch.load(localizer_path, map_location='cpu')))
        self.loc_head = nn.Sequential(l_mod.avg, nn.Flatten(), l_mod.loc)
        
        u_mod = VGG11UNet(seg_classes, in_channels)
        if os.path.exists(unet_path):
            u_mod.load_state_dict(_extract_state_dict(torch.load(unet_path, map_location='cpu')))
        self.seg_head = u_mod  # Keep the full separate UNet here as requested!
        
        # Load backbone from classifier initially
        self.enc.load_state_dict(c_mod.enc.state_dict())

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        bt, fts = self.enc(x, return_features=True)
        
        c_out = self.cls_head(bt)
        l_out = self.loc_head(bt)
        
        # Forward pass using the UNet's separate VGG encoder & decoder structure 
        s_out = self.seg_head(x)
        
        return {
            'classification': c_out,
            'localization': l_out,
            'segmentation': s_out
        }
