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
                    gdown.download(id="1HFpmi7275QMc45quQKtQlF7VzamNzLG4", output=localizer_path, quiet=False)
                if not os.path.exists(unet_path):
                    gdown.download(id="1z7JK5cHYOAicmNkPpHvz6mg4R67c9Rl7", output=unet_path, quiet=False)
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
        self.seg_head = nn.ModuleList([
            u_mod.up4, u_mod.dc4,
            u_mod.up3, u_mod.dc3,
            u_mod.up2, u_mod.dc2,
            u_mod.up1, u_mod.dc1,
            u_mod.up0, u_mod.hd
        ])
        
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
        l_raw = self.loc_head(bt)
        l_cxcy = torch.sigmoid(l_raw[:, 0:2])
        l_wh = torch.sigmoid(l_raw[:, 2:4]).clamp(min=1e-3, max=1.0)
        # Rescale the normalized outputs [0, 1] to original image size (224x224) 
        l_out = torch.cat([l_cxcy, l_wh], dim=1) * 224.0
        
        u4, d4, u3, d3, u2, d2, u1, d1, u0, hd = self.seg_head
        
        s = u4(bt)
        s = torch.cat([s, fts['f4']], dim=1)
        s = d4(s)
        
        s = u3(s)
        s = torch.cat([s, fts['f3']], dim=1)
        s = d3(s)
        
        s = u2(s)
        s = torch.cat([s, fts['f2']], dim=1)
        s = d2(s)
        
        s = u1(s)
        s = torch.cat([s, fts['f1']], dim=1)
        s = d1(s)
        
        s = u0(s)
        s_out = hd(s)
        
        return {
            'classification': c_out,
            'localization': l_out,
            'segmentation': s_out
        }
