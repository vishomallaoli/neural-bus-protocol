"""
Vision Encoder - ResNet-50
Extracts 2048D visual features from images
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchvision.transforms import functional as TF


class VisionEncoder(nn.Module):
    """ResNet-50 feature extractor."""

    def __init__(
        self,
        pretrained: bool = True,
        freeze: bool = True,
        image_size: int = 224,
    ):
        super().__init__()

        # Load ResNet-50 backbone up to avgpool (drop FC)
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B, 2048, 1, 1]
        self.feature_dim = 2048
        self.image_size = int(image_size)
        self._frozen = bool(freeze)

        # Freeze (and eval) if requested
        if self._frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()
        else:
            self.backbone.train()

        # ImageNet statistics for normalization (buffers follow module device)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    # ---------------- internal helpers ----------------
    def _as_batched_tensor(self, x: Union[torch.Tensor, Image.Image, str]) -> torch.Tensor:
        """Return a contiguous float32 tensor in [0,1] with shape [B,3,S,S]."""
        S = self.image_size

        # str → PIL
        if isinstance(x, str):
            x = Image.open(x).convert("RGB")

        # PIL → tensor
        if isinstance(x, Image.Image):
            if x.mode != "RGB":
                x = x.convert("RGB")
            x = TF.to_tensor(x)                 # [3,H,W], float32 in [0,1]
            x = TF.resize(x, [S, S], antialias=True)  # tensor resize
            x = x.unsqueeze(0)                  # [1,3,S,S]
            return x.contiguous()

        # Tensor paths
        if isinstance(x, torch.Tensor):
            # Support HWC single-image
            if x.ndim == 3 and x.shape[-1] in (1, 3):   # [H,W,C] → [C,H,W]
                x = x.permute(2, 0, 1)

            # Support BHWC batched
            if x.ndim == 4 and x.shape[-1] in (1, 3):   # [B,H,W,C] → [B,C,H,W]
                x = x.permute(0, 3, 1, 2)

            # Ensure batch dim
            if x.ndim == 3:                              # [C,H,W]
                x = x.unsqueeze(0)                       # [1,C,H,W]
            elif x.ndim != 4:
                raise ValueError(
                    f"Tensor image must be [3,H,W], [H,W,3], [B,3,H,W], or [B,H,W,3]; got {tuple(x.shape)}"
                )

            # dtype/range
            if not torch.is_floating_point(x):
                x = x.float() / 255.0
            else:
                x = x.to(dtype=torch.float32)
            x.clamp_(0.0, 1.0)

            # grayscale → RGB
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

            # resize if needed
            if x.shape[-2:] != (S, S):
                x = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)

            return x.contiguous()

        raise TypeError(f"Unsupported image type for VisionEncoder: {type(x)}")

    def _normalize_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization to batched tensor [B,3,S,S]."""
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    # ---------------- forward API ----------------
    def forward(self, x: Union[torch.Tensor, Image.Image, str]) -> torch.Tensor:
        """
        Extract features from a batch or single image.
        Accepts tensors, PIL Images, or file paths.

        Returns:
            [B, 2048] if B>1 else [2048]
        """
        # Convert to [B,3,S,S] in [0,1]
        x = self._as_batched_tensor(x)

        # Move to model device
        device = next(self.backbone.parameters()).device
        x = x.to(device)

        # Normalize
        x = self._normalize_imagenet(x)

        # Forward through backbone
        if self._frozen or not self.training:
            with torch.no_grad():
                feats = self.backbone(x)  # [B,2048,1,1]
        else:
            feats = self.backbone(x)      # allows grad if un-frozen

        feats = feats.flatten(1)          # [B,2048]

        # Return [2048] for single image to match existing code
        return feats if feats.size(0) > 1 else feats.squeeze(0)

    # Back-compat alias
    def encode(self, image: Union[Image.Image, torch.Tensor, str]) -> torch.Tensor:
        return self.forward(image)


# Quick test
if __name__ == "__main__":
    print("Testing Vision Encoder...")

    enc = VisionEncoder()
    print(f"✅ Loaded ResNet-50 (feature_dim={enc.feature_dim})")

    # Test with PIL
    pil_img = Image.new("RGB", (320, 240), "red")
    feats_pil = enc.forward(pil_img)
    print(f"PIL Input → {feats_pil.shape}")

    # Test with tensor [3,224,224] in [0,1]
    tensor_img = torch.rand(3, 224, 224)
    feats_tensor = enc.forward(tensor_img)
    print(f"Tensor Input → {feats_tensor.shape}")

    # Test with batched tensor [B,3,200,200]
    batch = torch.rand(2, 3, 200, 200)
    feats_batch = enc.forward(batch)
    print(f"Batched Tensor → {feats_batch.shape}")

    # Test BHWC single image
    hwc = torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8)
    feats_hwc = enc.forward(hwc)
    print(f"Single HWC Tensor → {feats_hwc.shape}")

    # Test BHWC batch
    bhwc = torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8)
    feats_bhwc = enc.forward(bhwc)
    print(f"Batch HxWxC Tensor → {feats_bhwc.shape}")
