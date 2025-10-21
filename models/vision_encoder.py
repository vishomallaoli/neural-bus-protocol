"""
Vision Encoder - ResNet-50
Extracts 2048D visual features from images
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from PIL import Image
from typing import Union


class VisionEncoder(nn.Module):
    """ResNet-50 feature extractor"""

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()

        # Load ResNet-50, keep everything up to avgpool (drops FC)
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # backbone outputs [B, 2048, 1, 1]
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048

        # Freeze weights if requested
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # Inference mode for speed + consistency
        self.backbone.eval()

        # ImageNet statistics for normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    # ---------------- internal helpers ----------------
    def _as_batched_tensor(self, x: Union[torch.Tensor, Image.Image, str]) -> torch.Tensor:
        """
        Convert input to a batched FloatTensor [B,3,224,224] in [0,1] (unnormalized).
        - If str: load image
        - If PIL: to tensor + resize via bilinear
        - If Tensor: accept [3,224,224] or [B,3,224,224]; resize if needed
        """
        if isinstance(x, str):
            x = Image.open(x).convert("RGB")

        if isinstance(x, Image.Image):
            # Convert PIL → tensor in [0,1], then resize to 224
            img = torch.frombuffer(x.tobytes(), dtype=torch.uint8)  # not used; keeping robust path below
            # Safer path: use torchvision functional without global transforms
            x = torch.from_numpy(
                (torch.ByteTensor(torch.ByteStorage.from_buffer(x.tobytes()))
                 .view(x.size[1], x.size[0], 3)
                 .numpy())
            )  # this low-level conversion is not ideal; we’ll switch to the standard way below
            # Replace with the standard conversion path:
            import torchvision.transforms.functional as F
            x = F.to_tensor(x) if isinstance(x, torch.Tensor) else F.to_tensor(Image.fromarray(x.numpy()))
            x = F.resize(x, [224, 224], antialias=True)  # [3,224,224]
            x = x.unsqueeze(0)  # [1,3,224,224]

        elif isinstance(x, torch.Tensor):
            # Expect [3,H,W] or [B,3,H,W]
            if x.ndim == 3:
                x = x.unsqueeze(0)  # [1,3,H,W]
            elif x.ndim != 4:
                raise ValueError(f"Tensor image must be [3,H,W] or [B,3,H,W], got {tuple(x.shape)}")

            # Ensure float32 in [0,1]
            if not torch.is_floating_point(x):
                x = x.float() / 255.0

            # Resize to 224 if needed
            if x.shape[-2:] != (224, 224):
                x = torch.nn.functional.interpolate(
                    x, size=(224, 224), mode="bilinear", align_corners=False
                )
        else:
            raise TypeError(f"Unsupported image type for VisionEncoder: {type(x)}")

        return x  # [B,3,224,224] in [0,1]

    def _normalize_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization to batched tensor [B,3,224,224]."""
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    # ---------------- forward API ----------------
    @torch.no_grad()
    def forward(self, x: Union[torch.Tensor, Image.Image, str]) -> torch.Tensor:
        """
        Extract features from a batch or single image.
        Accepts tensors, PIL Images, or file paths.

        Returns:
            [B, 2048] if B>1 else [2048]
        """
        # Convert to [B,3,224,224] in [0,1]
        x = self._as_batched_tensor(x)

        # Move to model device
        device = next(self.backbone.parameters()).device
        x = x.to(device)

        # Apply ImageNet normalization once here
        x = self._normalize_imagenet(x)

        # Forward through backbone (no grad; model is frozen/eval)
        feats = self.backbone(x)          # [B,2048,1,1]
        feats = feats.squeeze(-1).squeeze(-1)  # [B,2048]

        # Return [2048] for single image to match existing code
        return feats if feats.size(0) > 1 else feats.squeeze(0)

    # Kept for backwards compatibility with callers using .encode()
    def encode(self, image: Union[Image.Image, torch.Tensor, str]) -> torch.Tensor:
        """Alias for forward; kept for compatibility."""
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

    # Test with batched tensor [B,3,200,200] (auto-resize to 224)
    batch = torch.rand(2, 3, 200, 200)
    feats_batch = enc.forward(batch)
    print(f"Batched Tensor → {feats_batch.shape}")
