"""
Vision Encoder - ResNet-50
Extracts 2048D visual features from images
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Union


class VisionEncoder(nn.Module):
    """ResNet-50 feature extractor"""
    
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()

        # Load ResNet-50, remove classification head
        from torchvision.models import ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048

        # Freeze weights
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # ✅ Updated to handle both tensors and PIL images
    def forward(self, x: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
        """
        Extract features from a batch or single image.
        Accepts both tensors and PIL Images.
        """
        # 🔧 Only apply transform if PIL
        if isinstance(x, Image.Image):
            x = self.transform(x).unsqueeze(0)

        # 🔧 If single image tensor [3,224,224], batchify
        elif isinstance(x, torch.Tensor) and x.ndim == 3:
            x = x.unsqueeze(0)

        # 🔧 If already a normalized tensor, skip transforms
        elif isinstance(x, torch.Tensor) and x.ndim == 4:
            pass

        else:
            raise TypeError(f"Unexpected input type for VisionEncoder: {type(x)}")

        # Move to same device as model
        x = x.to(next(self.parameters()).device)

        # Extract features
        with torch.no_grad():
            features = self.backbone(x)

        return features.squeeze(-1).squeeze(-1)


    def encode(self, image: Union[Image.Image, torch.Tensor, str]) -> torch.Tensor:
        """
        Encode single image
        Accepts PIL Image, torch Tensor, or file path.
        """
        # Load from path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # 🔧 Only apply transform if PIL image
        if isinstance(image, Image.Image):
            img_tensor = self.transform(image).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            # If single image [3,224,224], add batch dim
            img_tensor = image.unsqueeze(0) if image.ndim == 3 else image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Move to model device
        device = next(self.parameters()).device
        img_tensor = img_tensor.to(device)

        # Extract features
        with torch.no_grad():
            features = self.forward(img_tensor)

        return features.squeeze(0)



# Quick test
if __name__ == "__main__":
    print("Testing Vision Encoder...")
    
    encoder = VisionEncoder()
    print(f"✅ Loaded ResNet-50 (feature_dim={encoder.feature_dim})")

    # Test with PIL
    test_img = Image.new('RGB', (224, 224), 'red')
    feats_pil = encoder.forward(test_img)
    print(f"PIL Input → {feats_pil.shape}")

    # Test with tensor
    tensor_img = torch.randn(1, 3, 224, 224)
    feats_tensor = encoder.forward(tensor_img)
    print(f"Tensor Input → {feats_tensor.shape}")
