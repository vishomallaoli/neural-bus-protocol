"""
Encoder Adapter
Maps vision features (2048D) to BUS packets (512D + text)
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from bus.schema import create_packet, BUSPacket


class EncoderAdapter(nn.Module):
    """
    Vision → BUS adapter
    Compresses 2048D vision features to 512D BUS vectors
    """

    def __init__(self, input_dim: int = 2048, output_dim: int = 512):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        hidden_dim = (input_dim + output_dim) // 2  # 1280

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to BUS space (grad-preserving)
        Args:
            vision_features: [B, 2048] or [2048]
        Returns:
            bus_vector: [B, 512] or [512]
        """
        return self.projection(vision_features)

    # ---------- NEW: graph-preserving helper for training ----------
    def forward_tensor(self, vision_features: torch.Tensor, text: str, question: str) -> torch.Tensor:
        """
        Graph-preserving variant used by the training path.
        Do NOT detach/serialize here.
        Returns a single-sample BUS vector [512] (or [B,512] if input was batched).
        """
        if vision_features.dim() == 1:
            vision_features = vision_features.unsqueeze(0)  # [1, 2048]

        # ensure device/dtype match module params
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        vision_features = vision_features.to(device=device, dtype=dtype)

        bus = self.forward(vision_features)  # [B, 512]
        return bus.squeeze(0)

    # ---------- Inference path (packet creation) ----------
    def encode(
        self,
        vision_features: torch.Tensor,
        text: str,
        intent: str = "vqa",
        **metadata,
    ) -> BUSPacket:
        """
        Create complete BUS packet (inference-friendly; uses no_grad)
        """
        if len(vision_features.shape) > 1:
            vision_features = vision_features.squeeze(0)

        # Inference: OK to detach for serialization
        with torch.no_grad():
            bus_vector = self.forward(vision_features)

        packet = create_packet(
            vector=bus_vector,
            text=text,
            source="vision.resnet50.v1",
            intent=intent,
            vision_model="resnet50",
            caption_model="blip2",
            **metadata,
        )
        return packet


# Quick test
if __name__ == "__main__":
    print("Testing Encoder Adapter...")

    adapter = EncoderAdapter()
    print(f"✅ Created adapter ({adapter.input_dim} → {adapter.output_dim})")

    vision_feat = torch.randn(2048)
    bus_vector = adapter(vision_feat)
    print(f"✅ Projection: {vision_feat.shape} → {bus_vector.shape}")

    packet = adapter.encode(
        vision_features=vision_feat,
        text="A test image",
        intent="vqa",
    )
    print(f"✅ Created packet: {packet.size_bytes()} bytes")
