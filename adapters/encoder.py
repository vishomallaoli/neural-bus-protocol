"""
Encoder Adapter
Maps vision features (2048D) to BUS vectors (512D) and optionally BUS packets.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

# Add project root to path (so bus.schema or schema can be imported)
sys.path.insert(0, str(Path(__file__).parent.parent))

# If your schema is under bus/schema.py, keep this:
from bus.schema import create_packet, BUSPacket
# If schema.py lives next to this file instead, use:
# from schema import create_packet, BUSPacket


class EncoderAdapter(nn.Module):
    """
    Encoder Adapter

    - Input:  vision features from ResNet-50, shape [2048] or [B, 2048]
    - Output: BUS vector, shape [512] or [B, 512]

    Optionally, `encode()` can wrap the BUS vector into a BUSPacket using
    the shared schema.
    """

    def __init__(self, input_dim: int = 2048, output_dim: int = 512):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        # Simple linear projection; you can extend this to a small MLP if desired
        self.proj = nn.Linear(self.input_dim, self.output_dim)

    # ---------------- core forward ----------------
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features → BUS vector.

        Args:
            vision_features: [D] or [B, D] tensor (D = input_dim)

        Returns:
            BUS vector of shape [output_dim] or [B, output_dim]
        """
        x = vision_features
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, D]

        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"EncoderAdapter expected last dim={self.input_dim}, "
                f"got {x.size(-1)}"
            )

        out = self.proj(x)  # [B, output_dim]
        return out if out.size(0) > 1 else out.squeeze(0)

    # ---------------- BUS packet helper ----------------
    @torch.no_grad()
    def encode(
        self,
        vision_features: torch.Tensor,
        text: str,
        intent: str = "vqa",
        source: str = "vision.resnet50.v1",
        vision_model: str = "resnet50",
        caption_model: str = "blip2",
        **metadata: Any,
    ) -> BUSPacket:
        """
        Convenience utility:
        Vision features + text → BUSPacket (vector + text + metadata)

        Args:
            vision_features: [D] or [B, D] tensor (we expect a single vector here)
            text: caption or description of the image
            intent: task, e.g. "vqa", "caption"
            source: string identifying the source component
            vision_model: vision encoder name/version
            caption_model: captioning model name/version
            **metadata: extra fields (e.g., image_id, question_id)

        Returns:
            BUSPacket
        """
        # For now, we assume a single feature vector, not a batch.
        if vision_features.dim() > 1:
            if vision_features.size(0) != 1:
                raise ValueError(
                    "EncoderAdapter.encode currently expects a single feature vector "
                    f"but got shape {vision_features.shape}. "
                    "Call this in a loop for batched use."
                )
            vision_features = vision_features.squeeze(0)

        bus_vector = self.forward(vision_features)  # [output_dim]

        packet = create_packet(
            vector=bus_vector,
            text=text,
            source=source,
            intent=intent,
            vision_model=vision_model,
            caption_model=caption_model,
            **metadata,
        )
        return packet


# Quick test
if __name__ == "__main__":
    print("Testing Encoder Adapter...")

    adapter = EncoderAdapter()
    print(f"✅ Created adapter ({adapter.input_dim} → {adapter.output_dim})")

    # Single vector
    vision_feat = torch.randn(2048)
    bus_vector = adapter(vision_feat)
    print(f"✅ Projection (single): {vision_feat.shape} → {bus_vector.shape}")

    # Batched
    vision_batch = torch.randn(4, 2048)
    bus_batch = adapter(vision_batch)
    print(f"✅ Projection (batch): {vision_batch.shape} → {bus_batch.shape}")

    # Packet creation
    packet = adapter.encode(
        vision_features=vision_feat,
        text="A test image",
        intent="vqa",
        image_id=123,
        question_id=456,
    )
    print(f"✅ Created packet: {packet.size_bytes()} bytes")
