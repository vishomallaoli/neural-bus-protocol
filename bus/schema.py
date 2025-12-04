"""
Neural BUS Packet Schema
Core data structure for cross-model communication
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any

import torch


# ---------------------------------------------------------------------------
# Header: high level routing and intent
# ---------------------------------------------------------------------------

@dataclass
class Header:
    """Routing and task information."""
    source: str          # e.g. "vision.resnet50.v1"
    intent: str          # e.g. "vqa", "captioning"
    timestamp: str       # ISO formatted timestamp

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Vector payload: dense embedding
# ---------------------------------------------------------------------------

@dataclass
class VectorPayload:
    """Dense embedding component of the BUS packet."""
    dim: int
    dtype: str
    data: torch.Tensor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "dtype": self.dtype,
            "data": self.data.cpu().numpy().tolist(),
        }


# ---------------------------------------------------------------------------
# Provenance: model versions and environment
# ---------------------------------------------------------------------------

@dataclass
class Provenance:
    """Model versioning and reproducibility information."""
    vision_model: str
    vision_version: str
    caption_model: str
    device: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# BUS Packet: header + payload + provenance
# ---------------------------------------------------------------------------

@dataclass
class BUSPacket:
    """
    Neural BUS communication packet.

    Standardized format combining:
    - Dense vector embeddings (machine-readable)
    - Text descriptions (human-readable)
    - Metadata for routing and provenance
    """
    header: Header
    # payload["vector"] is a VectorPayload, payload["text"] is a string
    payload: Dict[str, Any]
    provenance: Provenance
    metadata: Optional[Dict[str, Any]] = None

    # ---------------- serialization helpers ----------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize packet to a plain Python dictionary."""
        vector_payload: VectorPayload = self.payload["vector"]
        text: str = self.payload["text"]

        return {
            "header": self.header.to_dict(),
            "payload": {
                "vector": vector_payload.to_dict(),
                "text": text,
            },
            "provenance": self.provenance.to_dict(),
            "metadata": self.metadata,
        }

    def to_json(self, filepath: Optional[str] = None) -> str:
        """Serialize packet to JSON string, optionally saving to a file."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)

        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)

        return json_str

    # ---------------- convenience accessors ----------------

    def get_vector(self) -> torch.Tensor:
        """Extract the dense vector as a torch.Tensor."""
        return self.payload["vector"].data

    def get_text(self) -> str:
        """Extract the human-readable text description."""
        return self.payload["text"]

    def size_bytes(self) -> int:
        """Approximate size of the serialized packet in bytes."""
        return len(self.to_json().encode("utf-8"))

    def __repr__(self) -> str:
        return (
            "BUSPacket(\n"
            f"  source={self.header.source}\n"
            f"  intent={self.header.intent}\n"
            f"  vector_dim={self.payload['vector'].dim}\n"
            f"  text='{self.payload['text'][:50]}...'\n"
            f"  size={self.size_bytes()/1024:.2f}KB\n"
            ")"
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_packet(
    vector: torch.Tensor,
    text: str,
    source: str,
    intent: str,
    vision_model: str,
    caption_model: str,
    **metadata: Any,
) -> BUSPacket:
    """
    Factory function to create a BUSPacket.

    Args:
        vector: Feature embedding [D]
        text: Human-readable description
        source: Source identifier
        intent: Task type ("vqa", "caption", etc.)
        vision_model: Vision model name
        caption_model: Captioning model name
        **metadata: Additional task-specific metadata (e.g. image_id)

    Returns:
        BUSPacket instance
    """
    # Ensure 1D vector
    if len(vector.shape) > 1:
        vector = vector.squeeze()

    # Make sure we know the dtype as a string
    dtype_str = str(vector.dtype)

    header = Header(
        source=source,
        intent=intent,
        timestamp=datetime.now().isoformat(),
    )

    payload = {
        "vector": VectorPayload(
            dim=vector.shape[0],
            dtype=dtype_str,
            data=vector,
        ),
        "text": text,
    }

    provenance = Provenance(
        vision_model=vision_model,
        vision_version="torchvision-0.15.0",
        caption_model=caption_model,
        device=str(vector.device),
    )

    return BUSPacket(
        header=header,
        payload=payload,
        provenance=provenance,
        metadata=metadata if metadata else None,
    )


# ---------------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing BUS Schema...")

    test_vector = torch.randn(512)
    packet = create_packet(
        vector=test_vector,
        text="A red cup on a wooden table",
        source="vision.resnet50.v1",
        intent="vqa",
        vision_model="resnet50",
        caption_model="blip2",
        image_id=12345,
    )

    print("\n" + str(packet))
    print(f"\n✅ Packet created: {packet.size_bytes()} bytes")

    json_str = packet.to_json()
    print(f"✅ JSON serialization: {len(json_str)} chars")

    v = packet.get_vector()
    t = packet.get_text()
    print(f"✅ Vector extracted: {v.shape}")
    print(f"✅ Text extracted: {t}")
