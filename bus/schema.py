"""
Neural BUS Packet Schema
Core data structure for cross-model communication
"""

import json
import torch
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class Header:
    """Routing and task information"""
    source: str          # e.g., "vision.resnet50.v1"
    intent: str          # e.g., "vqa", "captioning"
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VectorPayload:
    """Dense embedding component"""
    dim: int
    dtype: str
    data: torch.Tensor
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "dtype": self.dtype,
            "data": self.data.cpu().numpy().tolist()
        }


@dataclass
class Provenance:
    """Model versioning and reproducibility"""
    vision_model: str
    vision_version: str
    caption_model: str
    device: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BUSPacket:
    """
    Neural BUS communication packet
    
    Standardized format combining:
    - Dense vector embeddings (machine-readable)
    - Text descriptions (human-readable)
    - Metadata for routing and provenance
    """
    header: Header
    payload: Dict[str, Any]  # Contains 'vector' and 'text'
    provenance: Provenance
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "header": self.header.to_dict(),
            "payload": {
                "vector": {
                    "dim": self.payload["vector"].dim,
                    "dtype": self.payload["vector"].dtype,
                    "data": self.payload["vector"].data.cpu().numpy().tolist()
                },
                "text": self.payload["text"]
            },
            "provenance": self.provenance.to_dict(),
            "metadata": self.metadata
        }
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Serialize to JSON"""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def get_vector(self) -> torch.Tensor:
        """Extract vector tensor"""
        return self.payload["vector"].data
    
    def get_text(self) -> str:
        """Extract text view"""
        return self.payload["text"]
    
    def size_bytes(self) -> int:
        """Calculate packet size"""
        return len(self.to_json().encode('utf-8'))
    
    def __repr__(self) -> str:
        return (
            f"BUSPacket(\n"
            f"  source={self.header.source}\n"
            f"  intent={self.header.intent}\n"
            f"  vector_dim={self.payload['vector'].dim}\n"
            f"  text='{self.payload['text'][:50]}...'\n"
            f"  size={self.size_bytes()/1024:.2f}KB\n"
            f")"
        )


def create_packet(
    vector: torch.Tensor,
    text: str,
    source: str,
    intent: str,
    vision_model: str,
    caption_model: str,
    **metadata
) -> BUSPacket:
    """
    Factory function to create BUS packet
    
    Args:
        vector: Feature embedding [512]
        text: Human-readable description
        source: Source identifier
        intent: Task type (vqa, caption, etc.)
        vision_model: Vision model name
        caption_model: Captioning model name
        **metadata: Additional task-specific data
    
    Returns:
        BUSPacket instance
    """
    # Ensure 1D vector
    if len(vector.shape) > 1:
        vector = vector.squeeze()
    
    return BUSPacket(
        header=Header(
            source=source,
            intent=intent,
            timestamp=datetime.now().isoformat()
        ),
        payload={
            "vector": VectorPayload(
                dim=vector.shape[0],
                dtype="float32",
                data=vector
            ),
            "text": text
        },
        provenance=Provenance(
            vision_model=vision_model,
            vision_version="torchvision-0.15.0",
            caption_model=caption_model,
            device=str(vector.device)
        ),
        metadata=metadata if metadata else None
    )


# Quick test
if __name__ == "__main__":
    print("Testing BUS Schema...")
    
    # Create test packet
    test_vector = torch.randn(512)
    packet = create_packet(
        vector=test_vector,
        text="A red cup on a wooden table",
        source="vision.resnet50.v1",
        intent="vqa",
        vision_model="resnet50",
        caption_model="blip2",
        image_id=12345
    )
    
    print("\n" + str(packet))
    print(f"\n✅ Packet created: {packet.size_bytes()} bytes")
    
    # Test JSON serialization
    json_str = packet.to_json()
    print(f"✅ JSON serialization: {len(json_str)} chars")
    
    # Test extraction
    v = packet.get_vector()
    t = packet.get_text()
    print(f"✅ Vector extracted: {v.shape}")
    print(f"✅ Text extracted: {t}")