"""
Decoder Adapter
Maps BUS vectors (512D) to LLM hidden space (llm_dim) and builds prompts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# If your schema is under bus/schema.py, keep this:
from bus.schema import BUSPacket
# If schema.py lives next to this file instead, use:
# from schema import BUSPacket


class DecoderAdapter(nn.Module):
    """
    Decoder Adapter

    - Input:  BUS vector, shape [512] or [B, 512]
    - Output: LLM soft prompt embedding, shape [llm_dim] or [B, llm_dim]

    Also provides:
      - format_prompt(caption, question) → prompt string
      - prepare(packet, question) → (llm_embedding, prompt)
    """

    def __init__(self, bus_dim: int = 512, llm_dim: int = 4096):
        """
        Args:
            bus_dim: dimensionality of BUS vector (default 512)
            llm_dim: dimensionality of LLM hidden state (e.g. 768 for DistilGPT2)
        """
        super().__init__()
        self.bus_dim = int(bus_dim)
        self.llm_dim = int(llm_dim)

        # Simple projection + nonlinearity; can be upgraded to a deeper MLP
        self.proj = nn.Sequential(
            nn.Linear(self.bus_dim, self.llm_dim),
            nn.Tanh(),
        )

    # ---------------- core forward ----------------
    def forward(self, bus_vector: torch.Tensor) -> torch.Tensor:
        """
        Project BUS vector → LLM hidden-space embedding.

        Args:
            bus_vector: [bus_dim] or [B, bus_dim]

        Returns:
            [llm_dim] or [B, llm_dim]
        """
        x = bus_vector
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, bus_dim]

        if x.size(-1) != self.bus_dim:
            raise ValueError(
                f"DecoderAdapter expected last dim={self.bus_dim}, "
                f"got {x.size(-1)}"
            )

        out = self.proj(x)  # [B, llm_dim]
        return out if out.size(0) > 1 else out.squeeze(0)

    # ---------------- prompt formatting ----------------
    def format_prompt(self, caption: str, question: str) -> str:
        """
        Build a textual prompt for the LLM given an image caption and question.

        This is used both in:
          - pipeline.__call__ (inference)
          - Trainer._format_prompt(...) (training with student-only setup)
        """
        parts = []
        caption = caption.strip()
        question = question.strip()

        if caption:
            parts.append(f"Image: {caption}")
        parts.append(f"Question: {question}")
        parts.append("Answer:")

        # Space after "Answer:" encourages cleaner generation
        return "\n".join(parts) + " "

    # ---------------- BUSPacket helper ----------------
    @torch.no_grad()
    def prepare(self, packet: BUSPacket, question: str) -> Tuple[torch.Tensor, str]:
        """
        Convenience utility:
        BUSPacket + question → (LLM soft prompt embedding, textual prompt)

        Args:
            packet: BUSPacket containing vector + text
            question: VQA-style question string

        Returns:
            (llm_embedding, prompt_str)
        """
        bus_vector = packet.get_vector()  # torch.Tensor [bus_dim]
        caption = packet.get_text()

        llm_embedding = self.forward(bus_vector)         # [llm_dim]
        prompt = self.format_prompt(caption, question)   # string
        return llm_embedding, prompt


# Quick test
if __name__ == "__main__":
    print("Testing Decoder Adapter...")

    adapter = DecoderAdapter(bus_dim=512, llm_dim=768)
    print(f"✅ Created adapter ({adapter.bus_dim} → {adapter.llm_dim})")

    # Single vector
    bus_vec = torch.randn(512)
    llm_emb = adapter(bus_vec)
    print(f"✅ Projection (single): {bus_vec.shape} → {llm_emb.shape}")

    # Batch
    bus_batch = torch.randn(4, 512)
    llm_batch = adapter(bus_batch)
    print(f"✅ Projection (batch): {bus_batch.shape} → {llm_batch.shape}")

    # Prompt formatting
    prompt = adapter.format_prompt("a cat on a mat", "What animal is on the mat?")
    print("\nPrompt:\n", prompt)
