"""
Decoder Adapter
Maps BUS packets (512D + text) to LLM inputs (4096D + prompt)
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from bus.schema import BUSPacket


class DecoderAdapter(nn.Module):
    """
    BUS → LLM adapter

    Projects 512D BUS vectors to 4096D LLM embedding space
    and formats text into prompts
    """

    def __init__(self, bus_dim: int = 512, llm_dim: int = 4096):
        super().__init__()

        self.bus_dim = bus_dim
        self.llm_dim = llm_dim

        hidden_dim = (bus_dim + llm_dim) // 2  # 2304

        self.projection = nn.Sequential(
            nn.Linear(bus_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, bus_vector: torch.Tensor) -> torch.Tensor:
        """
        Project BUS vector to LLM space (grad-preserving)
        Args:
            bus_vector: [B, 512] or [512]
        Returns:
            llm_embedding: [B, 4096] or [4096]
        """
        return self.projection(bus_vector)

    # ---------- NEW: graph-preserving helper for training ----------
    def forward_tensor(self, bus_tensor: torch.Tensor, question: str) -> torch.Tensor:
        """
        Graph-preserving variant used by the training path.
        Returns a single-sample soft prompt [4096] (or [B,4096] if input was batched).
        """
        if bus_tensor.dim() == 1:
            bus_tensor = bus_tensor.unsqueeze(0)  # [1, 512]

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        bus_tensor = bus_tensor.to(device=device, dtype=dtype)

        soft_prompt = self.forward(bus_tensor)  # [B, 4096]
        return soft_prompt.squeeze(0)

    # ---------- Inference path (packet decode) ----------
    def format_prompt(self, text: str, question: str) -> str:
        prompt = f"""[INST] You are a helpful AI assistant answering questions about images.

Image Context: {text}

Question: {question}

Provide a clear, concise answer. [/INST]"""
        return prompt

    def decode(self, packet: BUSPacket, question: str) -> tuple[torch.Tensor, str]:
        """
        Complete decoding: BUS packet → LLM inputs (inference-friendly; uses no_grad)
        Returns: (llm_embedding, formatted_prompt)
        """
        bus_vector = packet.get_vector()

        with torch.no_grad():
            llm_embedding = self.forward(bus_vector)

        prompt = self.format_prompt(packet.get_text(), question)
        return llm_embedding, prompt


# Quick test
if __name__ == "__main__":
    print("Testing Decoder Adapter...")

    adapter = DecoderAdapter()
    print(f"✅ Created adapter ({adapter.bus_dim} → {adapter.llm_dim})")

    bus_vector = torch.randn(512)
    llm_emb = adapter(bus_vector)
    print(f"✅ Projection: {bus_vector.shape} → {llm_emb.shape}")
