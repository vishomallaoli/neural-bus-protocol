"""
Neural BUS Pipeline
Orchestrates: Image + Question → BUS Packet → Answer or Logits

Teacher: Image + Question → BLIP-2 → LLM → Answer logits.
Student (ours): Image → ResNet-50 → BUS → LLM → Answer logits
"""

from __future__ import annotations

from typing import Union, Optional

import torch
from PIL import Image
from transformers import AutoTokenizer

from models.vision_encoder import VisionEncoder
from models.captioner import Captioner, MockCaptioner
from models.language_model import LanguageModel, MockLLM
from adapters.encoder import EncoderAdapter
from adapters.decoder import DecoderAdapter


class NeuralBUSPipeline:
    """
    Complete Neural BUS pipeline
    Flow: Image → Vision → Encoder → BUS → Decoder → LLM → Answer/logits

    Notes:
      - When return_logits=True (training), we SKIP captioning by default to avoid
        loading BLIP-2 twice (teacher + captioner). You can flip this by eagerly
        loading a Captioner below if you really want captions during training.
    """

    def __init__(self, device: str = "cpu", use_mock: bool = False):
        # ------------------------------------------------------------
        # Resolve device
        # ------------------------------------------------------------
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.use_mock = use_mock

        print("=" * 60)
        print("Initializing Neural BUS Pipeline")
        print("=" * 60)

        # ------------------------------------------------------------
        # [1/5] Vision Encoder
        # ------------------------------------------------------------
        print("\n[1/5] Loading Vision Encoder...")
        self.vision_encoder = VisionEncoder(pretrained=True, freeze=True).to(self.device)
        print(f"✅ ResNet-50 loaded (dim={self.vision_encoder.feature_dim})")

        # ------------------------------------------------------------
        # [2/5] Captioner (BLIP-2 or mock, lazy-loaded when real)
        # ------------------------------------------------------------
        print("\n[2/5] Preparing Captioner...")
        if use_mock:
            # In mock mode we can eagerly create a lightweight captioner
            self.captioner: Optional[MockCaptioner | Captioner] = MockCaptioner(
                device=self.device
            )
            print("✅ Mock Captioner ready")
        else:
            # Real BLIP-2 captioner is heavy → lazy-load on first inference
            self.captioner = None
            print("🕒 Will lazy-load BLIP-2 captioner on first inference")

        # ------------------------------------------------------------
        # [3/5] Encoder Adapter 2048 → 512 (BUS vector)
        # ------------------------------------------------------------
        print("\n[3/5] Loading Encoder Adapter...")
        # VisionEncoder.feature_dim is 2048 for ResNet-50
        self.encoder = EncoderAdapter(input_dim=2048, output_dim=512).to(self.device)
        print("✅ Encoder adapter (2048 → 512)")

        # ------------------------------------------------------------
        # [4/5] Student Language Model (DistilGPT2 or MockLLM)
        # ------------------------------------------------------------
        print("\n[4/5] Loading Language Model...")
        if use_mock:
            self.llm = MockLLM(device=self.device)
            # In mock mode we don't need a real tokenizer
            self.tokenizer = None
        else:
            # Use a lightweight student LLM instead of Mistral-7B
            # DistilGPT2: ~82M params, much faster to run and train adapters with.
            self.llm = LanguageModel(
                model_name="gpt2",
                device=self.device,
            )
            # Re-use the tokenizer provided by the LanguageModel wrapper
            self.tokenizer = self.llm.tokenizer

        print(f"✅ Language model loaded (hidden_size={self.llm.hidden_size})")

        # ------------------------------------------------------------
        # [5/5] Decoder Adapter 512 → hidden_size
        # ------------------------------------------------------------
        print("\n[5/5] Loading Decoder Adapter...")
        # Map 512D BUS vector into the LLM's hidden dimension (no hard-coded 4096)
        self.decoder = DecoderAdapter(
            bus_dim=512,
            llm_dim=self.llm.hidden_size,
        ).to(self.device)
        print(f"✅ Decoder adapter (512 → {self.llm.hidden_size})")

        print("\n" + "=" * 60)
        print("Pipeline Ready! 🚀")
        print("=" * 60 + "\n")

    
    @torch.inference_mode()
    def answer(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        question: str,
        max_new_tokens: int = 10,
    ) -> str:
        """
        Run the **student** (BUS + DistilGPT2) for a single image–question pair.

        image:
            - file path, or
            - PIL.Image, or
            - torch.Tensor in one of:
                [3,H,W], [H,W,3], [1,3,H,W], [B,3,H,W], [B,H,W,3]
        """
        # eval modes
        self.llm.model.eval()
        self.encoder.eval()
        self.decoder.eval()

        # 1) Vision → features
        features = self.vision_encoder.encode(image)   # [2048] or [B,2048]
        if features.dim() == 1:
            features = features.unsqueeze(0)           # [1,2048]

        # 2) BUS adapters
        bus_vec = self.encoder(features)               # [B,512]
        if bus_vec.dim() == 1:
            bus_vec = bus_vec.unsqueeze(0)             # just in case

        llm_emb = self.decoder(bus_vec)                # [B,H]
        if llm_emb.dim() == 1:
            llm_emb = llm_emb.unsqueeze(0)             # [1,H]

        # 3) Build prompt text (same format as training)
        prompt = self.decoder.format_prompt("", question)
        tok = self.llm.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = tok["input_ids"]                   # [1,T]

        # 4) Token embeddings + soft prompt
        tok_emb = self.llm.model.get_input_embeddings()(input_ids)  # [1,T,H]
        tok_emb = tok_emb.to(self.device, dtype=llm_emb.dtype)

        # llm_emb is [1,H] → [1,1,H]
        soft = llm_emb.unsqueeze(1)                    # [1,1,H]
        inputs_embeds = torch.cat([soft, tok_emb], dim=1)  # [1,1+T,H]

        # 5) Attention mask (1 for all positions)
        attn_mask = torch.ones(
            inputs_embeds.size()[:2],
            dtype=torch.long,
            device=self.device,
        )

        # 6) Generate
        out = self.llm.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.llm.tokenizer.eos_token_id,
        )

        # We drop the first token position (corresponding to the soft prompt slot)
        text = self.llm.tokenizer.decode(out[0, 1:], skip_special_tokens=True)
        return text.strip()

    # ============================================================
    def __call__(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        question: str,
        save_packet: str = None,
        return_logits: bool = False,
    ):
        """
        Run full Neural BUS forward pass.

        Args:
            image: path, PIL.Image, or tensor accepted by VisionEncoder
            question: natural-language question
            save_packet: optional path to save BUS packet (inference only)
            return_logits: if True, return raw logits for KD training
        """

        print("\n" + "=" * 60)
        print("Processing through Neural BUS")
        print("=" * 60)

        # 1) Vision features
        print("\n[1/5] Extracting vision features...")
        vision_features = self.vision_encoder.encode(image)  # → [2048] or [B,2048]
        print(f"      ✓ Features: {vision_features.shape}")

        # 2) Caption (skip during KD training to avoid extra BLIP-2 load)
        if return_logits:
            caption = ""
            print("[2/5] Skipping caption generation for training (KD).")
        else:
            print("[2/5] Generating caption...")
            if self.captioner is None:
                # Lazy-load only when needed (non-mock)
                self.captioner = Captioner(device=self.device)
            caption = self.captioner.caption(image)
            print(f"      ✓ Caption: '{caption}'")

        # ---------------- TRAINING vs INFERENCE BRANCH ----------------

        # Training branch: differentiable adapters → LLM logits
        if return_logits and not self.use_mock:
            print("[3/5] Building BUS vector (train mode, grad enabled)...")
            bus_vector = self.encoder.forward(vision_features)            # [512] or [B,512]

            print("[4/5] Decoding to LLM soft prompt (train mode, grad enabled)...")
            # We use caption="" above; decoder can still format prompt
            prompt = self.decoder.format_prompt(caption, question)
            llm_embedding = self.decoder.forward(bus_vector)              # [4096] or [B,4096]

            # Tokenize prompt
            tok = self.tokenizer(prompt, return_tensors="pt")
            input_ids = tok["input_ids"].to(self.device)
            attn_mask = tok.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

            # Token embeddings from the (frozen) LLM
            tok_emb = self.llm.model.get_input_embeddings()(input_ids)   # [B, T, H]

            # Ensure shapes / dtypes are aligned
            if llm_embedding.dim() == 1:
                llm_embedding = llm_embedding.unsqueeze(0)               # [1, H]
            llm_embedding = llm_embedding.to(tok_emb.device, dtype=tok_emb.dtype)
            soft_prompt = llm_embedding.unsqueeze(1)                      # [B, 1, H]

            # Concatenate soft prompt in front of token embeddings
            inputs_embeds = torch.cat([soft_prompt, tok_emb.detach()], dim=1)  # [B, 1+T, H]
            # Extend attention mask
            extra = torch.ones((attn_mask.size(0), 1), dtype=attn_mask.dtype, device=attn_mask.device)
            attn_mask = torch.cat([extra, attn_mask], dim=1)

            # Freeze LLM weights, but DO NOT wrap forward in no_grad()
            for p in self.llm.model.parameters():
                p.requires_grad_(False)

            print("[5/5] Forwarding LLM for logits (train mode)...")
            outputs = self.llm.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
            return outputs.logits  # [B, 1+T, vocab]

        # Mock training branch: fabricate logits with grad so KD works
        if return_logits and self.use_mock:
            print("[3-5] Mock train path → returning random logits with grad.")
            # Use a large-ish vocab (matches typical HF models)
            return torch.randn(1, 50, 50304, device=self.device, requires_grad=True)

        # Inference branch: create packet + decode (non-grad) then generate
        print("[3/5] Creating BUS packet...")
        packet = self.encoder.encode(
            vision_features=vision_features,
            text=caption,
            intent="vqa",
            question=question,
        )
        kb = packet.size_bytes() / 1024.0 if hasattr(packet, "size_bytes") else 0.0
        print(f"      ✓ Packet: {kb:.2f}KB")
        if save_packet and hasattr(packet, "to_json"):
            packet.to_json(save_packet)
            print(f"      ✓ Saved: {save_packet}")

        print("[4/5] Decoding for LLM (inference mode)...")
        with torch.no_grad():
            llm_embedding, prompt = self.decoder.decode(packet, question)
        print(f"      ✓ LLM embedding: {llm_embedding.shape}")
        print(f"      ✓ Prompt length: {len(prompt)} chars")

        print("[5/5] Generating answer...")
        if self.use_mock:
            try:
                from models.mock_answerer import answer_vqa
                answer = answer_vqa(image, caption, question)
            except Exception as e:
                print(f"ℹ️ Smart-mock fallback ({e}); using text mock.")
                answer = self.llm.generate(prompt)
        else:
            with torch.no_grad():
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                output_ids = self.llm.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                )
                answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        print(f"      ✓ Answer: '{answer}'")
        print("\n" + "=" * 60)
        print("Processing Complete! ✅")
        print("=" * 60 + "\n")

        return {
            "answer": answer,
            "packet": packet,
            "caption": caption,
            "vision_features": vision_features,
            "llm_embedding": llm_embedding,
            "prompt": prompt,
        }
