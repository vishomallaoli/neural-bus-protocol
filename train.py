#!/usr/bin/env python3
"""
Training Script (Research Grade)
Neural BUS Adapters via Knowledge Distillation from BLIP-2 → Mistral
FAST VERSION: batched teacher/student forward + prompt truncation
"""

from pathlib import Path
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

from pipeline import NeuralBUSPipeline
from data.vqa import VQADataset


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _pad_logits_to_max(seq_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Pad a list of [T, V] tensors to [B, Tmax, V] with zeros.
    """
    if len(seq_list) == 0:
        raise ValueError("seq_list is empty")
    maxT = max(t.size(0) for t in seq_list)
    V = seq_list[0].size(-1)
    device, dtype = seq_list[0].device, seq_list[0].dtype
    out = torch.zeros(len(seq_list), maxT, V, device=device, dtype=dtype)
    for i, t in enumerate(seq_list):
        out[i, : t.size(0)] = t
    return out


# ============================================================
# Trainer
# ============================================================
class Trainer:
    """Train encoder & decoder adapters with vocabulary-aligned KD loss."""

    def __init__(
        self,
        pipeline: NeuralBUSPipeline,
        learning_rate: float = 1e-4,
        max_len_teacher: int = 64,
        max_len_student: int = 128,
    ):
        self.pipeline = pipeline
        self.device = pipeline.device
        self.max_len_teacher = int(max_len_teacher)
        self.max_len_student = int(max_len_student)

        self.optimizer = optim.AdamW(
            [
                {"params": pipeline.encoder.parameters()},
                {"params": pipeline.decoder.parameters()},
            ],
            lr=learning_rate,
        )

        # Freeze LLM parameters (we still forward it)
        for p in self.pipeline.llm.model.parameters():
            p.requires_grad_(False)

        # -------------------------------------------------------
        # Teacher (BLIP-2) setup
        # -------------------------------------------------------
        try:
            from transformers import (
                Blip2Processor,
                Blip2ForConditionalGeneration,
                AutoTokenizer,
            )

            print("\n[Teacher] Loading BLIP-2 teacher model...")
            # Fast image processor where available
            self.teacher_processor = Blip2Processor.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                use_fast=True,
            )

            # fp16 only on CUDA; fp32 elsewhere
            _dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.teacher_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                dtype=_dtype,  # deprecation-safe; replaces torch_dtype
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.teacher_model.eval()
            print("✅ Teacher model loaded successfully")

            # ---------------- vocab alignment -------------------
            print("[Align] Building teacher→student vocabulary map...")
            self.teacher_tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.student_tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2"
            )

            Vt = int(self.teacher_model.config.vocab_size)  # teacher model vocab size
            student_vocab = self.student_tokenizer.get_vocab()
            unk_id = self.student_tokenizer.unk_token_id

            toks = self.teacher_tokenizer.convert_ids_to_tokens(list(range(Vt)))
            mapping = torch.full((Vt,), fill_value=unk_id, dtype=torch.long)
            for i, tok in enumerate(toks):
                if tok is None:
                    continue
                mapping[i] = student_vocab.get(tok, unk_id)

            self.teacher_to_student_map = mapping.to(self.device)
            print(f"✅ Vocabulary map built using model vocab: {Vt} teacher ids aligned.")
            self.use_teacher = True

        except Exception as e:
            print(f"⚠️  Could not load BLIP-2 teacher model ({e}). Using random logits.")
            self.use_teacher = False

        print(f"✅ Trainer initialized (lr={learning_rate})")

    # ---------------- batched forwards (FAST) ----------------
    def _batch_teacher_logits(self, images: torch.Tensor, questions: List[str]) -> torch.Tensor:
        """
        Batched teacher forward:
            images: [B,3,224,224] in [0,1]
            returns: [B, Tt, Vt]
        """
        if not self.use_teacher:
            B = images.size(0)
            return torch.randn(B, 50, 32000, device=self.device)

        pil_list = [to_pil_image(images[i].clamp(0, 1)) for i in range(images.size(0))]
        prompts = [f"Question: {q}\nAnswer:" for q in questions]

        inputs = self.teacher_processor(
            images=pil_list,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len_teacher,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.teacher_model(**inputs)
        return outputs.logits  # [B, Tt, Vt]

    def _batch_student_logits(self, images: torch.Tensor, questions: List[str]) -> torch.Tensor:
        """
        Batched student forward with frozen LLM:
            images: [B,3,224,224]
            returns: [B, Ts, Vs]
        """
        # 1) Vision features (batched)
        vision_features = self.pipeline.vision_encoder.encode(images)  # [B,2048]

        # 2) BUS vector via adapters (grad flows through adapters)
        bus_vector = self.pipeline.encoder.forward(vision_features)    # [B,512]
        llm_embedding = self.pipeline.decoder.forward(bus_vector)      # [B,4096]

        # 3) Build prompts (skip captions during training)
        prompts = [self.pipeline.decoder.format_prompt("", q) for q in questions]

        # 4) Tokenize as a batch
        tok = self.pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len_student,
        )
        input_ids = tok["input_ids"].to(self.device)
        attn_mask = tok.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

        # 5) Token embeddings from frozen LLM
        tok_emb = self.pipeline.llm.model.get_input_embeddings()(input_ids)  # [B,T,H]

        # 6) Soft prompt
        llm_embedding = llm_embedding.to(tok_emb.device, dtype=tok_emb.dtype)  # [B,H]
        soft_prompt = llm_embedding.unsqueeze(1)                                # [B,1,H]

        # 7) Concatenate and forward LLM (frozen) — keep grads for adapters
        inputs_embeds = torch.cat([soft_prompt, tok_emb.detach()], dim=1)       # [B,1+T,H]
        extra = torch.ones((attn_mask.size(0), 1), dtype=attn_mask.dtype, device=attn_mask.device)
        attn_mask = torch.cat([extra, attn_mask], dim=1)

        outputs = self.pipeline.llm.model(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
        return outputs.logits  # [B, 1+T, Vs]

    # ---------------- alignment ----------------
    def align_teacher_logits(self, teacher_logits: torch.Tensor, student_vocab_size: int) -> torch.Tensor:
        """
        Project teacher logits (vocab_t) → student space (vocab_s) via scatter_add.
        Rebuild mapping if sizes diverge.
        """
        Vt = teacher_logits.size(-1)

        if (not hasattr(self, "teacher_to_student_map")) or (
            self.teacher_to_student_map.numel() != Vt
        ):
            print("ℹ️ Rebuilding teacher→student map to match current teacher vocab size...")
            student_vocab = self.student_tokenizer.get_vocab()
            unk_id = self.student_tokenizer.unk_token_id
            toks = self.teacher_tokenizer.convert_ids_to_tokens(list(range(Vt)))
            mapping = torch.full((Vt,), fill_value=unk_id, dtype=torch.long)
            for i, tok in enumerate(toks):
                if tok is None:
                    continue
                mapping[i] = student_vocab.get(tok, unk_id)
            self.teacher_to_student_map = mapping.to(self.device)

        B, Tt, _ = teacher_logits.shape
        Vs = student_vocab_size
        aligned = torch.zeros(B, Tt, Vs, device=teacher_logits.device, dtype=teacher_logits.dtype)

        idx = self.teacher_to_student_map.view(1, 1, -1).expand(B, Tt, -1)
        aligned.scatter_add_(2, idx, teacher_logits)
        return aligned

    # ---------------- train step ----------------
    def train_step(self, batch):
        self.optimizer.zero_grad()

        images = batch["image"]          # [B,3,224,224]
        questions = batch["question"]    # list[str]

        # Batched forwards (FAST)
        teacher_logits = self._batch_teacher_logits(images, questions)    # [B,Tt,Vt]
        student_logits = self._batch_student_logits(images, questions)    # [B,Ts,Vs]

        # Align vocabularies: BLIP-2 (teacher) -> student vocab
        teacher_logits = self.align_teacher_logits(
            teacher_logits, student_logits.size(-1)
        )  # [B,Tt,Vs]

        # ---------------------------------------------------
        # Knowledge Distillation loss (length + dtype aligned)
        # ---------------------------------------------------
        import torch.nn.functional as F

        # 1) Align sequence lengths
        seq_len = min(student_logits.size(1), teacher_logits.size(1))
        student_logits = student_logits[:, :seq_len, :]
        teacher_logits = teacher_logits[:, :seq_len, :]

        # 2) Float32 for stability
        student_logits = student_logits.to(torch.float32)
        teacher_logits = teacher_logits.to(torch.float32)

        # Guard against inf/NaN
        student_logits = torch.nan_to_num(student_logits, nan=0.0, posinf=50.0, neginf=-50.0)
        teacher_logits = torch.nan_to_num(teacher_logits, nan=0.0, posinf=50.0, neginf=-50.0)

        student_logits = student_logits.clamp(min=-50.0, max=50.0)
        teacher_logits = teacher_logits.clamp(min=-50.0, max=50.0)

        # 3) Temperature softening
        temperature = 2.0
        B, T, V = student_logits.shape
        student_logp = F.log_softmax(student_logits / temperature, dim=-1).view(B * T, V)
        teacher_p    = F.softmax(teacher_logits  / temperature, dim=-1).view(B * T, V)

        loss = F.kl_div(student_logp, teacher_p, reduction="batchmean") * (temperature ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.pipeline.encoder.parameters()) + list(self.pipeline.decoder.parameters()),
            1.0,
        )
        self.optimizer.step()

        return loss.item()

    # ---------------- epoch ----------------
    def train_epoch(self, dataloader, epoch):
        self.pipeline.encoder.train()
        self.pipeline.decoder.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            loss = self.train_step(batch)
            total_loss += loss
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        return total_loss / max(1, len(dataloader))

    # ---------------- ckpt ----------------
    def save_checkpoint(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": self.pipeline.encoder.state_dict(),
                "decoder": self.pipeline.decoder.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"✅ Saved checkpoint: {path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train Neural BUS (FAST KD)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--subset", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-len-teacher", type=int, default=64, help="truncate teacher text tokens")
    parser.add_argument("--max-len-student", type=int, default=128, help="truncate student prompt tokens")

    args = parser.parse_args()

    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    print("\n" + "=" * 70)
    print(" " * 25 + "NEURAL BUS TRAINING (FAST)")
    print("=" * 70)
    print(f"\nEpochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Subset: {args.subset}")
    print(f"Device (requested): {args.device}")

    print("\nLoading dataset...")
    dataset = VQADataset(split=args.split, subset_size=args.subset)
    pin_memory = (args.device == "cuda")
    persistent = args.num_workers > 0
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=persistent,
        pin_memory=pin_memory,
    )

    print("\nInitializing pipeline...")
    pipeline = NeuralBUSPipeline(device=args.device, use_mock=False)  # no demo/mock mode
    print(f"Resolved device in pipeline: {pipeline.device}")

    trainer = Trainer(
        pipeline,
        learning_rate=args.lr,
        max_len_teacher=args.max_len_teacher,
        max_len_student=args.max_len_student,
    )

    print("\n" + "=" * 70)
    print(" " * 28 + "TRAINING")
    print("=" * 70 + "\n")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)
        avg_loss = trainer.train_epoch(dataloader, epoch)
        print(f"Average Loss: {avg_loss:.4f}")

        ckpt = f"{args.checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
        trainer.save_checkpoint(ckpt)

    print("\n" + "=" * 70)
    print("Training Complete! ✅")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
