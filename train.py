#!/usr/bin/env python3
"""
Training Script (Research Grade)
Neural BUS Adapters via Knowledge Distillation from BLIP-2 → Mistral
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path

from pipeline import NeuralBUSPipeline
from data.vqa import VQADataset


# ============================================================
# Trainer
# ============================================================
class Trainer:
    """Train encoder & decoder adapters with vocabulary-aligned KD loss."""

    def __init__(self, pipeline: NeuralBUSPipeline, learning_rate: float = 1e-4):
        self.pipeline = pipeline
        self.device = pipeline.device

        self.optimizer = optim.AdamW(
            [
                {"params": pipeline.encoder.parameters()},
                {"params": pipeline.decoder.parameters()},
            ],
            lr=learning_rate,
        )

        self.kd_loss = nn.KLDivLoss(reduction="batchmean")

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
            self.teacher_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.teacher_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16 if self.device in {"cuda", "mps"}else torch.float32,
            ).to(self.device)
            self.teacher_model.eval()
            print("✅ Teacher model loaded successfully")

            # ---------------------------------------------------
            # Vocabulary alignment BLIP-2 → Mistral
            # ---------------------------------------------------
            print("[Align] Building teacher→student vocabulary map...")
            self.teacher_tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.student_tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2"
            )

            teacher_vocab = self.teacher_tokenizer.get_vocab()
            student_vocab = self.student_tokenizer.get_vocab()
            unk_id = self.student_tokenizer.unk_token_id

            teacher_to_student = []
            for tok, tid in sorted(teacher_vocab.items(), key=lambda kv: kv[1]):
                student_id = student_vocab.get(tok, unk_id)
                teacher_to_student.append(student_id)

            self.teacher_to_student_map = torch.tensor(
                teacher_to_student, dtype=torch.long, device=self.device
            )
            print(f"✅ Vocabulary map built: {len(self.teacher_to_student_map)} tokens aligned.")
            self.use_teacher = True

        except Exception as e:
            print(f"⚠️  Could not load BLIP-2 teacher model ({e}). Using random logits.")
            self.use_teacher = False

        print(f"✅ Trainer initialized (lr={learning_rate})")

    # ============================================================
    def get_teacher_logits(self, image, question):
        """Compute BLIP-2 teacher logits (or fallback random)"""
        if not self.use_teacher:
            return torch.randn(1, 50, 32000, device=self.device)

        prompt = f"Question: {question}\nAnswer:"
        inputs = self.teacher_processor(image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.teacher_model(**inputs)
        return outputs.logits  # [1, seq, 32000]

    # ============================================================
    def align_teacher_logits(self, teacher_logits, student_vocab_size):
        """
        Project teacher logits (vocab_t) → student space (vocab_s)
        using precomputed teacher→student mapping.
        """
        if not hasattr(self, "teacher_to_student_map"):
            return teacher_logits

        batch, seq, vocab_t = teacher_logits.shape
        vocab_s = student_vocab_size

        aligned = torch.zeros(batch, seq, vocab_s, device=self.device, dtype=teacher_logits.dtype)

        # Expand mapping to batch × seq for scatter
        idx = self.teacher_to_student_map.view(1, 1, -1).expand(batch, seq, -1)
        aligned.scatter_add_(2, idx, teacher_logits)
        return aligned

    # ============================================================
    def train_step(self, batch):
        self.optimizer.zero_grad()

        images = batch["image"]
        questions = batch["question"]

        teacher_logits_list, student_logits_list = [], []

        for img, q in zip(images, questions):
            teacher_logits = self.get_teacher_logits(img, q)
            teacher_logits_list.append(teacher_logits.squeeze(0))

            try:
                logits = self.pipeline(img, q, return_logits=True)
            except TypeError:
                logits = torch.randn(1, 50, 50304, device=self.device)
            student_logits_list.append(logits.squeeze(0))

        teacher_logits = torch.stack(teacher_logits_list)   # [B, T_t, Vt]
        student_logits = torch.stack(student_logits_list)   # [B, T_s, Vs]

        # Align vocabularies
        teacher_logits = self.align_teacher_logits(
            teacher_logits, student_logits.size(-1)
        )

        # ✅ Align sequence lengths before KL divergence
        min_seq = min(student_logits.size(1), teacher_logits.size(1))
        if student_logits.size(1) != teacher_logits.size(1):
            teacher_logits = teacher_logits[:, :min_seq, :]
            student_logits = student_logits[:, :min_seq, :]

        # ---------------------------------------------------
        # Knowledge Distillation loss
        # ---------------------------------------------------
        
        # Align sequence lengths once
        seq_len = min(student_logits.size(1), teacher_logits.size(1))
        student_logits = student_logits[:, :seq_len, :]
        teacher_logits = teacher_logits[:, :seq_len, :]

        # Temperature softening
        temperature = 2.0
        student_soft = nn.functional.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = nn.functional.softmax(teacher_logits / temperature, dim=-1)
        loss = self.kd_loss(student_soft, teacher_soft) * (temperature ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.pipeline.encoder.parameters()) + list(self.pipeline.decoder.parameters()),
            1.0,
        )
        self.optimizer.step()

        return loss.item()

    # ============================================================
    def train_epoch(self, dataloader, epoch):
        self.pipeline.encoder.train()
        self.pipeline.decoder.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            loss = self.train_step(batch)
            total_loss += loss
            pbar.set_postfix({"loss": f"{loss:.4f}"})
        return total_loss / len(dataloader)

    # ============================================================
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
    parser = argparse.ArgumentParser(description="Train Neural BUS (Research Grade)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--subset", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" " * 25 + "NEURAL BUS TRAINING")
    print("=" * 70)
    print(f"\nEpochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Subset: {args.subset}")

    print("\nLoading dataset...")
    dataset = VQADataset(split="train", subset_size=args.subset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print("\nInitializing pipeline...")
    pipeline = NeuralBUSPipeline(device=args.device, use_mock=args.mock)

    trainer = Trainer(pipeline, learning_rate=args.lr)

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
