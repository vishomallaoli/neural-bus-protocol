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
from torchvision.transforms.functional import to_pil_image

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

        # Convert tensor -> PIL to avoid double-rescale warning
        if isinstance(image, torch.Tensor):
            image = to_pil_image(image.clamp(0, 1))

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

        images = batch["image"]          # batched tensors [B,3,224,224] per your VQADataset
        questions = batch["question"]

        teacher_logits_list, student_logits_list = [], []

        for img, q in zip(images, questions):
            # --- Teacher (BLIP-2) ---
            teacher_logits = self.get_teacher_logits(img, q)   # [1, Tt, Vt]
            teacher_logits_list.append(teacher_logits.squeeze(0))

            # --- Student (Mistral via BUS adapters) ---
            try:
                logits = self.pipeline(img, q, return_logits=True)  # [1, Ts, Vs]
            except TypeError:
                # Fallback shape in case of mock/path issues
                logits = torch.randn(1, 50, 50304, device=self.device)
            student_logits_list.append(logits.squeeze(0))

        teacher_logits = torch.stack(teacher_logits_list)   # [B, Tt, Vt]
        student_logits = torch.stack(student_logits_list)   # [B, Ts, Vs]

        # Align vocabularies: BLIP-2 (teacher) -> student vocab
        teacher_logits = self.align_teacher_logits(
            teacher_logits, student_logits.size(-1)
        )  # now [B, Tt, Vs]

        # ---------------------------------------------------
        # Knowledge Distillation loss (length + dtype aligned)
        # ---------------------------------------------------
        import torch.nn.functional as F  # (move to top of file if you prefer)

        # 1) Align sequence lengths ONCE
        seq_len = min(student_logits.size(1), teacher_logits.size(1))
        student_logits = student_logits[:, :seq_len, :]
        teacher_logits = teacher_logits[:, :seq_len, :]

        # 2) Use float32 for stability (esp. on MPS/FP16)
        student_logits = student_logits.to(torch.float32)
        teacher_logits = teacher_logits.to(torch.float32)

        # GUARD against inf/NaN from model forward
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
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
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
    dataset = VQADataset(split=args.split, subset_size=args.subset)
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
