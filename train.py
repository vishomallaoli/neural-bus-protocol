#!/usr/bin/env python3
"""
Training Script (Supervised, Student-Only)

Neural BUS Adapters trained with a frozen LLM (e.g., DistilGPT2).
- No BLIP-2 / BLIP-VQA in the training loop.
- Only the BUS encoder/decoder adapters receive gradients.
- The LLM remains frozen and is used via a soft prompt (BUS vector).

Expected VQADataset batch format:
    {
        "image":    torch.Tensor [B, 3, H, W],
        "question": list[str],
        "answer":   list[str],
    }
"""

from pathlib import Path
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipeline import NeuralBUSPipeline
from data.vqa import VQADataset


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Trainer
# ============================================================
class Trainer:
    """
    Train encoder & decoder adapters with supervised loss
    against ground-truth answers, using a frozen LLM.

    Trainable:
      - pipeline.encoder  (BUS encoder)
      - pipeline.decoder  (BUS → LLM hidden size)

    Frozen:
      - pipeline.vision_encoder
      - pipeline.llm.model (e.g., DistilGPT2)
    """

    def __init__(
        self,
        pipeline: NeuralBUSPipeline,
        learning_rate: float = 1e-4,
        max_len_student: int = 128,
    ):
        self.pipeline = pipeline
        self.device = pipeline.device
        self.max_len_student = int(max_len_student)

        # Optimizer over BUS adapters only
        self.optimizer = optim.AdamW(
            [
                {"params": self.pipeline.encoder.parameters()},
                {"params": self.pipeline.decoder.parameters()},
            ],
            lr=learning_rate,
        )

        # Freeze LLM parameters (we still forward it)
        for p in self.pipeline.llm.model.parameters():
            p.requires_grad_(False)

        # Convenience handle: we always use the LLM's tokenizer
        self.tokenizer = self.pipeline.llm.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

        print("\n[Trainer] Initialized (student-only, supervised)")
        print(f"  Device: {self.device}")
        print(f"  Max student length: {self.max_len_student}")
        print(f"  Pad token id: {self.pad_token_id}")

    # ---------------- internal helpers ----------------
    def _format_prompt(self, question: str) -> str:
        """
        Use the same prompt formatting as the decoder for consistency.
        We assume decoder.format_prompt(caption, question) exists.
        For training we pass an empty caption.
        """
        return self.pipeline.decoder.format_prompt("", question)

    def _build_texts(
        self,
        questions: List[str],
        answers: List[str],
    ) -> (List[str], List[str]):
        """
        Build:
          - prefixes: prompt text up to 'Answer:' (or equivalent)
          - full_texts: prefix + actual ground-truth answer
        """
        prefixes = [self._format_prompt(q) for q in questions]
        full_texts = [f"{p} {a}" for p, a in zip(prefixes, answers)]
        return prefixes, full_texts

    # ---------------- core training step ----------------
    def train_step(self, batch) -> float:
        self.pipeline.encoder.train()
        self.pipeline.decoder.train()
        self.optimizer.zero_grad()

        if "answer" not in batch:
            raise KeyError(
                f"Batch is missing 'answer' field for supervised training. "
                f"Got keys: {list(batch.keys())}"
            )

        images = batch["image"]          # [B,3,H,W]
        questions = batch["question"]    # list[str]
        answers = batch["answer"]        # list[str]

        # ---------------- vision → BUS → soft prompt ----------------
        # 1) Vision features
        vision_features = self.pipeline.vision_encoder.encode(images)   # [B, Dv]

        # 2) BUS vector via adapters
        bus_vector = self.pipeline.encoder(vision_features)             # [B, 512] (or similar)
        llm_embedding = self.pipeline.decoder(bus_vector)               # [B, H]

        # ---------------- text encoding ----------------
        prefixes, full_texts = self._build_texts(questions, answers)

        tok = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len_student,
        )
        input_ids = tok["input_ids"].to(self.device)         # [B, T]
        attn_mask = tok["attention_mask"].to(self.device)    # [B, T]

        B, T = input_ids.shape

        # Build labels: ignore prompt tokens with -100
        labels = input_ids.clone()

        for i, prefix in enumerate(prefixes):
            with torch.no_grad():
                prefix_ids = self.tokenizer(
                    prefix,
                    truncation=True,
                    max_length=self.max_len_student,
                    return_tensors="pt",
                )["input_ids"][0]
                prefix_len = int(prefix_ids.size(0))

            seq_len = int(attn_mask[i].sum().item())
            cut = min(prefix_len, seq_len)
            labels[i, :cut] = -100  # ignore prompt tokens

        # ---------------- LLM forward with soft prompt ----------------
        tok_emb = self.pipeline.llm.model.get_input_embeddings()(input_ids)  # [B, T, H]
        tok_emb = tok_emb.detach()

        llm_embedding = llm_embedding.to(tok_emb.device, dtype=tok_emb.dtype)  # [B, H]
        soft_prompt = llm_embedding.unsqueeze(1)                                # [B, 1, H]

        inputs_embeds = torch.cat([soft_prompt, tok_emb], dim=1)                # [B, 1+T, H]

        extra_mask = torch.ones(
            (attn_mask.size(0), 1),
            dtype=attn_mask.dtype,
            device=attn_mask.device,
        )
        attn_mask_ext = torch.cat([extra_mask, attn_mask], dim=1)               # [B, 1+T]

        labels_ext = torch.cat(
            [
                torch.full(
                    (labels.size(0), 1),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device,
                ),
                labels,
            ],
            dim=1,
        )  # [B, 1+T]

        outputs = self.pipeline.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask_ext,
            labels=labels_ext,
        )

        loss = outputs.loss

        # Backprop only through adapters
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.pipeline.encoder.parameters())
            + list(self.pipeline.decoder.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        return float(loss.detach().cpu())

    # ---------------- epoch loop ----------------
    def train_epoch(self, dataloader, epoch: int) -> float:
        self.pipeline.encoder.train()
        self.pipeline.decoder.train()

        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            loss = self.train_step(batch)
            total_loss += loss
            pbar.set_postfix(loss=loss)

        avg_loss = total_loss / max(1, len(dataloader))
        return avg_loss

    # ---------------- checkpointing ----------------
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
    parser = argparse.ArgumentParser(description="Train Neural BUS (Supervised Student-Only)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--subset", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-len-student",
        type=int,
        default=128,
        help="Truncate student prompt+answer tokens to this length.",
    )

    args = parser.parse_args()

    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    print("\n" + "=" * 70)
    print(" " * 20 + "NEURAL BUS TRAINING (STUDENT-ONLY)")
    print("=" * 70)
    print(f"\nEpochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Subset: {args.subset}")
    print(f"Split: {args.split}")
    print(f"Device (requested): {args.device}")

    # ---------------- dataset & dataloader ----------------
    print("\nLoading dataset...")
    subset_size = args.subset if args.subset > 0 else None
    dataset = VQADataset(split=args.split, subset_size=subset_size)

    pin_memory = (args.device == "cuda")
    persistent = args.num_workers > 0

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    # ---------------- pipeline ----------------
    print("\nInitializing pipeline...")
    pipeline = NeuralBUSPipeline(device=args.device, use_mock=False)
    print(f"Resolved device in pipeline: {pipeline.device}")

    # Print trainable parameter counts for sanity check
    print("Trainable params - encoder:", count_trainable(pipeline.encoder))
    print("Trainable params - decoder:", count_trainable(pipeline.decoder))
    print("Trainable params - vision:", count_trainable(pipeline.vision_encoder))
    print("Trainable params - llm:", count_trainable(pipeline.llm.model))

    trainer = Trainer(
        pipeline=pipeline,
        learning_rate=args.lr,
        max_len_student=args.max_len_student,
    )

    print("\nStarting training...\n")

    # ---- loss history for plotting later ----
    loss_history = []

    for epoch in range(1, args.epochs + 1):
        print("-" * 70)
        avg_loss = trainer.train_epoch(dataloader, epoch)
        print(f"Average Loss (epoch {epoch}): {avg_loss:.4f}")

        loss_history.append(avg_loss)

        ckpt_path = f"{args.checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
        trainer.save_checkpoint(ckpt_path)

    print("\nLoss history by epoch:", loss_history)

    print("\n" + "=" * 70)
    print("Training Complete! ✅")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

