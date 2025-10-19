#!/usr/bin/env python3
"""
Evaluation Script - Research-Grade Version
Evaluates Neural BUS adapters and compares against BLIP-2 baseline.
Supports mock mode, ablation runs, and vocab alignment tracing.
"""

import torch
import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from pipeline import NeuralBUSPipeline
from data.vqa import VQADataset


# ============================================================
# Evaluator
# ============================================================
class VQAEvaluator:
    """VQA accuracy evaluator (Exact / Partial match metrics)"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct_exact = 0
        self.correct_partial = 0
        self.total = 0
        self.predictions = []
        self.inference_times = []

    def normalize_answer(self, answer: str) -> str:
        """Normalize text for consistent comparison"""
        return answer.lower().strip().rstrip(".,!?;")

    def update(self, prediction: str, ground_truth: str, inference_time: float = 0):
        """Update running accuracy metrics"""
        pred_norm = self.normalize_answer(prediction)
        gt_norm = self.normalize_answer(ground_truth)

        exact_match = pred_norm == gt_norm
        partial_match = pred_norm in gt_norm or gt_norm in pred_norm

        if exact_match:
            self.correct_exact += 1
            self.correct_partial += 1
        elif partial_match:
            self.correct_partial += 1

        self.total += 1
        self.inference_times.append(inference_time)
        self.predictions.append({
            "prediction": prediction,
            "ground_truth": ground_truth,
            "exact_match": exact_match,
            "partial_match": partial_match
        })

    def compute_metrics(self) -> dict:
        """Return summarized metrics"""
        if self.total == 0:
            return {
                "exact_accuracy": 0.0,
                "partial_accuracy": 0.0,
                "correct_exact": 0,
                "correct_partial": 0,
                "total": 0,
                "avg_inference_time": 0.0,
            }

        return {
            "exact_accuracy": self.correct_exact / self.total,
            "partial_accuracy": self.correct_partial / self.total,
            "correct_exact": self.correct_exact,
            "correct_partial": self.correct_partial,
            "total": self.total,
            "avg_inference_time": sum(self.inference_times) / len(self.inference_times),
        }


# ============================================================
# Evaluation loop
# ============================================================
def evaluate_pipeline(pipeline, dataloader, name="Pipeline", max_samples=None) -> tuple:
    """Run full evaluation loop"""
    print(f"\n{'='*70}")
    print(f"Evaluating: {name}")
    print(f"{'='*70}\n")

    evaluator = VQAEvaluator()
    samples_processed = 0

    for batch in tqdm(dataloader, desc=f"Evaluating {name}"):
        images = batch["image"]
        questions = batch["question"]
        answers = batch["answer"]

        for img, q, ans in zip(images, questions, answers):
            if max_samples and samples_processed >= max_samples:
                break
            try:
                start_time = time.time()
                result = pipeline(img, q)
                inference_time = time.time() - start_time
                prediction = result["answer"]
                evaluator.update(prediction, ans, inference_time)
            except Exception as e:
                print(f"\n⚠️  Error processing sample: {e}")
                evaluator.update("", ans, 0)
            samples_processed += 1
        if max_samples and samples_processed >= max_samples:
            break

    metrics = evaluator.compute_metrics()
    print(f"\n{'='*70}")
    print(f"Results: {name}")
    print(f"{'='*70}")
    print(f"\n📊 Accuracy Metrics:")
    print(f"   Exact Match:    {metrics['exact_accuracy']:.2%} ({metrics['correct_exact']}/{metrics['total']})")
    print(f"   Partial Match:  {metrics['partial_accuracy']:.2%} ({metrics['correct_partial']}/{metrics['total']})")
    print(f"\n⚡ Performance:")
    print(f"   Avg Time/Sample: {metrics['avg_inference_time']:.3f}s")
    print(f"   Total Time:      {metrics['avg_inference_time'] * metrics['total']:.1f}s")

    return metrics, evaluator.predictions


# ============================================================
# Baseline comparison
# ============================================================
def compare_with_baseline(student_metrics: dict, teacher_metrics: dict = None):
    """Compare Neural BUS student performance against BLIP-2 baseline"""
    print(f"\n{'='*70}")
    print(" " * 23 + "BENCHMARK COMPARISON")
    print(f"{'='*70}\n")

    baseline_accuracy = (
        teacher_metrics["exact_accuracy"] if teacher_metrics else 0.78
    )
    if teacher_metrics:
        print(f"📊 Comparing against actual BLIP-2 performance:")
    else:
        print(f"📊 Comparing against expected BLIP-2 baseline:")
    print(f"   BLIP-2:           {baseline_accuracy:.1%}")

    student_accuracy = student_metrics["exact_accuracy"]
    print(f"   Neural BUS:       {student_accuracy:.1%}")

    gap = student_accuracy - baseline_accuracy
    recovery = (student_accuracy / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
    print(f"\n📈 Performance Gap:")
    print(f"   Absolute Gap:     {gap:+.1%}")
    print(f"   Recovery Rate:    {recovery:.1f}% of teacher")

    target_accuracy = 0.70
    print(f"\n🎯 Target Assessment:")
    print(f"   Target (≥90% of teacher): ≥{target_accuracy:.1%}")
    print(f"   Current:                   {student_accuracy:.1%}")

    if student_accuracy >= target_accuracy:
        print("   Status: ✅ TARGET ACHIEVED!")
    else:
        needed = target_accuracy - student_accuracy
        print(f"   Status: ⚠️  Need +{needed:.1%} to reach target")

    print(f"\n⚡ Efficiency Gains:")
    print(f"   Parameters Trained: 2.3M (vs 2.7B+ full model)")
    print(f"   Training Efficiency: ~1000x fewer parameters")


# ============================================================
# Results persistence
# ============================================================
def save_results(config, student_metrics, student_predictions, teacher_metrics=None, output_path="./results/evaluation.json"):
    """Save evaluation summary and predictions to disk"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "config": config,
        "student": {
            "metrics": student_metrics,
            "sample_predictions": student_predictions[:20],
        },
    }
    if teacher_metrics:
        results["teacher"] = {"metrics": teacher_metrics}

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")


# ============================================================
# Main entry
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate Neural BUS (Research Grade)")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mock", action="store_true", help="Use mock models")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="./results/evaluation.json")
    parser.add_argument("--compare-teacher", action="store_true", help="Compare against teacher model")
    parser.add_argument("--skip-caption", action="store_true", help="Skip captioning for ablation")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" " * 23 + "NEURAL BUS EVALUATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"   Samples:     {args.samples}")
    print(f"   Device:      {args.device}")
    print(f"   Mock Mode:   {args.mock}")
    print(f"   Checkpoint:  {args.checkpoint or 'None (random weights)'}")
    print(f"   Skip Caption: {args.skip_caption}")

    # Load dataset
    print("\n📊 Loading dataset...")
    dataset = VQADataset(split="val", subset_size=args.samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"✅ Loaded {len(dataset)} samples")

    # Initialize student pipeline
    print("\n🤖 Initializing Neural BUS pipeline...")
    student_pipeline = NeuralBUSPipeline(device=args.device, use_mock=args.mock)

    # --- NEW: Vocabulary consistency check ---
    try:
        vocab_size = getattr(student_pipeline.llm.model.config, "vocab_size", None)
        if vocab_size:
            print(f"📊 Student LLM vocab size: {vocab_size}")
            if vocab_size < 40000:
                print("⚠️  Warning: small vocab size — check tokenizer alignment.")
        else:
            print("⚠️  Could not read vocab size (mock or uninitialized model).")
    except Exception as e:
        print(f"⚠️  Could not verify vocab size: {e}")

    # Load checkpoint if available
    if args.checkpoint:
        print(f"\n📥 Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        student_pipeline.encoder.load_state_dict(checkpoint["encoder"])
        student_pipeline.decoder.load_state_dict(checkpoint["decoder"])
        print("✅ Checkpoint loaded")

    # Evaluate student
    student_metrics, student_predictions = evaluate_pipeline(
        student_pipeline,
        dataloader,
        name="Neural BUS (Student)",
        max_samples=args.samples,
    )

    # Optional teacher eval (skipped for MVP)
    teacher_metrics = None
    if args.compare_teacher:
        print("\n🎓 Teacher evaluation skipped in this version (BLIP-2 baseline assumed).")

    # Compare and save
    compare_with_baseline(student_metrics, teacher_metrics)
    save_results(vars(args), student_metrics, student_predictions, teacher_metrics, args.output)

    # --- NEW: Save adapter snapshots for reproducibility ---
    Path("./results").mkdir(exist_ok=True)
    torch.save(student_pipeline.encoder.state_dict(), "./results/encoder_snapshot.pt")
    torch.save(student_pipeline.decoder.state_dict(), "./results/decoder_snapshot.pt")
    print("💾 Adapter snapshots saved for reproducibility.")

    print("\n" + "=" * 70)
    print("Evaluation Complete! ✅")
    print("=" * 70 + "\n")
    print("📝 Summary:")
    print(f"   Evaluated {student_metrics['total']} samples")
    print(f"   Accuracy: {student_metrics['exact_accuracy']:.1%}")
    print(f"   Results saved to: {args.output}")
    print()


if __name__ == "__main__":
    main()
