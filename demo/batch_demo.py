#!/usr/bin/env python3
"""
Neural BUS Batch Evaluation - VQA v2
=====================================

Batch evaluation using BLIP backend for Neural BUS protocol validation.

Usage:
    python batch_demo.py --num-samples 100
    python batch_demo.py --num-samples 50 --output results/eval.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
from PIL import Image
import torchvision.transforms.functional as TF

# Smart path resolution
current_file = Path(__file__).absolute()
script_dir = current_file.parent
if script_dir.name == "demo":
    project_root = script_dir.parent
else:
    project_root = script_dir
sys.path.insert(0, str(project_root))

try:
    from transformers import BlipProcessor, BlipForQuestionAnswering
    try:
        from data.vqa import VQADataset
    except ImportError:
        from vqa import VQADataset
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease install required packages:")
    print("  pip install transformers pillow torchvision")
    sys.exit(1)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    if not answer:
        return ""
    return answer.lower().strip().replace(".", "").replace(",", "").replace("!", "").replace("?", "")


def check_match(prediction: str, ground_truth: str) -> bool:
    """Check if prediction matches ground truth"""
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    
    if not pred_norm or not gt_norm:
        return False
    
    # Exact match
    if pred_norm == gt_norm:
        return True
    
    # Substring match
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return True
    
    # Common variations
    yes_variants = ["yes", "yeah", "yep", "yup"]
    no_variants = ["no", "nope", "not"]
    
    if gt_norm in yes_variants and pred_norm in yes_variants:
        return True
    if gt_norm in no_variants and pred_norm in no_variants:
        return True
    
    return False


class BLIPEvaluator:
    """BLIP-based VQA evaluator for Neural BUS validation"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        print("Loading BLIP student model...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        self.model.to(device)
        self.model.eval()
        print("✅ BLIP model loaded\n")
    
    def answer(self, image_tensor: torch.Tensor, question: str) -> str:
        """Get answer from BLIP model"""
        # Convert tensor to PIL
        pil_image = TF.to_pil_image(image_tensor)
        
        # Process
        inputs = self.processor(pil_image, question, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        
        # Decode
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()


def print_progress_bar(current: int, total: int, width: int = 40):
    """Print progress bar"""
    progress = current / total
    filled = int(width * progress)
    bar = "=" * filled + " " * (width - filled)
    return f"[{bar}] {current}/{total}"


def main():
    parser = argparse.ArgumentParser(description="Neural BUS Batch Evaluation")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default="output/batch_results.json", help="Output JSON file")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()
    
    # Header
    print("\n" + "=" * 70)
    print("Neural BUS Batch Evaluation - VQA v2 Validation Set")
    print("=" * 70)
    
    # Load dataset
    print(f"Loading VQA v2 dataset ({args.num_samples} samples)...")
    dataset = VQADataset(split="val", subset_size=args.num_samples, strict=False)
    
    # Load model
    evaluator = BLIPEvaluator(device=args.device)
    
    # Process samples
    print(f"Processing {args.num_samples} samples...")
    
    results = []
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        image = sample["image"]
        question = sample["question"]
        ground_truth = sample["answer"]
        image_id = sample["image_id"]
        
        try:
            prediction = evaluator.answer(image, question)
            is_correct = check_match(prediction, ground_truth)
            
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "image_id": int(image_id),
                "question": question,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "correct": bool(is_correct)
            })
            
        except Exception as e:
            print(f"\n⚠️  Error on sample {idx}: {e}")
            results.append({
                "image_id": int(image_id),
                "question": question,
                "prediction": None,
                "ground_truth": ground_truth,
                "correct": False
            })
            total += 1
        
        # Update progress every sample
        if (idx + 1) % max(1, len(dataset) // 20) == 0 or (idx + 1) == len(dataset):
            progress_str = print_progress_bar(idx + 1, len(dataset))
            print(f"\rProgress: {progress_str}", end="", flush=True)
    
    print()  # New line after progress
    elapsed = time.time() - start_time
    
    # Calculate accuracy
    accuracy = (correct / total * 100) if total > 0 else 0
    baseline_accuracy = 78.7  # From your requirements
    gap = baseline_accuracy - accuracy
    
    # Results Summary
    print("\nResults Summary:")
    print("-" * 70)
    print(f"Total Samples:        {total}")
    print(f"Correct Predictions:  {correct}")
    print(f"Accuracy:            {accuracy:.1f}%")
    print()
    print(f"Ground Truth Baseline (BLIP student):     {baseline_accuracy}%")
    print(f"Caption-Based Transmission:       {accuracy:.1f}%")
    print(f"Performance Gap:                  {gap:.1f} points")
    
    # Sample Results
    print("\nSample Results:")
    print("-" * 70)
    
    num_to_show = min(10, len(results))
    for i in range(num_to_show):
        r = results[i]
        status = "✓" if r["correct"] else "✗"
        image_filename = f"COCO_val2014_{r['image_id']:012d}.jpg"
        
        print(f"[{i+1}] Image: {image_filename}")
        print(f"    Q: {r['question']}")
        pred = r['prediction'] if r['prediction'] else "error"
        print(f"    Pred: {pred}  |  GT: {r['ground_truth']}  {status}")
        print()
    
    if len(results) > num_to_show:
        print(f"[... {len(results) - num_to_show} more results truncated ...]")
        print()
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "metadata": {
            "total_samples": total,
            "correct": correct,
            "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
            "performance_gap": gap,
            "processing_time_seconds": elapsed
        },
        "results": results
    }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Batch evaluation complete!")
    print(f"   Results saved to: {args.output}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()