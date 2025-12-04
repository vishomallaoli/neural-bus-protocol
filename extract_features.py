#!/usr/bin/env python3
"""
Feature Extraction Script
Extracts vision features, captions, and text embeddings from VQA images
Runs on M1 MPS - should take ~15 minutes for 200 images
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# Import your existing modules
from models.vision_encoder import VisionEncoder
from models.captioner import Captioner
from data.vqa import VQADataset

print("=" * 70)
print("NEURAL BUS - FEATURE EXTRACTION")
print("=" * 70)

# Configuration
DEVICE = "mps"  # M1 GPU
NUM_SAMPLES = 200
SPLIT = "val"  # Using validation set
OUTPUT_DIR = Path("./results/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Device: {DEVICE}")
print(f"  Samples: {NUM_SAMPLES}")
print(f"  Split: {SPLIT}")
print(f"  Output: {OUTPUT_DIR}")
print()

# Initialize models
print("Loading models...")
print("-" * 70)

vision_encoder = VisionEncoder(pretrained=True, freeze=True).to(DEVICE)
print(f"✅ Vision Encoder loaded (ResNet-50, {vision_encoder.feature_dim}D)")

captioner = Captioner(device=DEVICE)
print(f"✅ Captioner loaded (BLIP-2)")

# Load dataset
print("\nLoading dataset...")
print("-" * 70)
dataset = VQADataset(
    data_dir="./data",
    split=SPLIT,
    subset_size=NUM_SAMPLES,
    images_root="./data/coco",
    strict=False
)
print(f"✅ Loaded {len(dataset)} samples")

# Storage
vision_features = []
captions = []
questions = []
answers = []
image_ids = []
question_ids = []

print("\nExtracting features...")
print("-" * 70)

# Process each image
for idx in tqdm(range(len(dataset)), desc="Processing images"):
    sample = dataset[idx]
    
    # Get image tensor [3, 224, 224]
    image_tensor = sample["image"]
    
    # Extract vision features
    with torch.no_grad():
        vis_feat = vision_encoder.forward(image_tensor.to(DEVICE))  # [2048]
        vis_feat_cpu = vis_feat.cpu().numpy()
    
    # Generate caption (pass tensor to captioner)
    with torch.no_grad():
        caption = captioner.caption(image_tensor, max_new_tokens=20, do_sample=False)
    
    # Store results
    vision_features.append(vis_feat_cpu)
    captions.append(caption)
    questions.append(sample["question"])
    answers.append(sample["answer"])
    image_ids.append(sample["image_id"])
    question_ids.append(sample["question_id"])

# Convert to arrays
vision_features = np.array(vision_features)  # [N, 2048]

print(f"\n✅ Extraction complete!")
print(f"   Vision features: {vision_features.shape}")
print(f"   Captions: {len(captions)}")

# Save to disk
print("\nSaving features...")
print("-" * 70)

# Save vision features as numpy
np.save(OUTPUT_DIR / "vision_features.npy", vision_features)
print(f"✅ Saved: vision_features.npy")

# Save captions and metadata as JSON
metadata = {
    "captions": captions,
    "questions": questions,
    "answers": answers,
    "image_ids": image_ids,
    "question_ids": question_ids,
    "config": {
        "num_samples": NUM_SAMPLES,
        "split": SPLIT,
        "vision_dim": 2048,
        "device": DEVICE
    }
}

with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Saved: metadata.json")

# Compute and save text embeddings using sentence-transformers
print("\nComputing text embeddings...")
print("-" * 70)

try:
    from sentence_transformers import SentenceTransformer
    
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight
    print("✅ Loaded sentence transformer")
    
    # Encode captions
    caption_embeddings = text_encoder.encode(
        captions, 
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Save
    np.save(OUTPUT_DIR / "caption_embeddings.npy", caption_embeddings)
    print(f"✅ Saved: caption_embeddings.npy ({caption_embeddings.shape})")
    
except ImportError:
    print("⚠️  sentence-transformers not installed")
    print("   Installing: pip install sentence-transformers")
    print("   Skipping text embeddings for now...")
    caption_embeddings = None

print("\n" + "=" * 70)
print("EXTRACTION COMPLETE! 🎉")
print("=" * 70)
print(f"\nResults saved to: {OUTPUT_DIR}")
print("\nFiles created:")
print("  - vision_features.npy    (vision embeddings)")
print("  - caption_embeddings.npy (text embeddings)")
print("  - metadata.json          (captions, questions, etc.)")
print("\nNext step: Run analysis.ipynb to generate plots")
print("=" * 70)