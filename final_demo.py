#!/usr/bin/env python3
"""
Neural BUS Batch Processing Demo
=================================

Processes multiple images through the Neural BUS pipeline.
Demonstrates the batch workflow from User Manual Section 4.2.

Usage:
    python batch_demo.py --images-dir data/coco/val2014 --num-samples 10 --mock
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import NeuralBUSPipeline


def main():
    parser = argparse.ArgumentParser(description="Batch process images with Neural BUS")
    
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/coco/val2014",
        help="Directory containing images"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of images to process"
    )
    
    parser.add_argument(
        "--question",
        type=str,
        default="What is in this image?",
        help="Question to ask about each image"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/batch",
        help="Directory to save BUS packets"
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock models"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Compute device"
    )
    
    args = parser.parse_args()
    
    # Setup
    images_path = Path(args.images_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not images_path.exists():
        print(f"❌ Error: Images directory not found: {images_path}")
        print("\nUsing mock mode with test.jpg instead")
        images = [Path("test.jpg")] * args.num_samples
    else:
        images = list(images_path.glob("*.jpg"))[:args.num_samples]
    
    print("\n" + "=" * 70)
    print(" " * 18 + "NEURAL BUS BATCH DEMO")
    print("=" * 70)
    print()
    print(f"Processing {len(images)} images...")
    print(f"Question: {args.question}")
    print(f"Device: {args.device}")
    print(f"Mode: {'Mock' if args.mock else 'Full Pipeline'}")
    print()
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = NeuralBUSPipeline(device=args.device, use_mock=args.mock)
    print("✅ Pipeline ready\n")
    
    # Process images
    print("=" * 70)
    print("Processing Images:")
    print("=" * 70)
    
    for i, img_path in enumerate(images, 1):
        try:
            save_path = output_path / f"{img_path.stem}_packet.json"
            
            result = pipeline(
                image=str(img_path),
                question=args.question,
                save_packet=str(save_path)
            )
            
            # Print result
            print(f"\n[{i}/{len(images)}] {img_path.name}")
            print(f"  Answer: {result['answer']}")
            if not args.mock:
                print(f"  Caption: {result['caption'][:60]}...")
            print(f"  Packet: {save_path.name}")
            
        except Exception as e:
            print(f"\n[{i}/{len(images)}] ❌ Failed: {img_path.name}")
            print(f"  Error: {e}")
    
    print("\n" + "=" * 70)
    print(f"✅ Batch processing complete!")
    print(f"   Processed: {len(images)} images")
    print(f"   Packets saved to: {output_path}/")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()