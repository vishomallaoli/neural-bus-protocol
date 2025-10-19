#!/usr/bin/env python3
"""
Neural BUS Demo
Quick test of the pipeline with a single image
"""

import argparse
from pathlib import Path
from PIL import Image

from pipeline import NeuralBUSPipeline


def main():
    parser = argparse.ArgumentParser(description="Neural BUS Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--question", type=str, required=True, help="Question about image")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--mock", action="store_true", help="Use mock models (fast)")
    parser.add_argument("--save-packet", type=str, default=None, help="Save BUS packet to JSON")
    
    args = parser.parse_args()
    
    # Validate image
    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        print("\nCreate a test image with:")
        print("  python -c \"from PIL import Image; Image.new('RGB', (224,224), 'red').save('test.jpg')\"")
        return
    
    print("\n" + "="*70)
    print(" "*25 + "NEURAL BUS DEMO")
    print("="*70)
    print(f"\nImage: {args.image}")
    print(f"Question: {args.question}")
    print(f"Device: {args.device}")
    print(f"Mock Mode: {args.mock}")
    
    # Initialize pipeline
    pipeline = NeuralBUSPipeline(device=args.device, use_mock=args.mock)
    
    # Process
    result = pipeline(
        image=args.image,
        question=args.question,
        save_packet=args.save_packet
    )
    
    # Display results
    print("\n" + "="*70)
    print(" "*28 + "RESULTS")
    print("="*70)
    print(f"\n📸 Caption: {result['caption']}")
    print(f"\n❓ Question: {args.question}")
    print(f"\n💡 Answer: {result['answer']}")
    print(f"\n📦 Packet Size: {result['packet'].size_bytes()/1024:.2f} KB")
    
    if args.save_packet:
        print(f"\n✅ Packet saved to: {args.save_packet}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()