#!/bin/bash
# Neural BUS Quick Setup Script

echo "=================================="
echo "Neural BUS - Quick Setup"
echo "=================================="

# Create directories
echo ""
echo "Creating directories..."
mkdir -p bus models adapters data results checkpoints assets/demo_images

# Create __init__.py files
echo "Creating __init__.py files..."
touch bus/__init__.py
touch models/__init__.py
touch adapters/__init__.py
touch data/__init__.py

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install torch torchvision transformers accelerate Pillow numpy tqdm

# Create test image
echo ""
echo "Creating test image..."
python3 -c "from PIL import Image; Image.new('RGB', (224, 224), 'red').save('test.jpg')"

echo ""
echo "=================================="
echo "Setup Complete! ✅"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Run demo: python demo.py --image test.jpg --question 'What color?' --mock"
echo "2. Read README.md for more info"
echo ""