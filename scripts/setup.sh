#!/usr/bin/env bash
# Neural BUS Quick Setup
set -euo pipefail

WITH_DATA="${1:-}"        # allowed: "", "val", "train", "both"
DATA_ROOT="${2:-data}"    # default dataset root

echo "=================================="
echo "Neural BUS - Quick Setup"
echo "=================================="

# 1) Create directories
echo ""
echo "Creating directories..."
mkdir -p bus models adapters data results checkpoints assets/demo_images scripts

# 2) Ensure __init__.py files (makes packages importable)
echo "Creating __init__.py files..."
touch bus/__init__.py models/__init__.py adapters/__init__.py data/__init__.py

# 3) Install dependencies (prefer requirements.txt if present)
echo ""
if [[ -f requirements.txt ]]; then
  echo "Installing dependencies from requirements.txt..."
  pip install -r requirements.txt
else
  echo "Installing core dependencies..."
  pip install torch torchvision transformers accelerate Pillow numpy tqdm urllib3<2
fi

# 4) Create a tiny test image
echo ""
echo "Creating test image..."
python3 - <<'PY'
from PIL import Image
Image.new('RGB', (224, 224), 'red').save('test.jpg')
print("Wrote test.jpg")
PY

# 5) Optionally download data
if [[ -n "${WITH_DATA}" ]]; then
  echo ""
  echo "Downloading VQA v2 data: '${WITH_DATA}' into '${DATA_ROOT}' ..."
  bash scripts/get_vqa_v2.sh "${DATA_ROOT}" "${WITH_DATA}"
fi

echo ""
echo "=================================="
echo "Setup Complete! ✅"
echo "=================================="
echo ""
echo "Next steps:"
echo "1) Quick mock demo:    python demo.py --image test.jpg --question 'What color?' --mock"
echo "2) Train (mock):       python train.py --epochs 1 --batch-size 2 --subset 10 --device cpu --mock"
echo "3) Eval (mock):        python evaluate_mvp.py --samples 5 --device cpu --mock"
echo ""
echo "To download real data later:"
echo "   bash scripts/get_vqa_v2.sh data val         # val only"
echo "   bash scripts/get_vqa_v2.sh data train       # train only"
echo "   bash scripts/get_vqa_v2.sh data both        # both splits"
echo ""
