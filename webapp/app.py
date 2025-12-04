#!/usr/bin/env python3
"""
Flask Backend for Neural BUS Web Demo
======================================

Provides API endpoint for processing images through Neural BUS pipeline.

Usage:
    python app.py
    
Then open: http://localhost:5001
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import io
import time
from pathlib import Path
import sys

import torch
from PIL import Image
import torchvision.transforms.functional as TF

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from transformers import BlipProcessor, BlipForQuestionAnswering
except ImportError:
    print("❌ Error: transformers not installed")
    print("Run: pip install transformers pillow flask flask-cors")
    sys.exit(1)

app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for development

# Global model variables (loaded once)
processor = None
model = None
device = "cpu"


def load_model():
    """Load BLIP model (called once on startup)"""
    global processor, model
    
    print("=" * 70)
    print("Loading BLIP VQA Model...")
    print("=" * 70)
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    model.to(device)
    model.eval()
    
    print("✅ BLIP model loaded successfully")
    print("=" * 70)
    print()


def decode_base64_image(base64_string):
    """
    Decode base64 image string to PIL Image
    
    Handles data URLs like: "data:image/png;base64,iVBORw0KG..."
    """
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    image_data = base64.b64decode(base64_string)
    
    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_data))
    
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


def process_vqa(image_pil, question):
    """
    Process VQA request using BLIP
    
    Args:
        image_pil: PIL Image
        question: str
    
    Returns:
        dict with answer and metadata
    """
    start_time = time.time()
    
    # Process with BLIP
    inputs = processor(image_pil, question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate answer
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=20)
    
    # Decode answer
    answer = processor.decode(outputs[0], skip_special_tokens=True).strip()
    
    inference_time = time.time() - start_time
    
    # Create mock BUS packet (for demo purposes)
    bus_packet = {
        "header": {
            "source": "vision.blip.v1",
            "intent": "vqa",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        },
        "payload": {
            "vector": {
                "dim": 512,
                "dtype": "float32",
                "data": "[vector data - 512 dimensions]"
            },
            "text": f"Q: {question} | A: {answer}"
        },
        "provenance": {
            "vision_model": "BLIP-vqa-base",
            "device": device
        }
    }
    
    return {
        "answer": answer,
        "confidence": 0.85,  # Mock confidence (BLIP doesn't return this directly)
        "inference_time": inference_time,
        "cycle_similarity": 0.92,  # Mock cycle consistency
        "bus_packet": bus_packet
    }


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'demo.html')


@app.route('/api/process', methods=['POST'])
def api_process():
    """
    API endpoint for processing images
    
    Request JSON:
        {
            "image": "data:image/png;base64,...",
            "question": "What is in the image?"
        }
    
    Response JSON:
        {
            "success": true,
            "result": {
                "answer": "a cat",
                "confidence": 0.85,
                "inference_time": 1.23,
                "cycle_similarity": 0.92,
                "bus_packet": {...}
            }
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400
        
        image_data = data.get('image')
        question = data.get('question', 'What is in this image?')
        
        if not image_data:
            return jsonify({
                "success": False,
                "error": "No image provided"
            }), 400
        
        # Decode image
        try:
            image_pil = decode_base64_image(image_data)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Failed to decode image: {str(e)}"
            }), 400
        
        # Process through Neural BUS (BLIP backend)
        result = process_vqa(image_pil, question)
        
        return jsonify({
            "success": True,
            "result": result
        })
        
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print(" " * 15 + "NEURAL BUS WEB DEMO")
    print("=" * 70)
    print()
    
    # Load model
    load_model()
    
    # Start server
    print("Starting Flask server...")
    print()
    print("=" * 70)
    print("🌐 Web Demo Running!")
    print("=" * 70)
    print()
    print("Open in your browser:")
    print("  👉 http://localhost:5001")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    app.run(host='0.0.0.0', port=5001, debug=False)


if __name__ == '__main__':
    main()