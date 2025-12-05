#!/usr/bin/env python3
"""
Neural BUS Web Demo
===================

Flask backend for the Neural BUS VQA demo.

- Serves the frontend at `/`
- Exposes `/api/process` for image + question
- Loads BLIP VQA model once at startup (works with gunicorn on Render)

Local dev:
    python app.py      # then open http://localhost:5001

Render:
    gunicorn app:app   # no need to call app.run()
"""

import base64
import io
import os
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image
import torch

# ---------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------

# Project root (directory containing this file)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Configure Flask:
# - templates/ for HTML (demo.html)
# - assets/ as static folder, served under /assets/ (matches demo.html)
app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "assets"),
    static_url_path="/assets",
)

CORS(app)  # Enable CORS (useful for future separate frontends)

# ---------------------------------------------------------------------
# Model globals
# ---------------------------------------------------------------------

processor = None
model = None
device = "cpu"  # change to "cuda" if you later add GPU support


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_model():
    """Load BLIP VQA model (called once at import/startup)."""
    global processor, model

    print("=" * 70)
    print("Loading BLIP VQA Model for Neural BUS Web API...")
    print("=" * 70)

    try:
        from transformers import BlipProcessor, BlipForQuestionAnswering
    except ImportError:
        print("❌ Error: transformers not installed")
        print("Run: pip install transformers")
        sys.exit(1)

    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    model.to(device)
    model.eval()

    print("✅ BLIP model loaded successfully")
    print("=" * 70)
    print()


def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Decode base64 image string to PIL Image.

    Supports data URLs like: "data:image/png;base64,iVBORw0KG..."
    """
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    # Decode base64
    image_data = base64.b64decode(base64_string)

    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_data))

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_vqa(image_pil: Image.Image, question: str) -> dict:
    """
    Process VQA request using BLIP.

    Args:
        image_pil: PIL Image
        question: str

    Returns:
        dict with answer, metrics, and a mock BUS packet
    """
    if processor is None or model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    start_time = time.time()

    # Prepare inputs
    inputs = processor(image_pil, question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate answer
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=20)

    # Decode answer
    answer = processor.decode(outputs[0], skip_special_tokens=True).strip()

    inference_time = time.time() - start_time

    # Mock BUS packet (demo)
    bus_packet = {
        "header": {
            "source": "vision.blip.v1",
            "intent": "vqa",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "payload": {
            "vector": {
                "dim": 512,
                "dtype": "float32",
                "data": "[vector data - 512 dimensions]",
            },
            "text": f"Q: {question} | A: {answer}",
        },
        "provenance": {
            "vision_model": "BLIP-vqa-base",
            "device": device,
        },
    }

    return {
        "answer": answer,
        "confidence": 0.85,          # Mock confidence
        "inference_time": inference_time,
        "cycle_similarity": 0.92,    # Mock cycle consistency
        "bus_packet": bus_packet,
    }


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main demo HTML page."""
    return render_template("demo.html")


@app.route("/api/process", methods=["POST"])
def api_process():
    """
    API endpoint for processing images.

    Request JSON:
        {
            "image": "data:image/png;base64,...",
            "question": "What is in the image?"
        }

    Response JSON:
        {
            "success": true,
            "result": { ... }
        }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"success": False, "error": "No JSON body provided"}), 400

        image_data = data.get("image")
        question = data.get("question", "What is in the image?")

        if not image_data:
            return jsonify({"success": False, "error": "No image provided"}), 400

        # Decode image
        try:
            image_pil = decode_base64_image(image_data)
        except Exception as e:
            return jsonify(
                {"success": False, "error": f"Failed to decode image: {e}"}
            ), 400

        # Process through VQA / Neural BUS backend
        result = process_vqa(image_pil, question)

        return jsonify({"success": True, "result": result})

    except Exception as e:
        print(f"❌ Error processing request: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model is not None,
        }
    )


# ---------------------------------------------------------------------
# Initialization (for gunicorn and local dev)
# ---------------------------------------------------------------------

# Load model at import time so:
# - gunicorn app:app works on Render
# - python app.py works locally
print("\n" + "=" * 70)
print("           NEURAL BUS WEB BACKEND STARTUP")
print("=" * 70 + "\n")

load_model()

if __name__ == "__main__":
    # Local development: `python app.py`
    # Use PORT from environment if available (for cloud platforms)
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
