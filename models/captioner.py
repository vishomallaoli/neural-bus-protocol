"""
Image Captioner - BLIP-2
Generates text descriptions from images
"""

import torch
from typing import Union
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from torchvision import transforms


class Captioner:
    """BLIP-2 image captioner"""

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: str = "cpu"):
        self.device = device
        self.model_name = model_name

        print(f"Loading {model_name}...")
        # Using the processor; keep defaults but we’ll optionally disable rescale at call time
        self.processor = Blip2Processor.from_pretrained(model_name)

        # Use fp16 on cuda/mps, fp32 on cpu
        dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✅ Captioner loaded")

        # Fallback for pad token (some tokenizer configs may not have it)
        self.pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = self.processor.tokenizer.eos_token_id

        # Helper to convert tensors → PIL when users pass tensors
        self._to_pil = transforms.ToPILImage()

    # ---------------- internal helpers ----------------
    def _normalize_image_input(self, image: Union[Image.Image, str, torch.Tensor]) -> Image.Image:
        """
        Accept str path, PIL Image, or torch Tensor and return a PIL Image (RGB).
        Ensures we don’t double-rescale later.
        """
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        elif isinstance(image, torch.Tensor):
            # Expect [C,H,W] in [0,1] or [-mean/std normalized]. Clamp to [0,1] before ToPIL
            if image.ndim != 3:
                raise ValueError("Tensor image must be [C,H,W].")
            img = self._to_pil(image.detach().cpu().clamp(0, 1))
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        return img

    def _move_inputs(self, inputs: dict) -> dict:
        """
        Move processor outputs to target device, casting floating tensors to model dtype.
        """
        moved = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if torch.is_floating_point(v):
                    moved[k] = v.to(self.device, dtype=self.model.dtype)
                else:
                    moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved

    # ---------------- public API ----------------
    def caption(self, image: Union[Image.Image, str, torch.Tensor], max_new_tokens: int = 30) -> str:
        """
        Generate a caption for an image.
        Accepts PIL Image, file path, or torch Tensor.
        """
        img = self._normalize_image_input(image)

        # If the user passed a Tensor originally, we already converted to PIL,
        # so keep processor defaults (no double-rescale warning).
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = self._move_inputs(inputs)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,          # sampling for diversity; set False for deterministic
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Use processor.tokenizer to decode; batch_decode returns a list
        text = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return text


class MockCaptioner:
    """Mock captioner for testing without downloads"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model_name = "mock"
        print("✅ Mock Captioner loaded")

    def caption(self, image: Union[Image.Image, str, torch.Tensor], max_new_tokens: int = 50) -> str:
        # Return a stable, deterministic mock caption
        return "A scene with various objects and details visible in the image"


# Quick test
if __name__ == "__main__":
    print("Testing Captioner...")
    test_img = Image.new("RGB", (224, 224), "blue")
    cap = MockCaptioner().caption(test_img)
    print(f"✅ Generated caption (mock): '{cap}'")
