"""
Image Captioner - BLIP-2
Generates text descriptions from images
"""

from __future__ import annotations

import contextlib
from typing import Union

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor


class Captioner:
    """BLIP-2 image captioner with safe preprocessing."""

    def __init__(self, model_name: str = "Salesforce/blip2-flan-t5-large", device: str = "cpu"):
        self.device = device
        self.model_name = model_name

        print(f"Loading {model_name}...")
        self.processor = Blip2Processor.from_pretrained(model_name)

        # fp16 on cuda/mps, fp32 on cpu
        dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        print("✅ Captioner loaded")

        # Some tokenizer configs lack an explicit pad token
        self.pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = self.processor.tokenizer.eos_token_id

    # ---------------- utilities ----------------

    @contextlib.contextmanager
    def _temp_image_flag(self, attr: str, value):
        """
        Temporarily set an attribute on the processor.image_processor (e.g., do_rescale).
        Ensures it is restored on exit even if generation throws.
        """
        ip = self.processor.image_processor
        old = getattr(ip, attr)
        try:
            setattr(ip, attr, value)
            yield
        finally:
            setattr(ip, attr, old)

    def _is_float_tensor_0_1(self, t: torch.Tensor) -> bool:
        return (
            isinstance(t, torch.Tensor)
            and t.dtype.is_floating_point
            and t.ndim == 3
            and t.min().item() >= 0.0
            and t.max().item() <= 1.0
        )

    def _prepare_inputs(
        self, image: Union[Image.Image, str, torch.Tensor]
    ) -> dict:
        """
        Return a dict of tensors from the processor, already moved to the target device/dtype.
        Rules:
          - str → load PIL RGB
          - PIL.Image → use as-is
          - torch.Tensor (float, 0..1, CHW) → pass directly to processor with do_rescale=False
          - torch.Tensor outside 0..1 → raise (likely mean/std normalized; pass PIL instead)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if isinstance(image, Image.Image):
            inputs = self.processor(images=image, return_tensors="pt")
        elif isinstance(image, torch.Tensor):
            if not (image.ndim == 3 and image.shape[0] in (1, 3)):
                raise ValueError("Tensor image must be CHW with C in {1,3}.")
            if self._is_float_tensor_0_1(image):
                # Avoid the "already rescaled" warning by disabling rescale temporarily.
                with self._temp_image_flag("do_rescale", False):
                    inputs = self.processor(images=image, return_tensors="pt")
            else:
                raise ValueError(
                    "Tensor image appears not to be 0..1 floats (likely mean/std normalized). "
                    "Pass a PIL image (RGB) or a float tensor in [0,1]."
                )
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Move to device and cast float tensors to model dtype
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
    def caption(
        self,
        image: Union[Image.Image, str, torch.Tensor],
        max_new_tokens: int = 15,          # ↓ shorter, faster
        do_sample: bool = False,           # ↓ deterministic by default
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int | None = 1,         # ↓ no beam search overhead by default
    ) -> str:
        """
        Generate a caption for an image.

        Tip: Set do_sample=True or num_beams>1 if you really want more diverse/optimized captions.
        """
        inputs = self._prepare_inputs(image)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,                 # helps speed on long prompts
        )

        # If user asks for beams, force deterministic decoding
        if num_beams is not None and num_beams > 1:
            gen_kwargs.update(dict(num_beams=num_beams, do_sample=False))
        else:
            # Greedy by default; enable sampling only if explicitly requested
            if do_sample:
                gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
            else:
                gen_kwargs.update(dict(do_sample=False, num_beams=1))

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

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
        # Deterministic mock caption
        return "A scene with various objects and details visible in the image"


# Quick test
if __name__ == "__main__":
    img = Image.new("RGB", (224, 224), "blue")
    cap = MockCaptioner().caption(img)
    print(f"✅ Generated caption (mock): '{cap}'")
