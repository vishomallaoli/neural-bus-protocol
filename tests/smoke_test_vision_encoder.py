# tests/smoke_test_vision_encoder.py
import os
import tempfile
import torch
from PIL import Image

from models.vision_encoder import VisionEncoder

def check_ok(name, cond):
    if not cond:
        raise AssertionError(f"[FAIL] {name}")
    print(f"[PASS] {name}")

def run_single_case(enc, x, expect_batch=False):
    out = enc(x)
    if expect_batch:
        check_ok("batched → [B,2048]", out.ndim == 2 and out.shape[1] == 2048)
    else:
        check_ok("single → [2048]", out.ndim == 1 and out.shape[0] == 2048)

def main():
    print("=== VisionEncoder Smoke Test ===")
    print(f"torch: {torch.__version__}")

    # --- 1) Frozen encoder (no grad), no weights download ---
    enc_frozen = VisionEncoder(pretrained=False, freeze=True, image_size=224)
    check_ok("frozen → params require_grad=False", all(not p.requires_grad for p in enc_frozen.backbone.parameters()))

    # Create a temp RGB image on disk for the 'str path' route
    tmpdir = tempfile.mkdtemp()
    img_path = os.path.join(tmpdir, "test.jpg")
    Image.new("RGB", (320, 240), "red").save(img_path)

    # a) str path
    run_single_case(enc_frozen, img_path)

    # b) PIL
    pil = Image.new("RGB", (300, 180), "blue")
    run_single_case(enc_frozen, pil)

    # c) CHW float32 [3,224,224] in [0,1]
    chw = torch.rand(3, 224, 224)
    run_single_case(enc_frozen, chw)

    # d) HWC uint8 [224,224,3] → should auto permute + scale
    hwc = torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8)
    run_single_case(enc_frozen, hwc)

    # e) BHWC uint8 [B,224,224,3]
    bhwc = torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8)
    out = enc_frozen(bhwc)
    check_ok("BHWC batch outputs", out.shape == (2, 2048))

    # f) BxCHW float32 [B,3,H,W] (auto-resize)
    bchw = torch.rand(3, 3, 200, 200)
    out = enc_frozen(bchw)
    check_ok("BCHW batch outputs", out.shape == (3, 2048))

    # g) grayscale CHW [1,H,W] → should auto-repeat to 3 channels
    gray_chw = torch.randint(0, 256, (1, 180, 240), dtype=torch.uint8)
    run_single_case(enc_frozen, gray_chw)

    # h) grayscale BHWC [B,H,W,1]
    gray_bhwc = torch.randint(0, 256, (2, 180, 240, 1), dtype=torch.uint8)
    out = enc_frozen(gray_bhwc)
    check_ok("grayscale BHWC batch outputs", out.shape == (2, 2048))

    # --- 2) Unfrozen encoder (grad check) ---
    enc_trainable = VisionEncoder(pretrained=False, freeze=False, image_size=224)
    enc_trainable.train()  # ensure training mode to allow grads
    inp = torch.rand(1, 3, 224, 224)
    feats = enc_trainable(inp)            # [2048]
    loss = feats.sum()
    loss.backward()

    any_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc_trainable.backbone.parameters())
    check_ok("unfrozen → backprop produces grads", any_grad)

    print("\nAll smoke tests passed ✅")

if __name__ == "__main__":
    main()
