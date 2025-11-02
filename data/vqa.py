"""
VQA v2 Dataset Loader (Tensor output for batching; no auto-download)

- Uses COCO + VQA v2 JSONs if present
- Falls back to a small mock set otherwise
- Returns *tensors* shaped [3, 224, 224] so DataLoader's default collate works
- No normalization here (let VisionEncoder handle it); values in [0,1]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

# Tolerate slightly truncated JPEGs instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------- safe image open ----------
def _safe_open_rgb(path: Path):
    """
    Open and fully decode an image safely.
    Returns a PIL.Image (RGB) on success, or (None, Exception) on failure.
    """
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)  # respect EXIF orientation
            im.load()                         # force full decode inside try
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError, AssertionError) as e:
        return None, e


# ---------- dataset ----------
class VQADataset(Dataset):
    def __init__(
        self,
        data_dir: str = "./data",
        split: str = "train",
        subset_size: Optional[int] = None,
        images_root: Optional[str] = "./data/coco",
        strict: bool = False,
    ):
        """
        Args:
            data_dir: directory with VQA JSONs
            split: 'train' or 'val'
            subset_size: take first N samples (for quick runs)
            images_root: root holding COCO images (…/train2014 or …/val2014)
            strict: if True, raise when files are missing (no mock fallback)
        """
        assert split in {"train", "val"}, "split must be 'train' or 'val'"
        self.data_dir = Path(data_dir)
        self.split = split
        self.images_root = Path(images_root) if images_root else None

        # Fixed-size tensor so default_collate can stack into [B,3,224,224]
        self.to_tensor = T.Compose([
            T.Resize((224, 224)),  # keep shapes uniform
            T.ToTensor(),          # [0,1] float32, no mean/std normalization
        ])

        # Expected files
        self.questions_file = self.data_dir / f"v2_OpenEnded_mscoco_{split}2014_questions.json"
        self.annotations_file = self.data_dir / f"v2_mscoco_{split}2014_annotations.json"
        self.images_dir = (self.images_root / f"{split}2014") if self.images_root else None

        have_json = self.questions_file.exists() and self.annotations_file.exists()
        have_images = self.images_dir is not None and self.images_dir.exists()

        self.use_mock = not (have_json and have_images)
        if self.use_mock:
            missing = []
            if not have_json:
                missing += [self.questions_file.name, self.annotations_file.name]
            if not have_images:
                missing += [f"{self.split}2014 images under {self.images_root}"]
            msg = (
                "⚠️  Real VQA/COCO files not found.\n"
                f"    Missing: {', '.join(missing)}\n"
                "    Run: bash scripts/get_vqa_v2.sh data [val|train|both]\n"
                "    Falling back to a small mock dataset."
            )
            if strict:
                raise FileNotFoundError(msg)
            else:
                print(msg)

        # Load pairs
        self.samples = self._load_pairs_mock() if self.use_mock else self._load_pairs_real()

        if subset_size:
            self.samples = self.samples[:subset_size]

        print(f"✅ Loaded {len(self.samples)} VQA samples ({split})")

    # ---------------- loaders ----------------
    def _load_pairs_real(self):
        questions = json.loads(self.questions_file.read_text())["questions"]
        annotations = json.loads(self.annotations_file.read_text())["annotations"]
        ann_map = {a["question_id"]: a for a in annotations}

        samples = []
        for q in questions:
            ann = ann_map.get(q["question_id"])
            if ann is None:
                continue
            answers = [a["answer"] for a in ann["answers"]]
            most_common = max(set(answers), key=answers.count) if answers else ""
            samples.append(
                {
                    "question_id": q["question_id"],
                    "image_id": q["image_id"],
                    "question": q["question"],
                    "answer": most_common,
                    "all_answers": answers,
                }
            )
        return samples

    def _load_pairs_mock(self):
        # tiny synthetic set; ensures varied answers/questions
        return [
            {
                "question_id": i,
                "image_id": i,
                "question": f"What is in image {i}?",
                "answer": f"object_{i % 10}",
                "all_answers": [f"object_{i % 10}"] * 10,
            }
            for i in range(100)
        ]

    # ---------------- utils ----------------
    def _coco_path(self, image_id: int) -> Optional[Path]:
        if self.images_dir is None:
            return None
        # COCO 2014 naming uses 12-digit zero padding
        fname = f"COCO_{self.split}2014_{image_id:012d}.jpg"
        return self.images_dir / fname

    # ---------------- dataset API ----------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        if self.use_mock:
            # deterministic colored block for variety
            pil = Image.new(
                "RGB",
                (256, 256),
                (30 + (s["image_id"] * 7) % 220, 80, 120),
            )
        else:
            p = self._coco_path(s["image_id"])
            pil = None
            if p is not None and p.exists():
                out = _safe_open_rgb(p)
                if isinstance(out, tuple):  # (None, error)
                    pil, err = None, out[1]
                    # Optional: log once every 500 failures to avoid spam
                    if getattr(self, "_bad_count", 0) % 500 == 0:
                        print(f"⚠️  Bad image at {p}: {err}")
                    self._bad_count = getattr(self, "_bad_count", 0) + 1
                else:
                    pil = out

            # rare JSON/image mismatch -> neutral fallback
            if pil is None:
                pil = Image.new("RGB", (256, 256), "gray")

        img_tensor = self.to_tensor(pil)  # [3,224,224], float32 in [0,1]

        return {
            "image": img_tensor,
            "question": s["question"],
            "answer": s["answer"],
            "question_id": s["question_id"],
            "image_id": s["image_id"],
        }


if __name__ == "__main__":
    ds = VQADataset(split="val", subset_size=3, strict=False)
    x = ds[0]
    print("OK:", tuple(x["image"].shape), x["question"], x["answer"], type(x["image"]))
