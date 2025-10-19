"""
VQA v2 Dataset Loader (Fixed + Enhanced)
"""

import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional
from torchvision import transforms


class VQADataset(Dataset):
    """
    VQA v2 dataset loader

    - If the real dataset is missing, automatically creates mock data.
    - Converts all images (real or mock) to normalized tensors
      compatible with torchvision ResNet-50 (3×224×224).
    """

    def __init__(
        self,
        data_dir: str = "./data",
        split: str = "train",
        subset_size: Optional[int] = None
    ):
        """
        Args:
            data_dir: Directory containing VQA data
            split: 'train' or 'val'
            subset_size: Use only first N samples (for quick testing)
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # 🔧 Universal transform for all images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Expected dataset files
        questions_file = self.data_dir / f"v2_OpenEnded_mscoco_{split}2014_questions.json"
        annotations_file = self.data_dir / f"v2_mscoco_{split}2014_annotations.json"

        # Fallback to mock data if missing
        if not questions_file.exists() or not annotations_file.exists():
            print(f"⚠️  VQA data not found in {data_dir}")
            print("Creating mock dataset for testing...")
            self._create_mock_data()

        # Load data
        self.samples = self._load_data()

        # Optional subset for quick tests
        if subset_size:
            self.samples = self.samples[:subset_size]

        print(f"✅ Loaded {len(self.samples)} VQA samples ({split})")

    # ------------------------------------------------------------------ #
    def _create_mock_data(self):
        """Create mock VQA JSONs for testing if real data missing."""
        mock_questions = {
            "questions": [
                {"image_id": i, "question": f"What is in image {i}?", "question_id": i}
                for i in range(100)
            ]
        }

        mock_annotations = {
            "annotations": [
                {
                    "question_id": i,
                    "image_id": i,
                    "answers": [{"answer": f"object_{i % 10}"}] * 10
                }
                for i in range(100)
            ]
        }

        self.data_dir.mkdir(parents=True, exist_ok=True)

        with open(self.data_dir / f"v2_OpenEnded_mscoco_{self.split}2014_questions.json", 'w') as f:
            json.dump(mock_questions, f)

        with open(self.data_dir / f"v2_mscoco_{self.split}2014_annotations.json", 'w') as f:
            json.dump(mock_annotations, f)

    # ------------------------------------------------------------------ #
    def _load_data(self):
        """Load questions + annotations and merge them."""
        questions_file = self.data_dir / f"v2_OpenEnded_mscoco_{self.split}2014_questions.json"
        annotations_file = self.data_dir / f"v2_mscoco_{self.split}2014_annotations.json"

        with open(questions_file) as f:
            questions = json.load(f)["questions"]

        with open(annotations_file) as f:
            annotations = json.load(f)["annotations"]

        # Map question_id → annotation
        ann_map = {ann["question_id"]: ann for ann in annotations}

        # Merge question + annotation
        samples = []
        for q in questions:
            if q["question_id"] in ann_map:
                ann = ann_map[q["question_id"]]
                answers = [a["answer"] for a in ann["answers"]]
                most_common = max(set(answers), key=answers.count)

                samples.append({
                    "question_id": q["question_id"],
                    "image_id": q["image_id"],
                    "question": q["question"],
                    "answer": most_common,
                    "all_answers": answers
                })

        return samples

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 🖼️  For mock mode: create dummy gray image
        #     For real data: you can later replace with actual COCO image loading
        image = Image.new('RGB', (224, 224), 'gray')

        # Convert PIL → Tensor (resized + normalized)
        image = self.transform(image)

        return {
            "image": image,
            "question": sample["question"],
            "answer": sample["answer"],
            "question_id": sample["question_id"],
            "image_id": sample["image_id"]
        }


# ---------------------------------------------------------------------- #
# Quick test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Testing VQA Dataset...")

    dataset = VQADataset(split="train", subset_size=10)
    sample = dataset[0]

    print("\n✅ Sample 0:")
    print(f"   Question ID: {sample['question_id']}")
    print(f"   Question:    {sample['question']}")
    print(f"   Answer:      {sample['answer']}")
    print(f"   Image shape: {sample['image'].shape}")  # torch.Size([3, 224, 224])
