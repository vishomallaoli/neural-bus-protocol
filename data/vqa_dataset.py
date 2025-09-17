# FILE: data/vqa_dataset.py

import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Dict, Tuple, Optional


class VQADataset(Dataset):
    """
    VQA v2 Dataset loader for Neural BUS training/evaluation.
    Requires the actual VQA v2 annotations, questions, and COCO images.
    """

    def __init__(
        self,
        data_dir: str = "data/vqa_v2",
        split: str = "train",
        subset_size: Optional[int] = 5000,
        transform: Optional[transforms.Compose] = None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.subset_size = subset_size

        # Default transform for ResNet-50
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transform

        # Load annotations and questions
        self.annotations = self._load_annotations()
        self.questions = self._load_questions()

        # Merge questions and answers
        self.data = self._merge_qa_pairs()

        # Apply subset limit
        if subset_size and len(self.data) > subset_size:
            self.data = self.data[:subset_size]

        print(f"Loaded {len(self.data)} VQA samples for {split} split")

    def _load_annotations(self) -> List[Dict]:
        """Load VQA v2 annotations (answers)."""
        anno_file = os.path.join(
            self.data_dir, f"v2_mscoco_{self.split}2014_annotations.json"
        )
        if not os.path.exists(anno_file):
            raise FileNotFoundError(f"Missing annotation file: {anno_file}")
        with open(anno_file, "r") as f:
            data = json.load(f)
        return data["annotations"]

    def _load_questions(self) -> List[Dict]:
        """Load VQA v2 questions."""
        q_file = os.path.join(
            self.data_dir, f"v2_OpenEnded_mscoco_{self.split}2014_questions.json"
        )
        if not os.path.exists(q_file):
            raise FileNotFoundError(f"Missing question file: {q_file}")
        with open(q_file, "r") as f:
            data = json.load(f)
        return data["questions"]

    def _merge_qa_pairs(self) -> List[Dict]:
        """Merge questions and annotations by question_id."""
        anno_dict = {a["question_id"]: a for a in self.annotations}

        merged = []
        for q in self.questions:
            q_id = q["question_id"]
            if q_id in anno_dict:
                merged.append(
                    {
                        "question_id": q_id,
                        "image_id": q["image_id"],
                        "question": q["question"],
                        "answers": anno_dict[q_id]["answers"],
                        "most_frequent_answer": self._get_most_frequent_answer(
                            anno_dict[q_id]["answers"]
                        ),
                    }
                )
        return merged

    def _get_most_frequent_answer(self, answers: List[Dict]) -> str:
        """Get the most frequent answer from 10 human annotations."""
        answer_counts = {}
        for ans in answers:
            answer = ans["answer"].lower().strip()
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        return max(answer_counts, key=answer_counts.get)

    def _get_image_path(self, image_id: str) -> str:
        """Get full path to COCO image file."""
        img_dir = "train2014" if self.split == "train" else "val2014"
        img_name = f"COCO_{img_dir}_{int(image_id):012d}.jpg"
        img_path = os.path.join(self.data_dir, "images", img_dir, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        return img_path

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str, str]:
        """
        Returns:
            image: Preprocessed image tensor [3, 224, 224]
            question: Question string
            answer: Most frequent answer string
            question_id: Unique identifier
        """
        item = self.data[idx]

        # Load and preprocess image
        image_path = self._get_image_path(item["image_id"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return (
            image,
            item["question"],
            item["most_frequent_answer"],
            str(item["question_id"]),
        )


def create_vqa_dataloaders(
    data_dir: str = "data/vqa_v2",
    batch_size: int = 16,
    subset_size: int = 5000,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for VQA v2."""

    train_dataset = VQADataset(
        data_dir=data_dir, split="train", subset_size=subset_size
    )
    val_dataset = VQADataset(
        data_dir=data_dir, split="val", subset_size=subset_size // 5
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing VQA Dataset Loader...")

    train_loader, val_loader = create_vqa_dataloaders(
        subset_size=100, batch_size=4
    )

    for batch_idx, (images, questions, answers, q_ids) in enumerate(train_loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"Images shape: {images.shape}")
        print(f"Questions: {questions}")
        print(f"Answers: {answers}")
        print(f"Question IDs: {q_ids}")
        break

    print("\nDataset loaded successfully!")
