#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForQuestionAnswering

from pipeline import NeuralBUSPipeline
from train import VQADataset  # reuse your dataset class

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_student_pipeline():
    pipe = NeuralBUSPipeline(device=device, use_mock=False)
    pipe.llm.model.eval()
    return pipe

def load_blip_vqa():
    model_name = "Salesforce/blip-vqa-base"
    print(f"Loading BLIP-VQA baseline: {model_name}")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model

def eval_models(split="val", subset_size=None, batch_size=8):
    dataset = VQADataset(split=split, subset_size=subset_size)
    loader = DataLoader(dataset, batch_size=batch_size)

    student = load_student_pipeline()
    blip_proc, blip = load_blip_vqa()

    student_correct = 0
    blip_correct = 0
    total = 0

    for batch in loader:
        images, questions, answers = batch  # adapt to your dataset

        # Ensure we have PIL images if BLIP needs them
        imgs_pil = [img if hasattr(img, "size") else to_pil_image(img) for img in images]

        # Student predictions
        student_preds = []
        for img, q in zip(imgs_pil, questions):
            ans = student.answer(img, q)  # or whatever your pipeline uses
            student_preds.append(ans.strip().lower())

        # BLIP-VQA predictions
        inputs = blip_proc(
            images=imgs_pil,
            text=list(questions),
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            out_ids = blip.generate(**inputs, max_new_tokens=10)
        blip_texts = blip_proc.batch_decode(out_ids, skip_special_tokens=True)
        blip_preds = [t.strip().lower() for t in blip_texts]

        # Simple exact match accuracy
        for gt, s_pred, t_pred in zip(answers, student_preds, blip_preds):
            gt_norm = str(gt).strip().lower()
            if s_pred == gt_norm:
                student_correct += 1
            if t_pred == gt_norm:
                blip_correct += 1
            total += 1

    print(f"Student (BUS + DistilGPT2) accuracy: {student_correct/total:.3f}")
    print(f"BLIP-VQA baseline accuracy:        {blip_correct/total:.3f}")

if __name__ == "__main__":
    eval_models(split="val", subset_size=1000, batch_size=8)
