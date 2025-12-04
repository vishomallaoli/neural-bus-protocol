from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "distilbert/distilgpt2"

print("Loading tokenizer/model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

prompt = "Question: Where is he looking?\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print("Output ids:", output_ids)
print("Decoded:", tokenizer.decode(output_ids[0], skip_special_tokens=True))
