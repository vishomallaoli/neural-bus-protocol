"""
Diagnostic test script to identify bus error source
"""
import torch
from pipeline import NeuralBUSPipeline
from data.vqa import VQADataset


def test_components_individually():
    """Test each component in isolation"""
    device = "cpu"
    torch.set_default_dtype(torch.float32)
    
    print("=" * 60)
    print("DIAGNOSTIC TEST - COMPONENT ISOLATION")
    print("=" * 60)
    
    # Test 1: Load dataset
    print("\n[Test 1] Loading dataset...")
    try:
        ds = VQADataset(split="val", subset_size=1)
        sample = ds[0]
        print(f"✅ Dataset loaded: {sample['question']}")
        print(f"   Image shape: {sample['image'].shape}")
        print(f"   Image dtype: {sample['image'].dtype}")
    except Exception as e:
        print(f"❌ Dataset failed: {e}")
        return
    
    # Test 2: Load pipeline
    print("\n[Test 2] Loading pipeline...")
    try:
        pipeline = NeuralBUSPipeline(device=device, use_mock=False)
        print("✅ Pipeline loaded")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return
    
    # Test 3: Load checkpoint
    print("\n[Test 3] Loading checkpoint...")
    try:
        state = torch.load("./checkpoints/nbus_vqa100k_e3.pt", map_location=device)
        pipeline.encoder.load_state_dict(state["encoder"])
        pipeline.decoder.load_state_dict(state["decoder"])
        
        # Force fp32
        pipeline.encoder = pipeline.encoder.to(device, dtype=torch.float32)
        pipeline.decoder = pipeline.decoder.to(device, dtype=torch.float32)
        print("✅ Checkpoint loaded")
    except FileNotFoundError:
        print("⚠️  Checkpoint not found - using random weights")
    except Exception as e:
        print(f"❌ Checkpoint loading failed: {e}")
        return
    
    # Test 4: Vision encoding
    print("\n[Test 4] Vision encoding...")
    try:
        image = sample["image"].to(device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            vision_features = pipeline.vision_encoder.encode(image)
        print(f"✅ Vision features: {vision_features.shape}, dtype: {vision_features.dtype}")
    except Exception as e:
        print(f"❌ Vision encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 5: Encoder adapter
    print("\n[Test 5] Encoder adapter...")
    try:
        with torch.no_grad():
            if vision_features.dim() == 1:
                vision_features = vision_features.unsqueeze(0)
            vision_features = vision_features.to(dtype=torch.float32)
            bus_vector = pipeline.encoder(vision_features)
        print(f"✅ BUS vector: {bus_vector.shape}, dtype: {bus_vector.dtype}")
    except Exception as e:
        print(f"❌ Encoder adapter failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 6: Decoder adapter
    print("\n[Test 6] Decoder adapter...")
    try:
        with torch.no_grad():
            if bus_vector.dim() == 1:
                bus_vector = bus_vector.unsqueeze(0)
            bus_vector = bus_vector.to(dtype=torch.float32)
            llm_embedding = pipeline.decoder(bus_vector)
        print(f"✅ LLM embedding: {llm_embedding.shape}, dtype: {llm_embedding.dtype}")
    except Exception as e:
        print(f"❌ Decoder adapter failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 7: Prompt formatting
    print("\n[Test 7] Prompt formatting...")
    try:
        prompt = pipeline.decoder.format_prompt("", sample["question"])
        print(f"✅ Prompt created ({len(prompt)} chars)")
        print(f"   First 100 chars: {prompt[:100]}")
    except Exception as e:
        print(f"❌ Prompt formatting failed: {e}")
        return
    
    # Test 8: Tokenization
    print("\n[Test 8] Tokenization...")
    try:
        inputs = pipeline.llm.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        print(f"✅ Tokenized: {input_ids.shape}, dtype: {input_ids.dtype}")
        print(f"   Token IDs: {input_ids[0][:10].tolist()}...")
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 9: LLM model check
    print("\n[Test 9] LLM model properties...")
    try:
        print(f"   Model dtype: {next(pipeline.llm.model.parameters()).dtype}")
        print(f"   Model device: {next(pipeline.llm.model.parameters()).device}")
        print(f"   Hidden size: {pipeline.llm.hidden_size}")
        print(f"   Vocab size: {pipeline.llm.vocab_size}")
        print("✅ LLM model properties OK")
    except Exception as e:
        print(f"❌ LLM model check failed: {e}")
        return
    
    # Test 10: Simple forward pass (no generation)
    print("\n[Test 10] LLM forward pass (no generation)...")
    try:
        with torch.no_grad():
            outputs = pipeline.llm.model(input_ids=input_ids)
            logits = outputs.logits
        print(f"✅ Forward pass: logits shape {logits.shape}, dtype: {logits.dtype}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 11: Generation with very short length
    print("\n[Test 11] Manual generation (1 token)...")
    try:
        with torch.no_grad():
            # Manual generation (avoid model.generate bug on M1)
            outputs = pipeline.llm.model(input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            output_ids = torch.cat([input_ids, next_token_id], dim=-1)
        
        print(f"✅ Generated 1 token: {output_ids.shape}")
        decoded = pipeline.llm.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"   Decoded: '{decoded[:50]}...'")
    except Exception as e:
        print(f"❌ Generation (1 token) failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 12: Full generation
    print("\n[Test 12] Full generation (16 tokens)...")
    try:
        pred = pipeline.llm.generate(prompt, max_new_tokens=16)
        print(f"✅ Full generation successful!")
        print(f"   Prediction: '{pred}'")
    except Exception as e:
        print(f"❌ Full generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_components_individually()