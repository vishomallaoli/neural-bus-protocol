"""
Language Model Interface
Wraps HuggingFace LLMs (Mistral, LLaMA, etc.)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LanguageModel:
    """LLM wrapper for answer generation + logits access"""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        device: str = "cpu",
    ):
        self.device = device
        self.model_name = model_name

        print(f"Loading {model_name}...")

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # (Optional but nice for causal LMs)
        self.tokenizer.padding_side = "left"

        # --- Dtype selection ---
        # Use fp16 on cuda/mps where it’s supported; fp32 on cpu
        dtype = torch.float16 if device == "cuda" else torch.float32

        # --- Model ---
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        # Always move model to target device *and* dtype explicitly
        # (This helps avoid "placeholder storage not allocated" on MPS.)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.hidden_size = int(self.model.config.hidden_size)
        self.vocab_size = int(self.model.config.vocab_size)
        print(
            f"✅ LLM loaded (hidden_size={self.hidden_size}, "
            f"vocab_size={self.vocab_size}, device={self.device}, dtype={dtype})"
        )

    # ------------------------------------------------------------
    def _to_device(self, batch_encoding):
        # BatchEncoding.to(device) works, but be explicit and safe:
        return {k: v.to(self.device) for k, v in batch_encoding.items()}

    # ------------------------------------------------------------
    def token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Return input token embeddings for given input_ids.
        Shape: [batch, seq_len, hidden_size]
        """
        emb = self.model.get_input_embeddings()(input_ids)
        # Ensure embeddings live on the same device/dtype as the model
        return emb.to(self.device, dtype=next(self.model.parameters()).dtype)

    # ============================================================
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """
        Generate an answer from a prompt (inference mode).
        Only decodes the *new* tokens, not the whole prompt again.
        """
        inputs = self._to_device(self.tokenizer(prompt, return_tensors="pt"))

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        prompt_len = inputs["input_ids"].shape[1]
        new_token_ids = output_ids[0][prompt_len:]
        answer = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        return answer

    # ============================================================
    def get_logits(self, prompt: str) -> torch.Tensor:
        """
        Get output logits (for knowledge distillation).
        Returns: [batch(=1), seq_len, vocab_size]
        """
        inputs = self._to_device(self.tokenizer(prompt, return_tensors="pt"))
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
        return outputs.logits

    # ============================================================
    def forward_logits(self, prompt: str) -> torch.Tensor:
        """
        Alias for pipeline integration (same as get_logits).
        """
        return self.get_logits(prompt)


# ============================================================
class MockLLM:
    """Mock LLM for testing and mock mode"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model_name = "mock"
        self.hidden_size = 4096
        self.vocab_size = 32000
        print("✅ Mock LLM loaded")

        # Provide a tiny config-like object so evaluators can read vocab_size
        class _Cfg:
            def __init__(self, hidden_size, vocab_size):
                self.hidden_size = hidden_size
                self.vocab_size = vocab_size
        self._config = _Cfg(self.hidden_size, self.vocab_size)

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        p = prompt.lower()
        if "color" in p:
            return "red"
        elif "what" in p:
            return "various objects"
        elif "how many" in p:
            return "three"
        else:
            return "yes"

    def get_logits(self, prompt: str) -> torch.Tensor:
        # Return dummy logits with correct shape
        return torch.randn(1, 50, self.vocab_size, device=self.device)

    # Mimic HF API bits used elsewhere
    @property
    def config(self):
        return self._config

    @property
    def model(self):
        # Minimal shim with callable forward + generate
        class _Out:
            def __init__(self, shape, device):
                self.logits = torch.randn(*shape, device=device)

        class _Emb:
            def __call__(self, input_ids):
                b, t = input_ids.shape
                return torch.randn(b, t, 4096, device=input_ids.device)

        class _Dummy:
            def __call__(self, *args, **kwargs):
                device = None
                if "input_ids" in kwargs and isinstance(kwargs["input_ids"], torch.Tensor):
                    device = kwargs["input_ids"].device
                elif "inputs_embeds" in kwargs and isinstance(kwargs["inputs_embeds"], torch.Tensor):
                    device = kwargs["inputs_embeds"].device
                else:
                    device = "cpu"
                return _Out((1, 50, 32000), device=device)

            def generate(self, *args, **kwargs):
                # Return random token ids; shape [1, 50]
                return torch.randint(0, 32000, (1, 50), device=kwargs.get("input_ids", torch.tensor([[0]])).device)

            def get_input_embeddings(self):
                return _Emb()

            def parameters(self):
                # Yield a dummy tensor for dtype checks if ever needed
                yield torch.randn(1)

        return _Dummy()


# Quick test
if __name__ == "__main__":
    print("Testing Language Model...")

    llm = MockLLM()
    test_prompt = "[INST] What color is the cup? [/INST]"
    answer = llm.generate(test_prompt)
    print(f"✅ Generated answer: '{answer}'")

    logits = llm.get_logits(test_prompt)
    print(f"✅ Logits shape: {logits.shape}")
