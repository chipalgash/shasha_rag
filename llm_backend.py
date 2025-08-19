class LLMBackend:
    def generate(self, system: str, prompt: str, max_new_tokens: int = 128) -> str:
        raise NotImplementedError

class OllamaBackend(LLMBackend):
    def __init__(self, model: str = "Qwen2.5:7b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")
        try:
            import requests  # lazy
            self._requests = requests
        except Exception as e:
            raise RuntimeError("requests is required for Ollama backend") from e

    def generate(self, system: str, prompt: str, max_new_tokens: int = 128) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0, "num_predict": max_new_tokens},
            "system": system,
            "prompt": prompt,
        }
        r = self._requests.post(f"{self.host}/api/generate", json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")

class HFLocalBackend(LLMBackend):
    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct", device: Optional[str] = None, load_4bit: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if load_4bit:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", quantization_config=bnb, torch_dtype=torch.bfloat16, trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

    def generate(self, system: str, prompt: str, max_new_tokens: int = 128) -> str:
        from transformers import TextStreamer
        import torch
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        # Simple chat template handling
        if hasattr(self.tok, "apply_chat_template"):
            text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            text = system + "\n" + prompt
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, temperature=0.0)
        gen = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return gen