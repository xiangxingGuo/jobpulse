import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch





class HFLocalExtractor:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_new_tokens: int = 512,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        dtype = torch.float16 if device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

    def extract(self, prompt: str) -> dict:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        input_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[0][input_len:]  # only newly generated tokens
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        return self._safe_parse_json(text)
    
    
    def _safe_parse_json(self, text: str) -> dict:
        """
        Extract the first JSON object from model output.
        """
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("No JSON object found in model output")

        json_str = text[start:end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Slow fallback (rare cases)
            return slow_fallback_extract_last_json(text)


def slow_fallback_extract_last_json(text: str) -> dict:
    """
    Slow but robust fallback:
    extract the LAST valid top-level JSON object from text.
    """
    depth = 0
    start = None
    candidates = []

    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i+1])
                    start = None

    if not candidates:
        raise ValueError("No JSON object found in model output")

    # Try from last to first
    for s in reversed(candidates):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            continue

    raise ValueError("Found JSON-like blocks, but none are valid JSON")
