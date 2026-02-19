import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from peft import PeftModel
from src.llm.json_repair import parse_json_object




class HFLocalExtractor:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_new_tokens: int = 512,
        lora_path: str | None = None,
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

        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model.eval()


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
            # raise ValueError("No JSON object found in model output")
                return {
                    "error": "No JSON object found in model output",
                    "raw_output": text
                }

        json_str = text[start:end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Slow fallback (rare cases)
            return slow_fallback_extract_last_json(text)


