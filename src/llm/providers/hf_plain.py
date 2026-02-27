from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import BaseExtractor
from src.llm.json_repair import parse_json_object


class HFPlainExtractor(BaseExtractor):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_new_tokens: int = 512,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
        device_map: Optional[str] = None,
        required_keys: Optional[Sequence[str]] = None,
        list_keys: Optional[Sequence[str]] = None,

        # NEW: generation controls
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(required_keys=required_keys, list_keys=list_keys)

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code

        # generation params
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.seed = seed

        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32
        self.dtype = dtype

        if device_map is None:
            device_map = "auto" if device == "cuda" else None
        self.device_map = device_map

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self.dtype,
            device_map=self.device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        # Optional deterministic seeding (industrial reproducibility)
        if self.seed is not None:
            try:
                torch.manual_seed(self.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.seed)
            except Exception:
                pass

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=self.repetition_penalty,
        )

        # Only apply sampling params when sampling is enabled.
        # (HF can behave oddly if temperature=0 but do_sample=True)
        if self.do_sample:
            gen_kwargs["temperature"] = max(1e-6, float(self.temperature))
            gen_kwargs["top_p"] = float(self.top_p)
            if self.top_k and int(self.top_k) > 0:
                gen_kwargs["top_k"] = int(self.top_k)

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        gen_ids = out[0][input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return text