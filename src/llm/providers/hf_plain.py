from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import BaseExtractor


class HFPlainExtractor(BaseExtractor):
    """
    Plain prompt -> model.generate -> parse JSON.

    Use this for:
      - baseline (no fine-tuning)
      - situations where you do NOT want chat template formatting

    Notes:
      - Expects `prompt` already includes everything (instructions + JD).
      - Decodes ONLY newly generated tokens to avoid "prompt+completion" mixing.
    """

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
    ) -> None:
        super().__init__(required_keys=required_keys, list_keys=list_keys)

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code

        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32
        self.dtype = dtype

        # If cuda, let HF decide sharding/placement; else keep on CPU
        if device_map is None:
            device_map = "auto" if device == "cuda" else None
        self.device_map = device_map

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        # Ensure pad token exists for generation
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

        # Move tensors to model device if needed
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # deterministic baseline
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode ONLY the completion tokens
        gen_ids = out[0][input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return text
