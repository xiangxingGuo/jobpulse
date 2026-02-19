from __future__ import annotations

from typing import Optional, Sequence

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.llm.json_repair import parse_json_object

from .base import BaseExtractor

# Keep this identical across training + inference if you include it in your dataset/messages
DEFAULT_SYSTEM_MSG = "You are an information extraction system."


class HFChatLoRAExtractor(BaseExtractor):
    """
    Chat-template inference + optional LoRA adapter.

    Use this for:
      - models trained using messages/chat templates
      - LoRA fine-tuned adapters (PEFT)

    Key point:
      - Uses tokenizer.apply_chat_template(..., add_generation_prompt=True)
        so the model sees the same "it's assistant turn now" marker it saw in training.
    """

    def __init__(
        self,
        base_model: str,
        lora_path: Optional[str] = None,
        system_msg: str = DEFAULT_SYSTEM_MSG,
        device: str = "cuda",
        max_new_tokens: int = 512,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
        device_map: Optional[str] = None,
        required_keys: Optional[Sequence[str]] = None,
        list_keys: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(required_keys=required_keys, list_keys=list_keys)

        self.base_model = base_model
        self.lora_path = lora_path
        self.system_msg = system_msg
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code

        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32
        self.dtype = dtype

        if device_map is None:
            device_map = "auto" if device == "cuda" else None
        self.device_map = device_map

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=self.dtype,
            device_map=self.device_map,
            trust_remote_code=trust_remote_code,
        )

        if lora_path:
            # Lazy import so baseline users don't need peft installed
            from peft import PeftModel  # type: ignore
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.model.eval()

    def _build_chat_text(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_msg},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            # add_generation_prompt=True inserts the assistant turn marker
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback: if tokenizer doesn't support templates, degrade gracefully
        return user_prompt

    def _generate(self, user_prompt: str) -> str:
        text = self._build_chat_text(user_prompt)
        inputs = self.tokenizer(text, return_tensors="pt")

        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = out[0][input_len:]
        completion = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return completion
