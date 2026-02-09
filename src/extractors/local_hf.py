from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.schemas.job_schema import JobStructured
from pathlib import Path
import time

DEBUG_DIR = Path("data/raw/llm_debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


SCHEMA_VERSION = "v1"
PROMPT_VERSION = "v1"

def _strip_to_json(text: str) -> str:
    """
    Best-effort: extract the first JSON object from model output.
    """
    # Remove common wrappers
    text = text.strip()
    # Try direct parse first
    if text.startswith("{") and text.endswith("}"):
        return text
    # Find first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output")
    return m.group(0).strip()


def _normalize_skills(skills: list[str]) -> list[str]:
    # simple normalization; expand this later
    out = []
    for s in skills:
        s2 = s.strip().lower()
        if not s2:
            continue
        out.append(s2)
    # unique while preserving order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


@dataclass
class LocalHFExtractor:
    model_name: str
    device: str = "cuda"
    max_new_tokens: int = 384
    temperature: float = 0.1
    top_p: float = 0.9

    # Configure 4-bit quantization



    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        # 4-bit load (bitsandbytes)
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
        )
        self.model.eval()

        # Some models don't have pad token set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prompt(self, title: str, company: str, location: str, description: str) -> str:
        # Keep prompt concise and schema-focused to reduce hallucination.
        return f"""You are an information extraction system.
Extract structured fields from the job posting. Return ONLY valid JSON. No markdown. No commentary.

JSON schema:
{{
  "role_category": "MLE|DS|DE|SWE|RE|Other",
  "seniority": "Intern|NewGrad|Junior|Mid|Senior|Staff|Other",
  "work_mode": "Remote|Hybrid|Onsite|Unknown",
  "location": string|null,
  "visa": {{
    "requires_us_auth": boolean,
    "opt_cpt_ok": boolean,
    "sponsorship_mentioned": boolean
  }},
  "skills": string[],
  "requirements": string[],
  "benefits": string[],
  "years_required": number|null,
  "confidence": number (0.0-1.0)
}}

Guidelines:
- If not mentioned, use Unknown/Other/null and set lower confidence.
- "US work authorization required" => requires_us_auth=true.
- Mention of "OPT/CPT" => opt_cpt_ok=true.
- If "sponsorship" is mentioned => sponsorship_mentioned=true.
- skills should be short tokens like "python", "sql", "pytorch", "spark", "docker", "kubernetes", "llm", "nlp".

Job:
Title: {title}
Company: {company}
Location: {location}

Description:
{description}
"""

    @torch.inference_mode()
    def extract(self, title: str, company: str, location: str, description: str, debug: bool = False, debug_id: str = "job") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        prompt = self._prompt(title, company, location, description)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature if self.temperature > 0 else None,
            top_p=self.top_p if self.temperature > 0 else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        input_len = inputs["input_ids"].shape[1]
        completion_ids = gen[0][input_len:]
        completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        if debug:
            ts = int(time.time())
            p = DEBUG_DIR / f"{debug_id}_{ts}.txt"
            p.write_text(completion, encoding="utf-8")
            print(f"[DEBUG] Saved completion to {p}")
            print("[DEBUG] Completion head:\n", completion[:1200])

        # The decoded string includes the prompt; strip prompt prefix by taking the last JSON object
        try:
            json_text = _strip_to_json(completion)
            data = json.loads(json_text)

            # Validate with Pydantic
            obj = JobStructured.model_validate(data)

            # Normalize skills
            data2 = obj.model_dump()
            data2["skills"] = _normalize_skills(data2.get("skills") or [])
            return data2, None
        except Exception as e:
            return None, str(e)

    def extract_with_retries(
        self,
        title: str,
        company: str,
        location: str,
        description: str,
        retries: int = 2,
        debug: bool = False,
        debug_id: str = "job",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        # retry with slightly more deterministic settings if needed
        data, err = self.extract(title, company, location, description, debug=debug, debug_id=debug_id)
        if data is not None:
            return data, None

        last_err = err
        # Retry with temperature=0
        orig_temp = self.temperature
        self.temperature = 0.0
        try:
            for _ in range(retries):
                data, err = self.extract(title, company, location, description, debug=debug, debug_id=debug_id)
                if data is not None:
                    return data, None
                last_err = err
        finally:
            self.temperature = orig_temp

        return None, last_err
