from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List, Literal

from src.orch.schema import ExtractLocalOutput, JobStructured
from src.llm.providers.hf_plain import HFPlainExtractor
from src.llm.providers.hf_chat_lora import HFChatLoRAExtractor

from pathlib import Path

PROMPTS = {
    "jd_extract_v1": Path("src/llm/prompts/jd_extract_v1.txt"),
    "jd_extract_v2": Path("src/llm/prompts/jd_extract_v2.txt"),
    "jd_extract_v3": Path("src/llm/prompts/jd_extract_v3.txt"),
}


def _build_prompt(prompt_name: str, jd_text: str) -> str:
    template = PROMPTS[prompt_name].read_text(encoding="utf-8", errors="ignore")
    # Assert check: Ensure the template contains the placeholder and that it gets replaced
    assert "{{JOB_DESCRIPTION}}" in template, "Prompt template missing {{JOB_DESCRIPTION}}"
    prompt = template.replace("{{JOB_DESCRIPTION}}", jd_text)
    assert "{{JOB_DESCRIPTION}}" not in prompt, "Placeholder not replaced"
    return prompt


def extract_local(
    job_id: str,
    jd_text: str,
    prompt_name: str = "jd_extract_v2",
    # model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    model: Literal["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"] = "Qwen/Qwen2.5-0.5B-Instruct",
    lora_path: Optional[str] = "models/qwen2.5-0.5b-jd-lora",
    mode: Literal["plain", "chat_lora"] = "plain",
    device: Literal["cuda", "cpu"] = "cuda",
    max_new_tokens: int = 1024
) -> ExtractLocalOutput:
    """
    Run local extraction to produce strict JSON for the schema.

    mode="plain": use HFPlainExtractor (baseline)
    mode="chat_lora": use HFChatLoRAExtractor (chat template; optional LoRA)
    """
    prompt = _build_prompt(prompt_name, jd_text)

    extractor_meta: Dict[str, Any] = {
        "mode": mode,
        "model": model,
        "lora_path": lora_path,
    }

    if mode == "plain":
        extractor = HFPlainExtractor(model_name=model, device=device, max_new_tokens=max_new_tokens)
    elif mode == "chat_lora":
        extractor = HFChatLoRAExtractor(base_model=model, lora_path=lora_path, device=device, max_new_tokens=max_new_tokens)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Use BaseExtractor's extract_with_result to keep raw_output / error
    result = extractor.extract_with_result(prompt)

    parse_ok = result.error is None and result.data is not None
    structured: Optional[JobStructured] = None
    if parse_ok:
        structured = result.data  # type: ignore

    return {
        "job_id": job_id,
        "structured": structured,
        "raw_output": result.raw_output,
        "parse_ok": parse_ok,
        "parse_repaired": False,  # no repair step for local extraction for now
        "extractor": extractor_meta,
    }
