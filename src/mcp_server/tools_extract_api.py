from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Literal


import httpx

from src.orch.schema import ExtractAPIOutput, JobStructured
from src.llm.providers.openai_compat_client import OpenAICompatClient
from src.llm.providers.openai_compat_providers import PROVIDERS
from src.llm.json_repair import parse_json_object


PROMPTS = {
    "jd_extract_v1": Path("src/llm/prompts/jd_extract_v1.txt"),
    "jd_extract_v2": Path("src/llm/prompts/jd_extract_v2.txt"),
    "jd_extract_v3": Path("src/llm/prompts/jd_extract_v3.txt"),
}

def _get_message_text(resp: Dict[str, Any]) -> str:
    """
    Robustly extract assistant text from OpenAI-compatible responses.
    Some providers may return message.content as null.
    """
    try:
        choice0 = (resp.get("choices") or [])[0] or {}
    except Exception:
        choice0 = {}

    msg = choice0.get("message") or {}

    content = msg.get("content", None)

    # 1) Standard: content is a string
    if isinstance(content, str):
        return content

    # 2) Some providers may return list-of-parts
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                # try common keys
                parts.append(p.get("text") or p.get("content") or "")
        return "".join(parts)

    # 3) Sometimes content is null; try fallbacks
    for k in ("reasoning_content", "reasoning", "output_text", "text"):
        v = msg.get(k)
        if isinstance(v, str) and v.strip():
            return v

    # 4) As a last resort, try older/nonstandard fields
    v = choice0.get("text")
    if isinstance(v, str) and v.strip():
        return v

    # 5) Give up: return empty string (caller may dump resp for debugging)
    return ""


def _build_prompt(prompt_name: str, jd_text: str) -> str:
    template = PROMPTS[prompt_name].read_text(encoding="utf-8", errors="ignore")
    # Assert check: Ensure the template contains the placeholder and that it gets replaced
    assert "{{JOB_DESCRIPTION}}" in template, "Prompt template missing {{JOB_DESCRIPTION}}"
    prompt = template.replace("{{JOB_DESCRIPTION}}", jd_text)
    assert "{{JOB_DESCRIPTION}}" not in prompt, "Placeholder not replaced"
    return prompt

async def extract_api(
    job_id: str,
    jd_text: str,
    prompt_name: str = "jd_extract_v2",
    provider: Literal["openai", "nvidia"] = "nvidia",
    model: Optional[str] = None,
    temperature: float = 0.6,
    max_tokens: int = 900,
    thinking: Literal["auto", "disabled", "enabled"] = "disabled",
) -> ExtractAPIOutput:
    """
    Extraction using an OpenAI-compatible API provider.
    Providers supported: openai, nvidia.

    Env:
      OPENAI_API_KEY for provider=openai
      NVIDIA_API_KEY for provider=nvidia
      Optional base URL override:
        OPENAI_BASE_URL, NVIDIA_BASE_URL
    """
    prov: ProviderName = provider  # type: ignore
    cfg = PROVIDERS[prov]
    if model is None:
        model = cfg.default_model

    prompt = _build_prompt(prompt_name, jd_text)

    system = (
        "You are an information extraction system. "
        "Return ONLY valid JSON matching the given schema. "
        "Do not include any explanation, markdown, or extra text."
    )

    payload: Dict[str, Any] = {
        "model": model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }

    # Kimi/NVIDIA: disable thinking by default unless explicitly enabled
    effective_thinking = thinking
    if effective_thinking == "auto":
        effective_thinking = "disabled" if provider == "nvidia" else "auto"

    if effective_thinking == "disabled":
        payload["extra_body"] = {"thinking": {"type": "disabled"}}
    elif effective_thinking == "enabled":
        payload["extra_body"] = {"thinking": {"type": "enabled"}}

    if provider == "openai":
        payload.pop("extra_body", None)  # OpenAI doesn't support thinking control (ignore if present)

    client = OpenAICompatClient(provider=prov)
    resp = await client.chat_completions(payload)

    content = _get_message_text(resp)
    usage = resp.get("usage", {})

    # If provider returned no content, keep the full response for debugging
    if not content.strip():
        content = json.dumps(resp, ensure_ascii=False)

    parsed, repaired_flag, used_text = parse_json_object(content)
    parse_ok = parsed is not None

    structured: Optional[JobStructured] = None
    if parse_ok:
        structured = parsed  # type: ignore

    return {
        "job_id": job_id,
        "structured": structured,
        "raw_output": content,
        "parse_ok": parse_ok,
        "used_text": used_text,
        "parse_repaired": repaired_flag,
        "usage": usage,
        "extractor": {"provider": provider, "model": model, "base_url": client.base_url},
    }