from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, List, Literal
from pathlib import Path

from src.llm.providers.hf_plain import HFPlainExtractor
from src.llm.providers.hf_chat_lora import HFChatLoRAExtractor
from src.llm.providers.openai_compat_client import OpenAICompatClient
from src.llm.providers.openai_compat_providers import PROVIDERS
from src.llm.json_repair import parse_json_object


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

@dataclass
class ExtractResult:
    structured: Optional[Dict[str, Any]]
    raw_output: str
    parse_ok: bool
    parse_repaired: bool
    extractor: Dict[str, Any]
    usage: Optional[Dict[str, Any]] = None
    warnings: Optional[list[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ExtractionService:
    """
    Service for extracting structured information from job descriptions.
    """

    async def extract_local(
        self,
        *,
        job_id: str,
        jd_text: str,
        prompt_name: str = "jd_extract_v2",
        model: Literal["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", 
                       "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
                       "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"] = "Qwen/Qwen2.5-0.5B-Instruct",
        lora_path: Optional[str] =  None,
        mode: Literal["plain", "chat_lora"] = "plain",
        device: Literal["cuda", "cpu"] = "cuda",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: Optional[int] = None,
    ) -> ExtractResult:
        """
        Local extraction to produce structured JSON for the schema.
        May not used when deploying in cloud (due to GPU requirements)
        """
        
        prompt = _build_prompt(prompt_name, jd_text)

        do_sample = bool(temperature and temperature > 0)

        extractor_meta: Dict[str, Any] = {
            "mode": mode,
            "model": model,
            "lora_path": lora_path,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "do_sample": do_sample,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
        }

        if mode == "plain":
            extractor = HFPlainExtractor(
                    model_name=model,
                    device=device,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                )
        elif mode == "chat_lora":
            extractor = HFChatLoRAExtractor(
                    base_model=model,
                    lora_path=lora_path,
                    device=device,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                )
        else:
            raise ValueError(f"Unknown mode: {mode}")  

        result = extractor.extract_with_result(prompt)
        parse_ok = result.error is None and result.data is not None

        return ExtractResult(
            structured=result.data if parse_ok else None,
            raw_output=result.raw_output,
            parse_ok=parse_ok,
            parse_repaired=False,  # no repair step for local extraction for now
            extractor=extractor_meta,
        )
    
    async def extract_api(
        self,
        *,
        job_id: str,
        jd_text: str,
        prompt_name: str = "jd_extract_v2",
        provider: Literal["openai", "nvidia"] = "nvidia",
        model: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: int = 900,
        thinking: Literal["auto", "disabled", "enabled"] = "disabled",
    ) -> ExtractResult:
        """
            Extraction using an OpenAI-compatible API provider.
            Providers supported: openai, nvidia.

            Env:
            OPENAI_API_KEY for provider=openai
            NVIDIA_API_KEY for provider=nvidia
            Optional base URL override:
                OPENAI_BASE_URL, NVIDIA_BASE_URL
            """
        
        prov: str = provider
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

        extractor_meta = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "thinking": effective_thinking,
            "base_url": client.base_url,
        }

        return ExtractResult(
            structured=parsed if parse_ok else None,
            raw_output=content,
            parse_ok=parse_ok,
            parse_repaired=repaired_flag,
            extractor=extractor_meta,
            usage=usage,
        )
