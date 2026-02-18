from __future__ import annotations

import json
from typing import Any, Dict, Optional, Literal

from src.orch.schema import JobStructured, QCResult, MatchOutput, ReportOutput
from src.llm.providers.openai_compat_client import OpenAICompatClient
from src.llm.providers.openai_compat_providers import PROVIDERS, ProviderName


def _build_report_prompt(structured: JobStructured, qc: QCResult, match: Optional[MatchOutput], resume_text: Optional[str]) -> str:
    return (
        "You are a career assistant. Write a concise markdown report.\n"
        "Rules:\n"
        "- Base everything on STRUCTURED_JSON. Do not invent requirements.\n"
        "- If resume_text is empty, state assumptions.\n"
        "Sections:\n"
        "1) Job summary (title/company/location)\n"
        "2) Key requirements (bullets)\n"
        "3) Skill gaps (bullets)\n"
        "4) 2-week action plan (bullets)\n\n"
        f"STRUCTURED_JSON:\n{json.dumps(structured, ensure_ascii=False, indent=2)}\n\n"
        f"QC:\n{json.dumps(qc, ensure_ascii=False, indent=2)}\n\n"
        f"MATCH:\n{json.dumps(match, ensure_ascii=False, indent=2) if match else 'null'}\n\n"
        f"RESUME_TEXT:\n{resume_text or ''}\n"
    )

def _get_message_text(resp: Dict[str, Any]) -> str:
    try:
        choice0 = (resp.get("choices") or [])[0] or {}
    except Exception:
        choice0 = {}

    msg = choice0.get("message") or {}
    content = msg.get("content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(p.get("text") or p.get("content") or "")
        return "".join(parts)

    for k in ("reasoning_content", "reasoning", "output_text", "text"):
        v = msg.get(k)
        if isinstance(v, str) and v.strip():
            return v

    v = choice0.get("text")
    if isinstance(v, str) and v.strip():
        return v

    return ""



async def generate_report_api(
    job_id: str,
    structured: JobStructured,
    qc: QCResult,
    match: Optional[MatchOutput] = None,
    resume_text: Optional[str] = None,
    provider: Literal["openai", "nvidia"] = "nvidia",
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> ReportOutput:
    """
    Generate a markdown report using an OpenAI-compatible API provider.
    Providers supported: openai, nvidia.
    """
    prov: ProviderName = provider  # type: ignore
    cfg = PROVIDERS[prov]
    if model is None:
        model = cfg.default_model

    client = OpenAICompatClient(provider=prov)

    user_content = _build_report_prompt(structured, qc, match, resume_text)
    messages = [
        {"role": "system", "content": "You write helpful, compact markdown reports."},
        {"role": "user", "content": user_content},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    resp = await client.chat_completions(payload)

    content = _get_message_text(resp)
    usage = resp.get("usage", {})

    if not content.strip():
        content = json.dumps(resp, ensure_ascii=False)


    return {
        "job_id": job_id,
        "report_md": content,
        "usage": usage,
        "meta": {"provider": provider, "model": model, "base_url": client.base_url},
    }
