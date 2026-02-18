from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

import httpx

from src.orch.schema import JobStructured, QCResult, MatchOutput, ReportOutput


class OpenAICompatClient:
    """
    Minimal OpenAI-compatible client (chat.completions-like).
    You can swap to a cheaper provider later as long as endpoint is compatible.
    """
    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def chat(self, model: str, messages: list[dict], temperature: float, max_tokens: int) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()


def _build_report_prompt(structured: JobStructured, qc: QCResult, match: Optional[MatchOutput], resume_text: Optional[str]) -> str:
    return (
        "You are a career assistant. Write a concise markdown report.\n"
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


async def generate_report_api(
    job_id: str,
    structured: JobStructured,
    qc: QCResult,
    match: Optional[MatchOutput] = None,
    resume_text: Optional[str] = None,
    provider: str = "openai_compat",
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> ReportOutput:
    """
    Generate a markdown report using an OpenAI-compatible API.
    """
    if provider != "openai_compat":
        raise ValueError("Only openai_compat is supported in MVP")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    client = OpenAICompatClient(base_url=base_url, api_key=api_key)

    user_content = _build_report_prompt(structured, qc, match, resume_text)
    messages = [
        {"role": "system", "content": "You write helpful, compact markdown reports."},
        {"role": "user", "content": user_content},
    ]

    resp = await client.chat(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)

    # OpenAI-style parsing
    content = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})

    return {
        "job_id": job_id,
        "report_md": content,
        "usage": usage,
        "meta": {"provider": provider, "model": model, "base_url": base_url},
    }
