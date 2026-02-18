from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import httpx

from src.orch.schema import ExtractAPIOutput, JobStructured

PROMPTS = {
    "jd_extract_v2": Path("src/llm/prompts/jd_extract_v2.txt"),
}


def _build_prompt(prompt_name: str, jd_text: str) -> str:
    template = PROMPTS[prompt_name].read_text(encoding="utf-8", errors="ignore")
    assert "{{JOB_DESCRIPTION}}" in template, "Prompt template missing {{JOB_DESCRIPTION}}"
    prompt = template.replace("{{JOB_DESCRIPTION}}", jd_text)
    assert "{{JOB_DESCRIPTION}}" not in prompt, "Placeholder not replaced"
    return prompt


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        # drop first fence line
        lines = lines[1:] if lines else lines
        # drop last fence line if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _extract_jsonish_tail(text: str) -> str:
    t = text
    i_obj = t.find("{")
    i_arr = t.find("[")
    if i_obj == -1 and i_arr == -1:
        return t.strip()
    if i_obj == -1:
        start = i_arr
    elif i_arr == -1:
        start = i_obj
    else:
        start = min(i_obj, i_arr)
    return t[start:].strip()


def _truncate_to_last_balanced(text: str) -> Optional[str]:
    """
    If output contains JSON followed by extra text, try to truncate
    at the last point where brackets are balanced (outside strings).
    """
    t = text
    stack: List[str] = []
    in_str = False
    esc = False
    started = False
    last_balanced_end: Optional[int] = None

    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "{[":
            stack.append(ch)
            started = True
        elif ch in "}]":
            if stack:
                top = stack[-1]
                if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                    stack.pop()

        if started and not in_str and not stack:
            last_balanced_end = i + 1

    if last_balanced_end is not None:
        return t[:last_balanced_end].strip()
    return None


def _repair_brackets(text: str, max_append: int = 256) -> str:
    """
    Best-effort: strip fences, keep json-ish tail, then balance {} and [].
    """
    t = _strip_code_fences(text)
    t = _extract_jsonish_tail(t)

    # First: if we can truncate to a balanced JSON, do it (fixes Extra data)
    truncated = _truncate_to_last_balanced(t)
    if truncated:
        return truncated

    # Otherwise: append missing closers (fixes missing } / ])
    stack: List[str] = []
    in_str = False
    esc = False

    for ch in t:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                top = stack[-1]
                if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                    stack.pop()

    closes = []
    for opener in reversed(stack):
        closes.append("}" if opener == "{" else "]")
    if closes:
        t = t + "".join(closes)[:max_append]
    return t.strip()


def _parse_json(text: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Returns (obj, repaired_flag)
    """
    try:
        obj = json.loads(text)
        return (obj if isinstance(obj, dict) else None, False)
    except Exception:
        pass

    repaired = _repair_brackets(text)
    try:
        obj = json.loads(repaired)
        return (obj if isinstance(obj, dict) else None, True)
    except Exception:
        return (None, True)


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 90.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()


async def extract_api(
    job_id: str,
    jd_text: str,
    prompt_name: str = "jd_extract_v2",
    provider: str = "openai_compat",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 900,
) -> ExtractAPIOutput:
    """
    Fallback extraction using an OpenAI-compatible API.

    Environment:
      OPENAI_API_KEY (required)
      OPENAI_BASE_URL (optional, default https://api.openai.com)
    """
    if provider != "openai_compat":
        raise ValueError("Only openai_compat is supported in MVP")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    prompt = _build_prompt(prompt_name, jd_text)

    system = (
        "You are an information extraction system. "
        "Return ONLY valid JSON matching the given schema. "
        "Do not include any explanation, markdown, or extra text."
    )

    payload = {
        "model": model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }

    client = OpenAICompatClient(base_url=base_url, api_key=api_key)
    resp = await client.chat_completions(payload)

    content = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})

    parsed, repaired_flag = _parse_json(content)
    parse_ok = parsed is not None

    structured: Optional[JobStructured] = None
    if parse_ok:
        structured = parsed  # type: ignore

    return {
        "job_id": job_id,
        "structured": structured,
        "raw_output": content,
        "parse_ok": parse_ok,
        "parse_repaired": repaired_flag,
        "usage": usage,
        "extractor": {"provider": provider, "model": model, "base_url": base_url},
    }
