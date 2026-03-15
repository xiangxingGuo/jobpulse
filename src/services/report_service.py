from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from src.llm.providers.openai_compat_client import OpenAICompatClient
from src.llm.providers.openai_compat_providers import PROVIDERS

@dataclass
class ReportResult:
    report_md: str
    meta: Dict[str, Any]
    usage: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReportService:
    """
    Graph-agnostic report generation service.
    """

    async def generate(
        self,
        *,
        job_id: str,
        structured: Dict[str, Any],
        qc: Dict[str, Any],
        match: Optional[Dict[str, Any]] = None,
        resume_text: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 900,
    ) -> ReportResult:
        system = (
            "You are a concise career intelligence assistant. "
            "Write a clear markdown report based on the structured job data and QC result. "
            "Do not invent facts that are not supported by the inputs."
        )

        user_prompt = self._build_prompt(
            job_id=job_id,
            structured=structured,
            qc=qc,
            match=match,
            resume_text=resume_text,
        )

        prov = provider
        cfg = PROVIDERS[prov]
        if model is None:
            model = cfg.default_model

        if not model:
            raise ValueError(f"No model resolved for provider={provider}")
        
        client = OpenAICompatClient(provider=prov)

        payload = {
            "model": model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        }


        resp = await client.chat_completions(payload)

        report_md = self._get_message_text(resp).strip()
        usage = resp.get("usage", {})

        if not report_md:
            report_md = "# Report generation failed\n\nThe provider returned an empty response."

        meta = {
            "provider": provider,
            "model": model,
            "base_url": client.base_url,
            "job_id": job_id,
        }

        return ReportResult(
            report_md=report_md,
            meta=meta,
            usage=usage,
        )

    def _build_prompt(
        self,
        *,
        job_id: str,
        structured: Dict[str, Any],
        qc: Dict[str, Any],
        match: Optional[Dict[str, Any]],
        resume_text: Optional[str],
    ) -> str:
        return (
            f"Job ID: {job_id}\n\n"
            f"QC Result:\n{self._to_pretty_json(qc)}\n\n"
            f"Structured Job Data:\n{self._to_pretty_json(structured)}\n\n"
            f"Match Data:\n{self._to_pretty_json(match)}\n\n"
            f"Resume Text:\n{resume_text or ''}\n\n"
            "Write a markdown report with:\n"
            "1. Role summary\n"
            "2. Core requirements\n"
            "3. Preferred qualifications\n"
            "4. Key risks / ambiguities\n"
            "5. Short recommendation\n"
        )

    def _to_pretty_json(self, obj: Any) -> str:
        import json
        return json.dumps(obj, ensure_ascii=False, indent=2)

    def _get_message_text(self, resp: Dict[str, Any]) -> str:
        try:
            choice0 = (resp.get("choices") or [])[0] or {}
        except Exception:
            choice0 = {}

        msg = choice0.get("message") or {}
        content = msg.get("content")

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

        text = choice0.get("text")
        if isinstance(text, str):
            return text

        return ""