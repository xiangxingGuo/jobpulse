from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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

        return await self._run_report_generation(
            system=system,
            user_prompt=user_prompt,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            job_id=job_id,
            report_kind="job_report_v1",
        )

    async def generate_skill_gap_report(
        self,
        *,
        job_id: str,
        structured: Dict[str, Any],
        qc: Dict[str, Any],
        resume_profile: Dict[str, Any],
        skill_gap: Dict[str, Any],
        resume_text: Optional[str] = None,
        market_context: Optional[Dict[str, Any]] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1100,
    ) -> ReportResult:
        system = (
            "You are a careful career intelligence assistant. "
            "Write a concise, practical markdown report for a candidate evaluating fit for a target role. "
            "Use only the provided inputs. "
            "Do not invent experience, qualifications, or job requirements. "
            "Keep the report grounded, structured, and useful for job applications."
        )

        user_prompt = self._build_skill_gap_prompt(
            job_id=job_id,
            structured=structured,
            qc=qc,
            resume_profile=resume_profile,
            skill_gap=skill_gap,
            resume_text=resume_text,
            market_context=market_context,
        )

        return await self._run_report_generation(
            system=system,
            user_prompt=user_prompt,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            job_id=job_id,
            report_kind="skill_gap_report_v1",
        )

    async def _run_report_generation(
        self,
        *,
        system: str,
        user_prompt: str,
        provider: str,
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        job_id: str,
        report_kind: str,
    ) -> ReportResult:
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
            "report_kind": report_kind,
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

    def _build_skill_gap_prompt(
        self,
        *,
        job_id: str,
        structured: Dict[str, Any],
        qc: Dict[str, Any],
        resume_profile: Dict[str, Any],
        skill_gap: Dict[str, Any],
        resume_text: Optional[str],
        market_context: Optional[Dict[str, Any]],
    ) -> str:
        market_context = market_context or {}

        return (
            f"Job ID: {job_id}\n\n"
            f"Structured Job Data:\n{self._to_pretty_json(structured)}\n\n"
            f"QC Result:\n{self._to_pretty_json(qc)}\n\n"
            f"Resume Profile:\n{self._to_pretty_json(resume_profile)}\n\n"
            f"Skill Gap Analysis:\n{self._to_pretty_json(skill_gap)}\n\n"
            f"Market Context:\n{self._to_pretty_json(market_context)}\n\n"
            f"Resume Text:\n{resume_text or ''}\n\n"
            "Write a markdown report for the candidate with the following sections:\n"
            "1. Target role snapshot\n"
            "2. Overall fit assessment\n"
            "3. Evidence-backed strengths\n"
            "4. Main gaps (separate must-have vs nice-to-have when possible)\n"
            "5. Transferable signals\n"
            "6. Resume tailoring suggestions\n"
            "7. 7-day and 30-day action plan\n"
            "8. Final recommendation\n\n"
            "Requirements:\n"
            "- Be grounded in the provided data only.\n"
            "- Do not claim the candidate has skills that are not supported.\n"
            "- If evidence is weak, say it is weak or ambiguous.\n"
            "- Keep the tone practical and concise.\n"
        )

    def _to_pretty_json(self, obj: Any) -> str:
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

        for k in ("reasoning_content", "reasoning", "output_text", "text"):
            v = msg.get(k)
            if isinstance(v, str) and v.strip():
                return v

        text = choice0.get("text")
        if isinstance(text, str):
            return text

        return ""
