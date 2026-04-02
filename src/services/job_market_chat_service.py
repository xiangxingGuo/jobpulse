from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional

from src.llm.json_repair import parse_json_object
from src.llm.providers.openai_compat_client import OpenAICompatClient
from src.llm.providers.openai_compat_providers import PROVIDERS
from src.services.job_market_chat_prompt import build_job_market_chat_messages
from src.services.job_search_service import JobSearchService
from src.services.resume_service import ResumeService
from src.services.skill_gap_service import SkillGapService


@dataclass
class JobMarketChatArtifacts:
    retrieved_jobs: List[Dict[str, Any]]
    target_job: Dict[str, Any]
    resume_profile: Dict[str, Any]
    skill_gap: Dict[str, Any]
    llm: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JobMarketChatService:
    """
    Single-turn grounded career/job-market chat.

    v1:
    - retrieves local job context
    - optionally uses target job / resume / skill-gap
    - returns strict JSON answer + source list
    """

    def __init__(
        self,
        *,
        job_search_service: Optional[JobSearchService] = None,
        resume_service: Optional[ResumeService] = None,
        skill_gap_service: Optional[SkillGapService] = None,
    ) -> None:
        self.job_search_service = job_search_service or JobSearchService()
        self.resume_service = resume_service or ResumeService()
        self.skill_gap_service = skill_gap_service or SkillGapService(
            job_search_service=self.job_search_service,
            resume_service=self.resume_service,
        )

    async def chat(
        self,
        *,
        question: str,
        top_k: int = 5,
        resume_text: Optional[str] = None,
        job_id: Optional[str] = None,
        provider: Literal["openai", "nvidia"] = "openai",
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        thinking: Literal["auto", "disabled", "enabled"] = "disabled",
    ) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            raise ValueError("question is empty")

        retrieved_jobs = self._retrieve_jobs(question=question, top_k=top_k)
        target_job = self._get_target_job(job_id)
        resume_profile = self._build_resume_profile(resume_text)
        skill_gap = self._build_optional_skill_gap(
            resume_text=resume_text,
            job_id=job_id,
        )

        llm_result = await self._run_llm(
            question=question,
            retrieved_jobs=retrieved_jobs,
            target_job=target_job,
            resume_profile=resume_profile,
            skill_gap=skill_gap,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
        )

        answer = llm_result.get("answer") or self._fallback_answer(
            question=question,
            retrieved_jobs=retrieved_jobs,
            target_job=target_job,
        )
        sources = llm_result.get("sources") or self._fallback_sources(
            retrieved_jobs=retrieved_jobs,
            target_job=target_job,
        )

        artifacts = JobMarketChatArtifacts(
            retrieved_jobs=retrieved_jobs,
            target_job=target_job,
            resume_profile=resume_profile,
            skill_gap=skill_gap,
            llm={
                "raw_output": llm_result.get("raw_output", ""),
                "parse_ok": llm_result.get("parse_ok"),
                "parse_repaired": llm_result.get("parse_repaired"),
                "provider": llm_result.get("provider"),
                "model": llm_result.get("model"),
                "base_url": llm_result.get("base_url"),
                "usage": llm_result.get("usage") or {},
            },
        )

        return {
            "answer": answer,
            "sources": sources,
            "meta": {
                "top_k": top_k,
                "provider": llm_result.get("provider"),
                "model": llm_result.get("model"),
                "job_id": job_id,
                "used_resume": bool((resume_text or "").strip()),
                "used_skill_gap": bool(skill_gap),
            },
            "artifacts": artifacts.to_dict(),
        }

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _retrieve_jobs(self, *, question: str, top_k: int) -> List[Dict[str, Any]]:
        hits = self.job_search_service.search_jobs(question, top_k=top_k)
        rows = []
        for h in hits:
            rows.append(
                {
                    "job_id": h.job_id,
                    "title": h.title,
                    "company": h.company,
                    "location": h.location,
                    "url": h.url,
                    "score": h.score,
                }
            )
        return rows

    def _get_target_job(self, job_id: Optional[str]) -> Dict[str, Any]:
        if not job_id:
            return {}

        row = self.job_search_service.get_job_by_id(job_id) or {}
        if not row:
            return {}

        return {
            "job_id": str(row.get("job_id", "")),
            "title": row.get("title"),
            "company": row.get("company"),
            "location": row.get("location") or row.get("location_text"),
            "url": row.get("url"),
            "skills": row.get("skills", []) or [],
            "description": self._truncate_text(
                row.get("description") or row.get("jd_text") or "",
                max_chars=2500,
            ),
        }

    def _truncate_text(self, text: str, max_chars: int = 2500) -> str:
        text = (text or "").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated]"

    def _json_safe_dict(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        import json

        return json.loads(json.dumps(obj, ensure_ascii=False, default=str))

    def _build_resume_profile(self, resume_text: Optional[str]) -> Dict[str, Any]:
        if not resume_text or not resume_text.strip():
            return {}
        profile = self.resume_service.parse_profile(resume_text)
        return profile.model_dump()

    def _build_optional_skill_gap(
        self,
        *,
        resume_text: Optional[str],
        job_id: Optional[str],
    ) -> Dict[str, Any]:
        if not resume_text or not resume_text.strip():
            return {}
        if not job_id:
            return {}

        try:
            out = self.skill_gap_service.analyze(
                resume_text=resume_text,
                job_id=job_id,
                include_market_context=True,
                market_top_k=5,
            )
            sg = out.get("skill_gap") or {}

            return {
                "fit_score": sg.get("fit_score"),
                "fit_band": sg.get("fit_band"),
                "summary": sg.get("summary"),
                "top_strengths": [x.get("skill") for x in (sg.get("strengths") or [])[:5]],
                "top_gaps": [x.get("skill") for x in (sg.get("gaps") or [])[:5]],
                "top_transferable": [
                    x.get("skill") for x in (sg.get("transferable_signals") or [])[:5]
                ],
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # LLM layer
    # ------------------------------------------------------------------

    async def _run_llm(
        self,
        *,
        question: str,
        retrieved_jobs: List[Dict[str, Any]],
        target_job: Dict[str, Any],
        resume_profile: Dict[str, Any],
        skill_gap: Dict[str, Any],
        provider: Literal["openai", "nvidia"],
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        thinking: Literal["auto", "disabled", "enabled"],
    ) -> Dict[str, Any]:
        prov = provider
        cfg = PROVIDERS[prov]
        if model is None:
            model = cfg.default_model

        if not model:
            raise ValueError(f"No model resolved for provider={provider}")

        messages = build_job_market_chat_messages(
            question=question,
            retrieved_jobs=retrieved_jobs,
            resume_profile=resume_profile,
            skill_gap=skill_gap,
            target_job=target_job,
        )

        payload: Dict[str, Any] = {
            "model": model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "messages": messages,
        }

        effective_thinking = thinking
        if effective_thinking == "auto":
            effective_thinking = "disabled" if provider == "nvidia" else "auto"

        if effective_thinking == "disabled":
            payload["extra_body"] = {"thinking": {"type": "disabled"}}
        elif effective_thinking == "enabled":
            payload["extra_body"] = {"thinking": {"type": "enabled"}}

        if provider == "openai":
            payload.pop("extra_body", None)

        client = OpenAICompatClient(provider=prov)
        resp = await client.chat_completions(payload)

        raw_output = self._get_message_text(resp)
        usage = resp.get("usage", {})

        parsed, repaired_flag, _ = parse_json_object(raw_output)
        parse_ok = isinstance(parsed, dict)

        normalized = self._normalize_output(parsed if parse_ok else None)

        return {
            "answer": normalized.get("answer", ""),
            "sources": normalized.get("sources", []),
            "raw_output": raw_output,
            "parse_ok": parse_ok,
            "parse_repaired": repaired_flag,
            "usage": usage,
            "provider": provider,
            "model": model,
            "base_url": client.base_url,
        }

    def _normalize_output(self, parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            return {"answer": "", "sources": []}

        answer = parsed.get("answer")
        if not isinstance(answer, str):
            answer = ""

        sources_out: List[Dict[str, Any]] = []
        for raw in parsed.get("sources", []) or []:
            if not isinstance(raw, dict):
                continue
            job_id = str(raw.get("job_id", "")).strip()
            title = str(raw.get("title", "")).strip()
            company = str(raw.get("company", "")).strip()
            reason = str(raw.get("reason", "")).strip()
            if not job_id:
                continue
            sources_out.append(
                {
                    "job_id": job_id,
                    "title": title,
                    "company": company,
                    "reason": reason,
                }
            )

        return {
            "answer": answer.strip(),
            "sources": sources_out[:8],
        }

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------

    def _fallback_answer(
        self,
        *,
        question: str,
        retrieved_jobs: List[Dict[str, Any]],
        target_job: Dict[str, Any],
    ) -> str:
        top_titles = [j.get("title") for j in retrieved_jobs[:3] if j.get("title")]
        title_text = ", ".join(top_titles) if top_titles else "similar roles in the corpus"

        if target_job:
            return (
                f"I could not produce a structured LLM answer, but the most relevant context "
                f"includes the selected target job and retrieved roles such as {title_text}. "
                f"Try asking a narrower question about skills, fit, or prioritization."
            )

        return (
            f"I could not produce a structured LLM answer, but the most relevant retrieved jobs "
            f"include {title_text}. Try asking a narrower question about requirements, trends, "
            f"or role prioritization."
        )

    def _fallback_sources(
        self,
        *,
        retrieved_jobs: List[Dict[str, Any]],
        target_job: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        if target_job and target_job.get("job_id"):
            out.append(
                {
                    "job_id": str(target_job.get("job_id")),
                    "title": target_job.get("title", ""),
                    "company": target_job.get("company", ""),
                    "reason": "selected target job",
                }
            )

        for row in retrieved_jobs:
            job_id = str(row.get("job_id", "")).strip()
            if not job_id:
                continue
            if any(s["job_id"] == job_id for s in out):
                continue
            out.append(
                {
                    "job_id": job_id,
                    "title": row.get("title", ""),
                    "company": row.get("company", ""),
                    "reason": "retrieved for semantic relevance",
                }
            )
            if len(out) >= 5:
                break

        return out

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

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
