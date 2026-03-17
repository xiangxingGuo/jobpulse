from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal

from src.llm.json_repair import parse_json_object
from src.llm.providers.openai_compat_client import OpenAICompatClient
from src.llm.providers.openai_compat_providers import PROVIDERS
from src.schemas.skill_gap import (
    EvidenceItem,
    GapItem,
    ResumeProfile,
    ResumeSuggestion,
    SkillGapResult,
    StrengthItem,
)
from src.services.job_search_service import JobSearchService
from src.services.resume_service import ResumeService
from src.services.skill_gap_prompt import build_skill_gap_analysis_messages


AnalysisMode = Literal["baseline", "hybrid"]


@dataclass
class SkillGapAnalyzeArtifacts:
    baseline: Dict[str, Any]
    job_context: Dict[str, Any]
    market_context: Dict[str, Any]
    llm: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SkillGapService:
    """
    Hybrid-ready skill-gap analyzer.

    v1:
    - baseline analysis remains the default and always works without LLM
    - hybrid mode overlays LLM reasoning on top of baseline signals
    - fit_score / fit_band / confidence stay deterministic
    """

    def __init__(
        self,
        *,
        resume_service: Optional[ResumeService] = None,
        job_search_service: Optional[JobSearchService] = None,
    ) -> None:
        self.resume_service = resume_service or ResumeService()
        self.job_search_service = job_search_service or JobSearchService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        *,
        resume_text: str,
        job_id: str,
        include_market_context: bool = True,
        market_top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Synchronous baseline-only path.
        Safe default for current API/UI callers.
        """
        ctx = self._build_analysis_context(
            resume_text=resume_text,
            job_id=job_id,
            include_market_context=include_market_context,
            market_top_k=market_top_k,
        )

        result = self._build_baseline_result(
            job_id=job_id,
            resume_profile=ctx["resume_profile"],
            job_detail=ctx["job_detail"],
            baseline=ctx["baseline"],
            market_context=ctx["market_context"],
        )

        artifacts = SkillGapAnalyzeArtifacts(
            baseline=ctx["baseline"],
            job_context=self._job_context(ctx["job_detail"], job_id),
            market_context=ctx["market_context"],
            llm=None,
        )

        return {
            "resume_profile": ctx["resume_profile"].model_dump(),
            "skill_gap": result.model_dump(),
            "artifacts": artifacts.to_dict(),
        }

    async def analyze_async(
        self,
        *,
        resume_text: str,
        job_id: str,
        include_market_context: bool = True,
        market_top_k: int = 5,
        analysis_mode: AnalysisMode = "baseline",
        provider: Literal["openai", "nvidia"] = "openai",
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1400,
        thinking: Literal["auto", "disabled", "enabled"] = "disabled",
    ) -> Dict[str, Any]:
        """
        Async path that supports both baseline and hybrid modes.
        """
        if analysis_mode == "baseline":
            return self.analyze(
                resume_text=resume_text,
                job_id=job_id,
                include_market_context=include_market_context,
                market_top_k=market_top_k,
            )

        ctx = self._build_analysis_context(
            resume_text=resume_text,
            job_id=job_id,
            include_market_context=include_market_context,
            market_top_k=market_top_k,
        )

        baseline_result = self._build_baseline_result(
            job_id=job_id,
            resume_profile=ctx["resume_profile"],
            job_detail=ctx["job_detail"],
            baseline=ctx["baseline"],
            market_context=ctx["market_context"],
        )

        llm_result = await self._analyze_with_llm(
            resume_profile=ctx["resume_profile"],
            resume_text=resume_text,
            job_detail=ctx["job_detail"],
            baseline=ctx["baseline"],
            market_context=ctx["market_context"],
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
        )

        final_result = self._merge_baseline_and_llm(
            baseline_result=baseline_result,
            llm_result=llm_result.get("structured"),
        )

        final_meta = dict(final_result.meta or {})
        final_meta["analysis_mode"] = "hybrid"
        final_meta["llm"] = {
            "provider": llm_result.get("provider"),
            "model": llm_result.get("model"),
            "parse_ok": llm_result.get("parse_ok"),
            "parse_repaired": llm_result.get("parse_repaired"),
            "base_url": llm_result.get("base_url"),
        }
        final_result.meta = final_meta

        artifacts = SkillGapAnalyzeArtifacts(
            baseline=ctx["baseline"],
            job_context=self._job_context(ctx["job_detail"], job_id),
            market_context=ctx["market_context"],
            llm={
                "raw_output": llm_result.get("raw_output", ""),
                "structured": llm_result.get("structured"),
                "parse_ok": llm_result.get("parse_ok"),
                "parse_repaired": llm_result.get("parse_repaired"),
                "usage": llm_result.get("usage") or {},
                "provider": llm_result.get("provider"),
                "model": llm_result.get("model"),
                "base_url": llm_result.get("base_url"),
            },
        )

        return {
            "resume_profile": ctx["resume_profile"].model_dump(),
            "skill_gap": final_result.model_dump(),
            "artifacts": artifacts.to_dict(),
        }

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _build_analysis_context(
        self,
        *,
        resume_text: str,
        job_id: str,
        include_market_context: bool,
        market_top_k: int,
    ) -> Dict[str, Any]:
        if not resume_text or not resume_text.strip():
            raise ValueError("resume_text is empty")

        job_id = str(job_id).strip()
        if not job_id:
            raise ValueError("job_id is empty")

        resume_profile = self.resume_service.parse_profile(resume_text)
        job_detail = self.job_search_service.get_job_by_id(job_id)

        if not job_detail:
            raise RuntimeError(f"No job found with job_id: {job_id}")

        market_context = (
            self.job_search_service.get_market_context_summary(
                job_id=job_id,
                top_k=market_top_k,
            )
            if include_market_context
            else {
                "anchor_job_id": job_id,
                "top_k": 0,
                "similar_jobs": [],
                "companies": [],
                "titles": [],
                "locations": [],
            }
        )

        baseline = self._build_baseline(
            resume_profile=resume_profile,
            job_detail=job_detail,
        )

        return {
            "resume_profile": resume_profile,
            "job_detail": job_detail,
            "market_context": market_context,
            "baseline": baseline,
        }

    def _job_context(self, job_detail: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        return {
            "job_id": job_id,
            "title": job_detail.get("title"),
            "company": job_detail.get("company"),
            "location": job_detail.get("location"),
            "skills": job_detail.get("skills", []),
        }

    # ------------------------------------------------------------------
    # Baseline logic
    # ------------------------------------------------------------------

    def _build_baseline(
        self,
        *,
        resume_profile: ResumeProfile,
        job_detail: Dict[str, Any],
    ) -> Dict[str, Any]:
        resume_skills = {s.lower() for s in resume_profile.explicit_skills}
        inferred_skills = {s.lower() for s in resume_profile.inferred_skills}
        all_resume_skills = sorted(resume_skills | inferred_skills)

        job_skills_raw = job_detail.get("skills", []) or []
        job_skills = sorted(
            {
                str(skill).strip().lower()
                for skill in job_skills_raw
                if str(skill).strip()
            }
        )

        shared = sorted(set(all_resume_skills) & set(job_skills))
        missing = sorted(set(job_skills) - set(all_resume_skills))

        return {
            "resume_skills": all_resume_skills,
            "job_skills": job_skills,
            "shared_skills": shared,
            "missing_skills": missing,
            "shared_count": len(shared),
            "missing_count": len(missing),
            "job_skill_count": len(job_skills),
        }

    def _build_baseline_result(
        self,
        *,
        job_id: str,
        resume_profile: ResumeProfile,
        job_detail: Dict[str, Any],
        baseline: Dict[str, Any],
        market_context: Dict[str, Any],
    ) -> SkillGapResult:
        job_skills = baseline["job_skills"]
        shared = set(baseline["shared_skills"])
        missing = set(baseline["missing_skills"])

        strengths: List[StrengthItem] = []
        gaps: List[GapItem] = []
        transferable_signals: List[StrengthItem] = []

        for skill in sorted(shared):
            strengths.append(
                StrengthItem(
                    skill=skill,
                    support="direct",
                    rationale=f"Resume provides direct evidence for required or relevant skill '{skill}'.",
                    evidence=[
                        EvidenceItem(
                            claim=f"Candidate demonstrates {skill}",
                            source="resume",
                            snippet=self._resume_snippet_for_skill(
                                resume_profile=resume_profile,
                                skill=skill,
                            ),
                            score=1.0,
                        ),
                        EvidenceItem(
                            claim=f"Job mentions {skill}",
                            source="job",
                            snippet=f"Job skill requirement includes '{skill}'.",
                            score=1.0,
                        ),
                    ],
                )
            )

        for skill in sorted(missing):
            category = self._classify_gap_category(skill)
            severity = self._classify_gap_severity(category)

            gaps.append(
                GapItem(
                    skill=skill,
                    category=category,
                    severity=severity,
                    rationale=(
                        f"The job appears to require or value '{skill}', but the current resume "
                        "does not show direct evidence for it."
                    ),
                    evidence=[
                        EvidenceItem(
                            claim=f"Job mentions {skill}",
                            source="job",
                            snippet=f"Job skill requirement includes '{skill}'.",
                            score=1.0,
                        )
                    ],
                    actionable=True,
                )
            )

        transferable_pairs = self._infer_transferable_signals(
            resume_profile=resume_profile,
            missing_skills=sorted(missing),
        )
        for target_skill, supporting_signal in transferable_pairs:
            transferable_signals.append(
                StrengthItem(
                    skill=target_skill,
                    support="transferable",
                    rationale=(
                        f"Resume does not directly list '{target_skill}', but it shows adjacent "
                        f"experience through '{supporting_signal}'."
                    ),
                    evidence=[
                        EvidenceItem(
                            claim=f"Adjacent experience for {target_skill}",
                            source="resume",
                            snippet=f"Resume profile includes transferable signal '{supporting_signal}'.",
                            score=0.7,
                        )
                    ],
                )
            )

        fit_score = self._compute_fit_score(
            shared_count=baseline["shared_count"],
            missing_count=baseline["missing_count"],
            transferable_count=len(transferable_signals),
            total_job_skills=max(1, baseline["job_skill_count"]),
        )
        fit_band = self._fit_band(fit_score)
        confidence = self._confidence(
            resume_profile=resume_profile,
            baseline=baseline,
        )

        summary = self._build_summary(
            job_detail=job_detail,
            fit_score=fit_score,
            fit_band=fit_band,
            strengths=strengths,
            gaps=gaps,
            transferable_signals=transferable_signals,
        )

        resume_suggestions = self._build_resume_suggestions(
            resume_profile=resume_profile,
            gaps=gaps,
            transferable_signals=transferable_signals,
        )

        action_plan_7d = self._build_action_plan_7d(gaps=gaps)
        action_plan_30d = self._build_action_plan_30d(gaps=gaps)

        return SkillGapResult(
            job_id=job_id,
            fit_score=fit_score,
            fit_band=fit_band,
            confidence=confidence,
            strengths=strengths[:8],
            gaps=gaps[:10],
            transferable_signals=transferable_signals[:6],
            resume_suggestions=resume_suggestions[:6],
            action_plan_7d=action_plan_7d,
            action_plan_30d=action_plan_30d,
            summary=summary,
            meta={
                "version": "skill_gap_v2_baseline",
                "analysis_mode": "baseline",
                "job_title": job_detail.get("title"),
                "company": job_detail.get("company"),
                "location": job_detail.get("location"),
                "market_titles": market_context.get("titles", []),
                "market_companies": market_context.get("companies", []),
                "baseline": {
                    "shared_count": baseline["shared_count"],
                    "missing_count": baseline["missing_count"],
                    "job_skill_count": baseline["job_skill_count"],
                },
            },
        )

    # ------------------------------------------------------------------
    # LLM layer
    # ------------------------------------------------------------------

    async def _analyze_with_llm(
        self,
        *,
        resume_profile: ResumeProfile,
        resume_text: str,
        job_detail: Dict[str, Any],
        baseline: Dict[str, Any],
        market_context: Dict[str, Any],
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

        messages = build_skill_gap_analysis_messages(
            resume_profile=resume_profile.model_dump(),
            resume_text=resume_text,
            job_detail=job_detail,
            baseline=baseline,
            market_context=market_context,
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

        if not raw_output.strip():
            raw_output = json.dumps(resp, ensure_ascii=False)

        parsed, repaired_flag, _ = parse_json_object(raw_output)
        parse_ok = isinstance(parsed, dict)

        normalized = self._normalize_llm_output(parsed if parse_ok else None)

        return {
            "structured": normalized,
            "raw_output": raw_output,
            "parse_ok": parse_ok,
            "parse_repaired": repaired_flag,
            "usage": usage,
            "provider": provider,
            "model": model,
            "base_url": client.base_url,
            "thinking": effective_thinking,
        }

    def _normalize_llm_output(self, parsed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(parsed, dict):
            return None

        out: Dict[str, Any] = {
            "strengths": [],
            "gaps": [],
            "transferable_signals": [],
            "resume_suggestions": [],
            "summary": "",
        }

        for raw in parsed.get("strengths", []) or []:
            item = self._safe_strength_item(raw)
            if item is not None:
                out["strengths"].append(item)

        for raw in parsed.get("gaps", []) or []:
            item = self._safe_gap_item(raw)
            if item is not None:
                out["gaps"].append(item)

        for raw in parsed.get("transferable_signals", []) or []:
            item = self._safe_strength_item(raw, force_support="transferable")
            if item is not None:
                out["transferable_signals"].append(item)

        for raw in parsed.get("resume_suggestions", []) or []:
            item = self._safe_resume_suggestion(raw)
            if item is not None:
                out["resume_suggestions"].append(item)

        summary = parsed.get("summary")
        if isinstance(summary, str):
            out["summary"] = summary.strip()

        return out

    def _merge_baseline_and_llm(
        self,
        *,
        baseline_result: SkillGapResult,
        llm_result: Optional[Dict[str, Any]],
    ) -> SkillGapResult:
        if not llm_result:
            fallback = baseline_result.model_copy(deep=True)
            meta = dict(fallback.meta or {})
            meta["llm_merge"] = "fallback_to_baseline"
            fallback.meta = meta
            return fallback

        baseline_strengths = list(baseline_result.strengths)
        llm_strengths = list(llm_result.get("strengths") or [])
        merged_strengths = self._dedupe_strengths(baseline_strengths + llm_strengths)

        baseline_transferable = list(baseline_result.transferable_signals)
        llm_transferable = list(llm_result.get("transferable_signals") or [])
        merged_transferable = self._dedupe_strengths(
            baseline_transferable + llm_transferable
        )

        llm_gaps = list(llm_result.get("gaps") or [])
        merged_gaps = self._dedupe_gaps(llm_gaps or list(baseline_result.gaps))

        llm_suggestions = list(llm_result.get("resume_suggestions") or [])
        merged_suggestions = self._dedupe_suggestions(
            llm_suggestions or list(baseline_result.resume_suggestions)
        )

        summary = (llm_result.get("summary") or "").strip() or baseline_result.summary

        merged = baseline_result.model_copy(deep=True)
        merged.strengths = merged_strengths[:8]
        merged.transferable_signals = merged_transferable[:6]
        merged.gaps = merged_gaps[:10]
        merged.resume_suggestions = merged_suggestions[:6]
        merged.summary = summary

        meta = dict(merged.meta or {})
        meta["llm_merge"] = "hybrid_overlay"
        meta["llm_strength_count"] = len(llm_strengths)
        meta["llm_gap_count"] = len(llm_gaps)
        merged.meta = meta
        return merged

    # ------------------------------------------------------------------
    # Safe item parsing
    # ------------------------------------------------------------------

    def _safe_strength_item(
        self,
        raw: Any,
        *,
        force_support: Optional[str] = None,
    ) -> Optional[StrengthItem]:
        if not isinstance(raw, dict):
            return None

        try:
            evidence = [self._safe_evidence_item(ev) for ev in (raw.get("evidence") or [])]
            evidence = [ev for ev in evidence if ev is not None]

            return StrengthItem(
                skill=str(raw.get("skill", "")).strip(),
                support=force_support or str(raw.get("support", "direct")).strip(),
                rationale=str(raw.get("rationale", "")).strip(),
                evidence=evidence,
            )
        except Exception:
            return None

    def _safe_gap_item(self, raw: Any) -> Optional[GapItem]:
        if not isinstance(raw, dict):
            return None

        try:
            evidence = [self._safe_evidence_item(ev) for ev in (raw.get("evidence") or [])]
            evidence = [ev for ev in evidence if ev is not None]

            return GapItem(
                skill=str(raw.get("skill", "")).strip(),
                category=str(raw.get("category", "ambiguous")).strip(),
                severity=str(raw.get("severity", "medium")).strip(),
                rationale=str(raw.get("rationale", "")).strip(),
                evidence=evidence,
                actionable=bool(raw.get("actionable", True)),
            )
        except Exception:
            return None

    def _safe_resume_suggestion(self, raw: Any) -> Optional[ResumeSuggestion]:
        if not isinstance(raw, dict):
            return None

        try:
            before = raw.get("before")
            after = raw.get("after")
            return ResumeSuggestion(
                type=str(raw.get("type", "clarify")).strip(),
                target=str(raw.get("target", "")).strip(),
                before=str(before).strip() if isinstance(before, str) else None,
                after=str(after).strip() if isinstance(after, str) else None,
                rationale=str(raw.get("rationale", "")).strip(),
            )
        except Exception:
            return None

    def _safe_evidence_item(self, raw: Any) -> Optional[EvidenceItem]:
        if not isinstance(raw, dict):
            return None

        try:
            score = raw.get("score")
            score_value = None
            if score is not None:
                try:
                    score_value = max(0.0, min(1.0, float(score)))
                except Exception:
                    score_value = None

            return EvidenceItem(
                claim=str(raw.get("claim", "")).strip() or "Support for analysis item",
                source=str(raw.get("source", "resume")).strip(),
                snippet=str(raw.get("snippet", "")).strip() or "No snippet provided.",
                score=score_value,
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Utility merge helpers
    # ------------------------------------------------------------------

    def _dedupe_strengths(self, items: List[StrengthItem]) -> List[StrengthItem]:
        out: List[StrengthItem] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            key = (item.skill.lower().strip(), item.support.lower().strip())
            if not key[0] or key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _dedupe_gaps(self, items: List[GapItem]) -> List[GapItem]:
        out: List[GapItem] = []
        seen: set[str] = set()
        for item in items:
            key = item.skill.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _dedupe_suggestions(self, items: List[ResumeSuggestion]) -> List[ResumeSuggestion]:
        out: List[ResumeSuggestion] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            key = (item.type.lower().strip(), item.target.lower().strip())
            if not key[1] or key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    # ------------------------------------------------------------------
    # Existing heuristics
    # ------------------------------------------------------------------

    def _classify_gap_category(self, skill: str) -> str:
        must_have = {
            "python",
            "sql",
            "pytorch",
            "tensorflow",
            "docker",
            "aws",
            "gcp",
            "azure",
            "llm",
            "rag",
            "transformers",
            "fastapi",
        }
        return "must_have" if skill in must_have else "nice_to_have"

    def _classify_gap_severity(self, category: str) -> str:
        if category == "must_have":
            return "high"
        if category == "ambiguous":
            return "low"
        return "medium"

    def _compute_fit_score(
        self,
        *,
        shared_count: int,
        missing_count: int,
        transferable_count: int,
        total_job_skills: int,
    ) -> int:
        direct_ratio = shared_count / max(1, total_job_skills)
        missing_ratio = missing_count / max(1, total_job_skills)
        transferable_ratio = transferable_count / max(1, total_job_skills)

        score = 55
        score += int(35 * direct_ratio)
        score += int(10 * transferable_ratio)
        score -= int(20 * missing_ratio)

        return max(0, min(100, score))

    def _fit_band(self, fit_score: int) -> str:
        if fit_score >= 75:
            return "strong"
        if fit_score >= 55:
            return "moderate"
        return "stretch"

    def _confidence(
        self,
        *,
        resume_profile: ResumeProfile,
        baseline: Dict[str, Any],
    ) -> float:
        signal_count = (
            len(resume_profile.explicit_skills)
            + len(resume_profile.ml_domains)
            + len(resume_profile.deployment_signals)
        )
        if baseline["job_skill_count"] == 0:
            return 0.45
        conf = 0.45 + min(0.45, signal_count * 0.03)
        return round(min(0.9, conf), 2)

    def _build_summary(
        self,
        *,
        job_detail: Dict[str, Any],
        fit_score: int,
        fit_band: str,
        strengths: List[StrengthItem],
        gaps: List[GapItem],
        transferable_signals: List[StrengthItem],
    ) -> str:
        title = job_detail.get("title") or "this role"
        company = job_detail.get("company") or "the company"

        strength_text = ", ".join(s.skill for s in strengths[:3]) or "limited directly matched skills"
        gap_text = ", ".join(g.skill for g in gaps[:3]) or "no major visible gaps"
        transfer_text = ", ".join(t.skill for t in transferable_signals[:2])

        if transfer_text:
            return (
                f"Estimated fit for {title} at {company}: {fit_band} ({fit_score}/100). "
                f"Direct strengths include {strength_text}. "
                f"Main visible gaps include {gap_text}. "
                f"There are also transferable signals that may partially support {transfer_text}."
            )

        return (
            f"Estimated fit for {title} at {company}: {fit_band} ({fit_score}/100). "
            f"Direct strengths include {strength_text}. "
            f"Main visible gaps include {gap_text}."
        )

    def _build_resume_suggestions(
        self,
        *,
        resume_profile: ResumeProfile,
        gaps: List[GapItem],
        transferable_signals: List[StrengthItem],
    ) -> List[ResumeSuggestion]:
        suggestions: List[ResumeSuggestion] = []

        if resume_profile.deployment_signals:
            suggestions.append(
                ResumeSuggestion(
                    type="add_evidence",
                    target="project bullets",
                    rationale=(
                        "Make deployment and productionization details more explicit, since those signals "
                        "often map well to ML/AI engineering roles."
                    ),
                    before=None,
                    after=None,
                )
            )

        for item in transferable_signals[:2]:
            suggestions.append(
                ResumeSuggestion(
                    type="clarify",
                    target=f"experience related to {item.skill}",
                    rationale=(
                        f"Explicitly connect existing experience to '{item.skill}' so recruiters can see "
                        "the transferable alignment faster."
                    ),
                    before=None,
                    after=None,
                )
            )

        for gap in gaps[:2]:
            suggestions.append(
                ResumeSuggestion(
                    type="rewrite",
                    target=f"skills/experience related to {gap.skill}",
                    rationale=(
                        f"If you have any real exposure to '{gap.skill}', make that evidence clearer in the resume. "
                        "Do not add experience that you do not have."
                    ),
                    before=None,
                    after=None,
                )
            )

        return suggestions

    def _build_action_plan_7d(self, *, gaps: List[GapItem]) -> List[str]:
        top_gaps = [g.skill for g in gaps[:3]]
        if not top_gaps:
            return [
                "Refine resume bullets to emphasize direct evidence for your strongest matching skills.",
                "Prepare one concise project story that demonstrates end-to-end ML or AI engineering work.",
            ]

        return [
            f"Review the fundamentals of {top_gaps[0]} and connect them to your existing projects.",
            "Update resume bullets to show measurable engineering outcomes and concrete technical ownership.",
            "Prepare interview-ready explanations for one matching project and one gap area.",
        ]

    def _build_action_plan_30d(self, *, gaps: List[GapItem]) -> List[str]:
        top_gaps = [g.skill for g in gaps[:4]]
        if not top_gaps:
            return [
                "Apply broadly to roles with strong overlap and tailor the resume per target role family.",
                "Build one polished portfolio write-up showing system design, deployment, and evaluation tradeoffs.",
            ]

        joined = ", ".join(top_gaps[:3])
        return [
            f"Build or extend one project that gives credible evidence for missing areas such as {joined}.",
            "Create one targeted resume version for ML/AI engineering roles with stronger skills-to-project mapping.",
            "Track repeated market requirements across similar jobs and prioritize the highest-frequency gaps.",
        ]

    def _infer_transferable_signals(
        self,
        *,
        resume_profile: ResumeProfile,
        missing_skills: List[str],
    ) -> List[tuple[str, str]]:
        signals: List[tuple[str, str]] = []

        deployment_set = {s.lower() for s in resume_profile.deployment_signals}
        explicit_set = {s.lower() for s in resume_profile.explicit_skills}
        domain_set = {s.lower() for s in resume_profile.ml_domains}

        if "kubernetes" in missing_skills and "docker" in explicit_set:
            signals.append(("kubernetes", "docker"))

        if "mlops" in missing_skills and (
            deployment_set.intersection({"docker", "aws", "gcp", "azure", "fastapi"})
        ):
            signals.append(("mlops", sorted(deployment_set)[0]))

        if "rag" in missing_skills and (
            "retrieval" in domain_set or "embedding" in domain_set
        ):
            signals.append(("rag", "retrieval"))

        if "transformers" in missing_skills and "huggingface" in explicit_set:
            signals.append(("transformers", "huggingface"))

        if "aws" in missing_skills and "docker" in explicit_set:
            signals.append(("aws", "docker"))

        return signals

    def _resume_snippet_for_skill(
        self,
        *,
        resume_profile: ResumeProfile,
        skill: str,
    ) -> str:
        if skill in {s.lower() for s in resume_profile.explicit_skills}:
            return f"Resume profile explicitly includes '{skill}'."
        if skill in {s.lower() for s in resume_profile.inferred_skills}:
            return f"Resume profile inferred '{skill}' from prior experience."
        return f"Resume shows supporting evidence for '{skill}'."

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