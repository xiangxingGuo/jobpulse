from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from src.services.resume_service import ResumeService
from src.services.job_search_service import JobSearchService
from src.schemas.skill_gap import (
    EvidenceItem,
    GapItem,
    ResumeProfile,
    ResumeSuggestion,
    SkillGapResult,
    StrengthItem,
)


@dataclass
class SkillGapAnalyzeArtifacts:
    baseline: Dict[str, Any]
    job_context: Dict[str, Any]
    market_context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SkillGapService:
    """
    v1 deterministic baseline skill-gap analyzer.

    Design goals:
    - no LLM dependency required
    - grounded in resume profile + job detail + semantic retrieval
    - emits structured SkillGapResult
    - easy to upgrade later with LLM enrichment
    """

    def __init__(
        self,
        *,
        resume_service: Optional[ResumeService] = None,
        job_search_service: Optional[JobSearchService] = None,
    ) -> None:
        self.resume_service = resume_service or ResumeService()
        self.job_search_service = job_search_service or JobSearchService()

    def analyze(
        self,
        *,
        resume_text: str,
        job_id: str,
        include_market_context: bool = True,
        market_top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Main entrypoint.

        Returns a dict payload for now so downstream callers can use:
        - resume_profile
        - skill_gap
        - artifacts/debug context
        """
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

        result = self._build_skill_gap_result(
            job_id=job_id,
            resume_profile=resume_profile,
            job_detail=job_detail,
            baseline=baseline,
            market_context=market_context,
        )

        artifacts = SkillGapAnalyzeArtifacts(
            baseline=baseline,
            job_context={
                "job_id": job_id,
                "title": job_detail.get("title"),
                "company": job_detail.get("company"),
                "location": job_detail.get("location"),
                "skills": job_detail.get("skills", []),
            },
            market_context=market_context,
        )

        return {
            "resume_profile": resume_profile.model_dump(),
            "skill_gap": result.model_dump(),
            "artifacts": artifacts.to_dict(),
        }

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

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

    def _build_skill_gap_result(
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
                "version": "skill_gap_v1_baseline",
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