from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict


FitBand = Literal["strong", "moderate", "stretch"]
SupportType = Literal["direct", "transferable"]
GapCategory = Literal["must_have", "nice_to_have", "ambiguous"]
GapSeverity = Literal["high", "medium", "low"]
SuggestionType = Literal["rewrite", "add_evidence", "reorder", "clarify"]
EvidenceSource = Literal["resume", "job", "market", "baseline"]


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(..., description="The claim this evidence supports.")
    source: EvidenceSource = Field(..., description="Where the evidence comes from.")
    snippet: str = Field(..., min_length=1, description="Short supporting text span.")
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional normalized relevance/confidence score.",
    )


class StrengthItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str = Field(..., min_length=1)
    support: SupportType = Field(..., description="Direct match or transferable signal.")
    rationale: str = Field(..., min_length=1)
    evidence: list[EvidenceItem] = Field(default_factory=list)


class GapItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str = Field(..., min_length=1)
    category: GapCategory = Field(...)
    severity: GapSeverity = Field(...)
    rationale: str = Field(..., min_length=1)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    actionable: bool = Field(
        default=True,
        description="Whether the gap can be realistically addressed with an action plan.",
    )


class ResumeSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: SuggestionType = Field(...)
    target: str = Field(
        ...,
        min_length=1,
        description="What part of the resume/project this suggestion targets.",
    )
    before: str | None = Field(
        default=None,
        description="Optional original wording.",
    )
    after: str | None = Field(
        default=None,
        description="Optional improved wording suggestion.",
    )
    rationale: str = Field(..., min_length=1)


class ResumeProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explicit_skills: list[str] = Field(default_factory=list)
    inferred_skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    ml_domains: list[str] = Field(default_factory=list)
    deployment_signals: list[str] = Field(default_factory=list)
    research_signals: list[str] = Field(default_factory=list)
    projects: list[dict[str, Any]] = Field(default_factory=list)
    evidence_spans: list[dict[str, Any]] = Field(default_factory=list)
    text_preview: str = Field(default="")


class SkillGapResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(..., min_length=1)

    fit_score: int = Field(..., ge=0, le=100)
    fit_band: FitBand = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)

    strengths: list[StrengthItem] = Field(default_factory=list)
    gaps: list[GapItem] = Field(default_factory=list)
    transferable_signals: list[StrengthItem] = Field(default_factory=list)

    resume_suggestions: list[ResumeSuggestion] = Field(default_factory=list)
    action_plan_7d: list[str] = Field(default_factory=list)
    action_plan_30d: list[str] = Field(default_factory=list)

    summary: str = Field(..., min_length=1)
    meta: dict[str, Any] = Field(default_factory=dict)


class SkillGapAnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resume_profile: ResumeProfile
    skill_gap: SkillGapResult
    report_md: str = Field(default="")
    meta: dict[str, Any] = Field(default_factory=dict)