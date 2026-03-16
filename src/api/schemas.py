from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.schemas.skill_gap import ResumeProfile, SkillGapResult

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)


class SearchResult(BaseModel):
    job_id: str
    title: str | None = None
    company: str | None = None
    location: str | None = None
    url: str | None = None
    score: float


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[SearchResult]


class SimilarResponse(BaseModel):
    job_id: str
    top_k: int
    results: list[SearchResult]


class HealthResponse(BaseModel):
    status: str
    index_ready: bool


class JobDetailResponse(BaseModel):
    job_id: str
    url: str | None = None
    title: str | None = None
    company: str | None = None
    location_text: str | None = None
    description: str | None = None
    skills: list[str] = []
    scrape_status: str | None = None
    scrape_error: str | None = None
    content_hash: str | None = None
    last_seen_at_utc: str | None = None
    structured: dict[str, Any] | None = None
    structured_meta: dict[str, Any] | None = None


class RecentRunsResponse(BaseModel):
    runs: list[dict[str, Any]]

class MetricsSummaryResponse(BaseModel):
    runs_considered: int
    scrape: dict[str, Any]
    counts: dict[str, int]
    latest_runs: list[dict[str, Any]]

class AnalyticsItem(BaseModel):
    name: str
    count: int


class AnalyticsSummaryResponse(BaseModel):
    total_jobs: int
    top_skills: list[AnalyticsItem]
    top_companies: list[AnalyticsItem]
    top_locations: list[AnalyticsItem]
    top_titles: list[AnalyticsItem]

class ResumeMatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=20)
    top_k: int = Field(default=5, ge=1, le=20)


class ResumeProfileResponse(BaseModel):
    skills: list[str]
    text_preview: str


class ResumeMatchItem(BaseModel):
    job_id: str
    title: str | None = None
    company: str | None = None
    location: str | None = None
    url: str | None = None
    semantic_score: float
    shared_skills: list[str]
    missing_skills: list[str]
    match_reasons: list[str]


class ResumeMatchResponse(BaseModel):
    resume_profile: ResumeProfileResponse
    matches: list[ResumeMatchItem]

class ResumeParseResponse(BaseModel):
    filename: str
    text_preview: str
    chars: int
    resume_text: str

class ResumeAnalyzeFitRequest(BaseModel):
    job_id: str = Field(..., min_length=1)
    resume_text: str = Field(..., min_length=20)
    include_market_context: bool = True
    market_top_k: int = Field(default=5, ge=1, le=20)
    include_report: bool = True


class ResumeAnalyzeFitResponse(BaseModel):
    resume_profile: ResumeProfile
    skill_gap: SkillGapResult
    report_md: str = ""
    meta: dict[str, Any] = {}