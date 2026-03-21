from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File

from src.api.schemas import (
    AnalyticsSummaryResponse,
    HealthResponse,
    JobDetailResponse,
    JobMarketChatRequest,
    JobMarketChatResponse,
    MetricsSummaryResponse,
    RecentRunsResponse,
    ResumeAnalyzeFitRequest,
    ResumeAnalyzeFitResponse,
    ResumeMatchRequest,
    ResumeMatchResponse,
    ResumeParseResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SimilarResponse,
    LexSkillGapRequest,
    LexSkillGapResponse,
)
from src.db import (
    fetch_analytics_summary,
    fetch_job_detail,
    fetch_metrics_summary,
    fetch_recent_scrape_runs,
)
from src.observability.artifact_writer import (
    JobMarketChatArtifactWriter,
    SkillGapArtifactWriter,
)
from src.resume.parse import extract_resume_text
from src.retrieval.resume_match import match_resume_to_jobs
from src.services.job_market_chat_service import JobMarketChatService
from src.services.job_search_service import JobSearchService
from src.services.report_service import ReportService
from src.services.skill_gap_service import SkillGapService

import os
import uuid

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _search_service, _skill_gap_service, _report_service, _job_market_chat_service
    try:
        _search_service = JobSearchService(index_dir=INDEX_DIR)
        _skill_gap_service = SkillGapService(job_search_service=_search_service)
        _report_service = ReportService()
        _job_market_chat_service = JobMarketChatService(
            job_search_service=_search_service,
            skill_gap_service=_skill_gap_service,
        )
        print("Search / skill-gap / report / chat services preloaded.")
    except Exception as e:
        print(f"Failed to preload services: {e}")
        _search_service = None
        _skill_gap_service = None
        _report_service = None
        _job_market_chat_service = None
    yield

INDEX_DIR = Path("data/vectors")
SKILL_GAP_ARTIFACTS_DIR = Path(os.getenv("ARTIFACT_DIR", "data/artifacts")) / "skill_gap"
JOB_MARKET_CHAT_ARTIFACTS_DIR = Path(os.getenv("ARTIFACT_DIR", "data/artifacts")) / "job_market_chat"
_search_service: JobSearchService | None = None
_skill_gap_service: SkillGapService | None = None
_report_service: ReportService | None = None
_job_market_chat_service: JobMarketChatService | None = None

app = FastAPI(
    title="JobPulse API",
    version="0.1.0",
    description="Semantic job search and retrieval API for JobPulse",
    lifespan=lifespan,
)


def get_search_service() -> JobSearchService:
    global _search_service
    if _search_service is None:
        _search_service = JobSearchService(index_dir=INDEX_DIR)
    return _search_service


def get_skill_gap_service() -> SkillGapService:
    global _skill_gap_service
    if _skill_gap_service is None:
        _skill_gap_service = SkillGapService(job_search_service=get_search_service())
    return _skill_gap_service


def get_report_service() -> ReportService:
    global _report_service
    if _report_service is None:
        _report_service = ReportService()
    return _report_service

def get_job_market_chat_service() -> JobMarketChatService:
    global _job_market_chat_service
    if _job_market_chat_service is None:
        _job_market_chat_service = JobMarketChatService(
            job_search_service=get_search_service(),
            skill_gap_service=get_skill_gap_service(),
        )
    return _job_market_chat_service

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    index_ready = (INDEX_DIR / "jobs.faiss").exists() and (INDEX_DIR / "meta.jsonl").exists()
    return HealthResponse(status="ok", index_ready=index_ready)


@app.post("/jobs/search", response_model=SearchResponse)
def search_jobs(req: SearchRequest) -> SearchResponse:
    svc = get_search_service()
    results = svc.search_jobs(req.query, top_k=req.top_k)

    items = [
        SearchResult(
            job_id=r.job_id,
            title=r.title,
            company=r.company,
            location=r.location,
            url=r.url,
            score=float(r.score),
        )
        for r in results
    ]
    return SearchResponse(query=req.query, top_k=req.top_k, results=items)


@app.get("/jobs/{job_id}", response_model=JobDetailResponse)
def get_job(job_id: str) -> JobDetailResponse:
    row = fetch_job_detail(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    return JobDetailResponse(**row)


@app.get("/jobs/{job_id}/similar", response_model=SimilarResponse)
def similar_jobs(job_id: str, top_k: int = 10) -> SimilarResponse:
    svc = get_search_service()
    try:
        results = svc.similar_jobs_for_job(job_id, top_k=top_k)
    except ValueError:
        raise HTTPException(status_code=404, detail="job not found in vector index")

    items = [
        SearchResult(
            job_id=r.job_id,
            title=r.title,
            company=r.company,
            location=r.location,
            url=r.url,
            score=float(r.score),
        )
        for r in results
    ]
    return SimilarResponse(job_id=job_id, top_k=top_k, results=items)


@app.get("/runs/recent", response_model=RecentRunsResponse)
def recent_runs(limit: int = 10) -> RecentRunsResponse:
    rows = fetch_recent_scrape_runs(limit=limit)
    return RecentRunsResponse(runs=rows)

@app.get("/metrics/summary", response_model=MetricsSummaryResponse)
def metrics_summary(limit: int = 20) -> MetricsSummaryResponse:
    row = fetch_metrics_summary(limit=limit)
    return MetricsSummaryResponse(**row)

@app.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
def analytics_summary(limit: int = 10) -> AnalyticsSummaryResponse:
    row = fetch_analytics_summary(limit=limit)
    return AnalyticsSummaryResponse(**row)

@app.post("/resume/match", response_model=ResumeMatchResponse)
def resume_match(req: ResumeMatchRequest) -> ResumeMatchResponse:
    row = match_resume_to_jobs(req.resume_text, top_k=req.top_k)
    return ResumeMatchResponse(**row)


@app.post("/resume/analyze-fit", response_model=ResumeAnalyzeFitResponse)
async def resume_analyze_fit(req: ResumeAnalyzeFitRequest) -> ResumeAnalyzeFitResponse:
    skill_gap_svc = get_skill_gap_service()

    try:
        if req.analysis_mode == "hybrid":
            out = await skill_gap_svc.analyze_async(
                resume_text=req.resume_text,
                job_id=req.job_id,
                include_market_context=req.include_market_context,
                market_top_k=req.market_top_k,
                analysis_mode="hybrid",
                provider=req.provider,
                model=req.model,
                temperature=0.2,
                max_tokens=1400,
            )
        else:
            out = skill_gap_svc.analyze(
                resume_text=req.resume_text,
                job_id=req.job_id,
                include_market_context=req.include_market_context,
                market_top_k=req.market_top_k,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"skill gap analysis failed: {e}")

    report_md = ""
    report_meta: dict[str, object] = {}

    if req.include_report:
        report_svc = get_report_service()
        search_svc = get_search_service()
        job = search_svc.get_job_by_id(req.job_id) or {}

        try:
            report = await report_svc.generate_skill_gap_report(
                job_id=req.job_id,
                structured=job,
                qc={
                    "status": "hybrid" if req.analysis_mode == "hybrid" else "baseline_only",
                    "ok": True,
                    "reasons": [],
                },
                resume_profile=out["resume_profile"],
                skill_gap=out["skill_gap"],
                resume_text=req.resume_text,
                market_context=(out.get("artifacts") or {}).get("market_context", {}),
                provider=req.provider,
                model=req.model,
            )
            report_md = report.report_md
            report_meta = {
                "report_meta": report.meta,
                "report_usage": report.usage or {},
            }
        except Exception as e:
            report_md = (
                "# Skill Gap Report\n\n"
                "Structured analysis succeeded, but markdown report generation failed.\n\n"
                f"- error: {e}"
            )
            report_meta = {
                "report_error": str(e),
            }

    artifact_meta: dict[str, object] = {}

    try:
        artifact_run_id = uuid.uuid4().hex[:10]
        artifact_writer = SkillGapArtifactWriter(SKILL_GAP_ARTIFACTS_DIR)
        search_svc = get_search_service()
        structured_job = search_svc.get_job_by_id(req.job_id) or {}

        artifact_dir = artifact_writer.write(
            run_id=artifact_run_id,
            job_id=req.job_id,
            resume_profile=out["resume_profile"],
            skill_gap=out["skill_gap"],
            report_md=report_md,
            meta={
                "job_id": req.job_id,
                "include_market_context": req.include_market_context,
                "market_top_k": req.market_top_k,
                "analysis_mode": req.analysis_mode,
                "provider": req.provider,
                "model": req.model,
                **report_meta,
            },
            artifacts=out.get("artifacts") or {},
            resume_text=req.resume_text,
            structured_job=structured_job,
        )

        artifact_meta = {
            "artifact_run_id": artifact_run_id,
            "artifact_dir": str(artifact_dir),
        }
    except Exception as e:
        artifact_meta = {
            "artifact_error": str(e),
        }

    return ResumeAnalyzeFitResponse(
        resume_profile=out["resume_profile"],
        skill_gap=out["skill_gap"],
        report_md=report_md,
        meta={
            "job_id": req.job_id,
            "include_market_context": req.include_market_context,
            "market_top_k": req.market_top_k,
            "analysis_mode": req.analysis_mode,
            "provider": req.provider,
            "model": req.model,
            **report_meta,
            **artifact_meta,
        },
    )



@app.post("/resume/parse", response_model=ResumeParseResponse)
async def resume_parse(file: UploadFile = File(...)) -> ResumeParseResponse:
    data = await file.read()

    try:
        text = extract_resume_text(file.filename or "resume.txt", data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to parse resume: {e}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="parsed resume text is empty")

    return ResumeParseResponse(
        filename=file.filename or "resume.txt",
        text_preview=" ".join(text.split())[:300],
        chars=len(text),
        resume_text=text,
    )

@app.post("/chat/job-market", response_model=JobMarketChatResponse)
async def job_market_chat(req: JobMarketChatRequest) -> JobMarketChatResponse:
    svc = get_job_market_chat_service()

    try:
        out = await svc.chat(
            question=req.question,
            top_k=req.top_k,
            resume_text=req.resume_text,
            job_id=req.job_id,
            provider=req.provider,
            model=req.model,
            temperature=0.2,
            max_tokens=1200,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"job market chat failed: {e}")

    artifact_meta: dict[str, object] = {}

    try:
        artifact_run_id = uuid.uuid4().hex[:10]
        artifact_writer = JobMarketChatArtifactWriter(JOB_MARKET_CHAT_ARTIFACTS_DIR)

        artifact_dir = artifact_writer.write(
            run_id=artifact_run_id,
            question=req.question,
            answer=out.get("answer", ""),
            sources=out.get("sources", []),
            meta={
                **(out.get("meta") or {}),
                "provider": req.provider,
                "model": req.model,
            },
            artifacts=out.get("artifacts") or {},
            resume_text=req.resume_text,
            job_id=req.job_id,
        )

        artifact_meta = {
            "artifact_run_id": artifact_run_id,
            "artifact_dir": str(artifact_dir),
        }
    except Exception as e:
        artifact_meta = {
            "artifact_error": str(e),
        }

    return JobMarketChatResponse(
        answer=out.get("answer", ""),
        sources=out.get("sources", []),
        meta={
            **(out.get("meta") or {}),
            **artifact_meta,
        },
    )

@app.post("/lex/analyze-skill-gap", response_model=LexSkillGapResponse)
async def lex_analyze_skill_gap(req: LexSkillGapRequest) -> LexSkillGapResponse:
    svc = get_job_market_chat_service()

    background_text = (req.candidate_background or "").strip()
    resume_text = (req.resume_text or "").strip()

    if not background_text and not resume_text:
        raise HTTPException(
            status_code=400,
            detail="Either candidate_background or resume_text must be provided."
        )

    question_parts = [
        f"I am targeting a {req.target_role} role.",
        f"My experience level is {req.experience_level or 'unspecified'}.",
    ]

    if background_text:
        question_parts.append(f"My background is: {background_text}.")

    if resume_text:
        question_parts.append(
            "Use my uploaded resume as additional grounding for the analysis."
        )

    question_parts.append(
        "Based on the current job market, what skills am I likely missing, "
        "what are the biggest gaps, and what should I learn next? "
        "Keep the answer concise and practical."
    )

    question = " ".join(question_parts)

    try:
        out = await svc.chat(
            question=question,
            top_k=req.top_k,
            resume_text=resume_text or background_text or None,
            job_id=None,
            provider=req.provider,
            model=req.model,
            temperature=0.2,
            max_tokens=700,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"lex skill gap analysis failed: {e}")

    return LexSkillGapResponse(
        answer=out.get("answer", ""),
        sources=out.get("sources", []),
        meta=out.get("meta", {}) or {},
    )