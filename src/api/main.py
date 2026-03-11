from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    HealthResponse,
    JobDetailResponse,
    RecentRunsResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SimilarResponse,
)
from src.db import fetch_job_detail, fetch_recent_scrape_runs
from src.retrieval.search import JobSearchService
from src.api.schemas import MetricsSummaryResponse
from src.db import fetch_job_detail, fetch_recent_scrape_runs, fetch_metrics_summary
from src.api.schemas import AnalyticsSummaryResponse
from src.db import fetch_analytics_summary
from src.api.schemas import ResumeMatchRequest, ResumeMatchResponse
from src.retrieval.resume_match import match_resume_to_jobs
from fastapi import FastAPI, HTTPException, UploadFile, File
from src.api.schemas import ResumeParseResponse
from src.resume.parse import extract_resume_text



INDEX_DIR = Path("data/vectors")

app = FastAPI(
    title="JobPulse API",
    version="0.1.0",
    description="Semantic job search and retrieval API for JobPulse",
)

_search_service: JobSearchService | None = None


def get_search_service() -> JobSearchService:
    global _search_service
    if _search_service is None:
        _search_service = JobSearchService(index_dir=INDEX_DIR)
    return _search_service


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
            job_id=str(r.get("job_id")),
            title=r.get("title"),
            company=r.get("company"),
            location=r.get("location"),
            url=r.get("url"),
            score=float(r.get("score", 0.0)),
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
        results = svc.similar_jobs(job_id, top_k=top_k)
    except ValueError:
        raise HTTPException(status_code=404, detail="job not found in vector index")

    items = [
        SearchResult(
            job_id=str(r.get("job_id")),
            title=r.get("title"),
            company=r.get("company"),
            location=r.get("location"),
            url=r.get("url"),
            score=float(r.get("score", 0.0)),
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
