from __future__ import annotations

import os
import json
import time
import math
import random
import hashlib
import asyncio
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List
from collections import Counter, defaultdict

from playwright.async_api import async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright_stealth import Stealth

from src.config import ScrapeConfig
from src.scrape.list import collect_job_links
from src.scrape.detail import parse_job_detail, to_dict
from src.extract import extract_skills

from src import db
import argparse

# ----------------------------
# Utils
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()

def pctl(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1))
    return xs[k]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def emit_log(cfg: ScrapeConfig, level: str, event: str, **fields: Any) -> None:
    payload = {"ts_utc": utc_now_iso(), "level": level.upper(), "event": event, "run_id": cfg.run_id, **fields}
    if cfg.log_json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        msg = " ".join([f"{k}={v}" for k, v in payload.items()])
        print(msg)

def compute_backoff(cfg: ScrapeConfig, attempt: int) -> float:
    base = cfg.backoff_base_sec * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0, 0.25 * base)
    return min(cfg.backoff_max_sec, base + jitter)

def is_recoverable_exception(e: Exception) -> bool:
    if isinstance(e, PlaywrightTimeoutError):
        return True
    msg = str(e).lower()
    return ("timeout" in msg) or ("navigation" in msg and "failed" in msg)

async def polite_sleep(cfg: ScrapeConfig) -> None:
    a, b = cfg.sleep_range_sec
    await asyncio.sleep(random.uniform(a, b))
    if random.random() < cfg.extra_pause_prob:
        c, d = cfg.extra_pause_range_sec
        await asyncio.sleep(random.uniform(c, d))

def gate_job(cfg: ScrapeConfig, job: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    url = (job.get("url") or "").lower()
    desc = (job.get("description") or "").lower()

    if cfg.require_jobs_path and "/jobs/" not in url:
        return False, "gate_not_job_url"

    for marker in cfg.bad_markers:
        if marker.lower() in desc:
            return False, "gate_ui_chrome_marker"

    if len(desc) < cfg.min_description_len:
        return False, "gate_desc_too_short"

    return True, None

def artifact_paths(cfg: ScrapeConfig) -> Dict[str, Path]:
    base = cfg.artifact_dir / cfg.run_id
    return {
        "base": base,
        "bad_samples": base / "bad_samples",
        "fail_samples": base / "fail_samples",
        "run_summary": base / "run_summary.json",
        "config": base / "config.json",
    }

def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def ms(sec: float) -> int:
    return int(sec * 1000)


class RunMetrics:
    def __init__(self) -> None:
        self.counts = Counter()
        self.fail_reasons = Counter()
        self.gate_reasons = Counter()
        self.stage_durations_sec = defaultdict(list)
        self.per_job_total_sec = []
        self.desc_len = []
        self.skills_per_job = []
        self.company_present = 0
        self.location_present = 0
        self.seen_jobs_for_fields = set()
        self.seen_jobs_for_desc = set()
        self.seen_jobs_for_skills = set()

    def inc(self, key: str, n: int = 1) -> None:
        self.counts[key] += n

    def fail(self, reason: str) -> None:
        self.counts["jobs_failed"] += 1
        self.fail_reasons[reason] += 1

    def gate(self, reason: str) -> None:
        self.counts["jobs_gated"] += 1
        self.gate_reasons[reason] += 1

    def record_stage(self, stage: str, sec: float) -> None:
        self.stage_durations_sec[stage].append(sec)

    def record_job_total(self, sec: float) -> None:
        self.per_job_total_sec.append(sec)

    def record_desc_len(self, job_id: str, desc_len: int) -> None:
        if job_id in self.seen_jobs_for_desc:
            return
        self.seen_jobs_for_desc.add(job_id)
        self.desc_len.append(desc_len)

    def record_field_presence(self, job_id: str, has_company: bool, has_location: bool) -> None:
        if job_id in self.seen_jobs_for_fields:
            return
        self.seen_jobs_for_fields.add(job_id)
        if has_company:
            self.company_present += 1
        if has_location:
            self.location_present += 1

    def record_skills_n(self, job_id: str, skills_n: int) -> None:
        if job_id in self.seen_jobs_for_skills:
            return
        self.seen_jobs_for_skills.add(job_id)
        self.skills_per_job.append(int(skills_n))
    
    
    def summary(self) -> Dict[str, Any]:
        def summarize(xs: List[float]) -> Dict[str, Any]:
            if not xs:
                return {"count": 0}
            return {
                "count": len(xs),
                "min": min(xs),
                "p50": pctl(xs, 0.50),
                "p95": pctl(xs, 0.95),
                "max": max(xs),
                "mean": sum(xs) / len(xs),
            }
        
        fields_denom = len(self.seen_jobs_for_fields)
        dq = {
            "desc_len": summarize(self.desc_len),
            "skills_per_job": summarize(self.skills_per_job),
            "company_nonnull_rate": (self.company_present / fields_denom) if fields_denom else None,
            "location_nonnull_rate": (self.location_present / fields_denom) if fields_denom else None,
        }
        
        return {
            "counts": dict(self.counts),
            "fail_reasons": dict(self.fail_reasons),
            "gate_reasons": dict(self.gate_reasons),
            "stage_timings_sec": {k: summarize(v) for k, v in self.stage_durations_sec.items()},
            "per_job_total_sec": summarize(self.per_job_total_sec),
            "data_quality": dq,
        }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)

    group = ap.add_mutually_exclusive_group()
    group.add_argument("--headless", action="store_true")
    group.add_argument("--headed", action="store_true")

    args = ap.parse_args()

    cfg = ScrapeConfig()

    new_headless = cfg.headless
    if args.headless:
        new_headless = True
    elif args.headed:
        new_headless = False

    cfg = replace(
        cfg,
        pages=args.pages if args.pages is not None else cfg.pages,
        limit=args.limit if args.limit is not None else cfg.limit,
        headless=new_headless,
    )

    arts = artifact_paths(cfg)

    ensure_dir(arts["base"])
    ensure_dir(arts["bad_samples"])
    ensure_dir(arts["fail_samples"])

    # init db + begin run (audit)
    db.init_db()

    cfg_payload = asdict(cfg)
    cfg_payload["artifact_dir"] = str(cfg.artifact_dir)
    save_json(arts["config"], cfg_payload)
    db.begin_run(cfg.run_id, cfg_payload)

    auth_file = "data/auth_state.json"
    if not os.path.exists(auth_file):
        emit_log(cfg, "error", "auth_missing", auth_file=auth_file)
        db.end_run(cfg.run_id, {"error": "auth_missing"}, elapsed_sec=0.0, slo_met=None)
        raise SystemExit("Missing auth state: run login script first to generate data/auth_state.json")

    metrics = RunMetrics()
    run_t0 = time.monotonic()

    emit_log(cfg, "info", "run_start", config={"pages": cfg.pages, "per_page": cfg.per_page, "limit": cfg.limit, "headless": cfg.headless})

    bad_saved = 0
    fail_saved = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=cfg.headless)
        context = await browser.new_context(storage_state=auth_file)
        page = await context.new_page()
        stealth = Stealth()
        await stealth.apply_stealth_async(context)
        page.set_default_timeout(cfg.selector_timeout_ms)
        page.set_default_navigation_timeout(cfg.goto_timeout_ms)

        # ---- collect links
        t0 = time.monotonic()
        job_links = await collect_job_links(page, pages=cfg.pages, per_page=cfg.per_page, timeout_ms=cfg.networkidle_timeout_ms)
        dt = time.monotonic() - t0
        metrics.record_stage("collect_links", dt)
        metrics.inc("links_collected_total", len(job_links))
        db.record_event(cfg.run_id, stage="collect_links", status="ok", reason=None, elapsed_ms=ms(dt), details={"count": len(job_links)})
        emit_log(cfg, "info", "links_collected", count=len(job_links), elapsed_sec=round(dt, 3))

        # filter existing
        existing_ids = set(db.get_existing_job_ids())
        new_links = [x for x in job_links if x.get("job_id") not in existing_ids]
        metrics.inc("links_new", len(new_links))
        emit_log(cfg, "info", "links_filtered", existing=len(existing_ids), new=len(new_links))

        new_links = new_links[: cfg.limit]
        emit_log(cfg, "info", "links_capped", limit=cfg.limit, processing=len(new_links))

        # ---- per-job processing
        for i, link in enumerate(new_links, 1):
            job_id = link.get("job_id")
            url = link.get("url")
            job_t0 = time.monotonic()
            emit_log(cfg, "info", "job_start", i=i, total=len(new_links), job_id=job_id, url=url)

            await polite_sleep(cfg)

            # parse with retries
            jd_obj = None
            last_exc: Optional[Exception] = None
            for attempt in range(1, cfg.max_retries + 2):
                try:
                    t1 = time.monotonic()
                    jd_obj = await parse_job_detail(page, url, timeout_ms=cfg.goto_timeout_ms)
                    dt1 = time.monotonic() - t1
                    metrics.record_stage("parse_detail", dt1)
                    db.record_event(cfg.run_id, job_id=job_id, url=url, stage="parse_detail", status="ok", elapsed_ms=ms(dt1))
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    recoverable = is_recoverable_exception(e)
                    reason = f"parse_failed:{type(e).__name__}"
                    db.record_event(
                        cfg.run_id, job_id=job_id, url=url, stage="parse_detail",
                        status="fail", reason=reason, details={"attempt": attempt, "recoverable": recoverable, "error": str(e)[:400]}
                    )
                    emit_log(cfg, "warn" if recoverable else "error", "job_parse_error",
                             job_id=job_id, url=url, attempt=attempt, recoverable=recoverable,
                             error_type=type(e).__name__, error=str(e)[:400])
                    if (not recoverable) or attempt >= (cfg.max_retries + 1):
                        break
                    await asyncio.sleep(compute_backoff(cfg, attempt))

            if jd_obj is None:
                reason = f"parse_failed:{type(last_exc).__name__ if last_exc else 'unknown'}"
                metrics.fail(reason)
                if cfg.save_bad_samples and fail_saved < cfg.bad_sample_max:
                    save_json(arts["fail_samples"] / f"{fail_saved:03d}_{job_id or 'unknown'}.json",
                              {"job_id": job_id, "url": url, "reason": reason, "error": str(last_exc) if last_exc else None})
                    fail_saved += 1
                continue

            job = to_dict(jd_obj)
            metrics.inc("jobs_parsed_ok", 1)

            # DQ: record early quality (before skills)
            jid = job_id or job.get("job_id") or "unknown"
            desc = job.get("description") or ""
            metrics.record_desc_len(jid, len(desc))
            metrics.record_field_presence(
                jid,
                has_company=bool((job.get("company") or "").strip()),
                has_location=bool((job.get("location_text") or "").strip()),
            )

            db.record_event(
                cfg.run_id,
                job_id=job_id,
                url=url,
                stage="dq_snapshot",
                status="ok",
                details={
                    "desc_len": len(job.get("description") or ""),
                    "has_company": bool((job.get("company") or "").strip()),
                    "has_location": bool((job.get("location_text") or "").strip()),
                },
            )

            # gates
            ok, gate_reason = gate_job(cfg, job)
            if not ok:
                metrics.gate(gate_reason or "gated_unknown")
                db.record_event(cfg.run_id, job_id=job_id, url=url, stage="gate", status="gated", reason=gate_reason)
                emit_log(cfg, "info", "job_gated", job_id=job_id, url=url, reason=gate_reason)

                # record in jobs table as "gated" (optional but good for audit)
                try:
                    db.upsert_job(job, scrape_status="gated", scrape_error=gate_reason)
                except Exception:
                    pass

                if cfg.save_bad_samples and bad_saved < cfg.bad_sample_max:
                    save_json(
                        arts["bad_samples"] / f"{bad_saved:03d}_{job_id or 'unknown'}.json",
                        {
                            "job_id": job_id,
                            "url": url,
                            "reason": gate_reason,
                            "title": job.get("title"),
                            "company": job.get("company"),
                            "desc_len": len(job.get("description") or ""),
                            "desc_sha1": sha1_text(job.get("description") or ""),
                            "desc_preview": (job.get("description") or "")[:800],
                        },
                    )
                    bad_saved += 1
                continue
                
            new_hash = job.get("content_hash") or sha1_text(job.get("description") or "")
            old_hash = db.get_job_content_hash(job_id) if job_id else None
            
            # db upsert
            try:
                t2 = time.monotonic()
                db.upsert_job(job, scrape_status="success", scrape_error=None)
                dt2 = time.monotonic() - t2
                metrics.record_stage("db_upsert_job", dt2)
                metrics.inc("jobs_upserted", 1)
                db.record_event(cfg.run_id, job_id=job_id, url=url, stage="db_upsert_job", status="ok", elapsed_ms=ms(dt2))
            except Exception as e:
                reason = f"db_upsert_failed:{type(e).__name__}"
                metrics.fail(reason)
                db.record_event(cfg.run_id, job_id=job_id, url=url, stage="db_upsert_job", status="fail", reason=reason, details={"error": str(e)[:400]})
                emit_log(cfg, "error", "db_upsert_error", job_id=job_id, url=url, error_type=type(e).__name__, error=str(e)[:400])

                if cfg.save_bad_samples and fail_saved < cfg.bad_sample_max:
                    save_json(arts["fail_samples"] / f"{fail_saved:03d}_{job_id or 'unknown'}.json",
                              {"job_id": job_id, "url": url, "reason": reason, "error": str(e)})
                    fail_saved += 1
                continue

            # skills: skip if content hash unchanged
            try:

                if old_hash and new_hash and old_hash == new_hash:
                    metrics.inc("skills_skipped_hash_unchanged", 1)
                    db.record_event(cfg.run_id, job_id=job_id, url=url, stage="extract_skills", status="ok", reason="skipped_hash_unchanged")
                    emit_log(cfg, "info", "skills_skipped", job_id=job_id, reason="hash_unchanged")

                else:
                    t3 = time.monotonic()
                    skills = extract_skills(job.get("description") or "")
                    dt3 = time.monotonic() - t3
                    metrics.record_stage("extract_skills", dt3)
                    db.record_event(cfg.run_id, job_id=job_id, url=url, stage="extract_skills", status="ok", elapsed_ms=ms(dt3), details={"skills_n": len(skills)})

                    t4 = time.monotonic()
                    db.replace_job_skills(job_id, skills)
                    dt4 = time.monotonic() - t4
                    metrics.record_stage("db_replace_skills", dt4)
                    db.record_event(cfg.run_id, job_id=job_id, url=url, stage="db_replace_skills", status="ok", elapsed_ms=ms(dt4))

                    metrics.inc("skills_jobs_written", 1)
                    metrics.inc("skills_total_written", len(skills))

                    metrics.record_skills_n(jid, len(skills))
                    db.record_event(
                        cfg.run_id, job_id=job_id, url=url,
                        stage="dq_skills",
                        status="ok",
                        details={"skills_n": len(skills)}
                    )

            except Exception as e:
                reason = f"skills_failed:{type(e).__name__}"
                metrics.fail(reason)
                db.record_event(cfg.run_id, job_id=job_id, url=url, stage="extract_skills", status="fail", reason=reason, details={"error": str(e)[:400]})
                emit_log(cfg, "warn", "skills_error", job_id=job_id, url=url, error_type=type(e).__name__, error=str(e)[:400])

            metrics.record_job_total(time.monotonic() - job_t0)
            emit_log(cfg, "info", "job_done", job_id=job_id, url=url, elapsed_sec=round(time.monotonic() - job_t0, 3))

        await context.close()
        await browser.close()

    run_elapsed = time.monotonic() - run_t0
    summary = metrics.summary()
    summary.update({
        "run_id": cfg.run_id,
        "finished_utc": utc_now_iso(),
        "elapsed_sec": run_elapsed,
        "artifacts_dir": str(arts["base"]),
        "config": {"pages": cfg.pages, "per_page": cfg.per_page, "limit": cfg.limit, "headless": cfg.headless, "max_retries": cfg.max_retries},
    })

    processed = summary["counts"].get("jobs_parsed_ok", 0)
    succeeded = summary["counts"].get("jobs_upserted", 0)
    success_rate = (succeeded / processed) if processed else None
    slo_met = (success_rate is not None and success_rate >= cfg.slo_success_rate)
    summary["slo"] = {"success_rate": success_rate, "target_success_rate": cfg.slo_success_rate, "met": slo_met}

    dq = summary.get("data_quality", {})
    desc_p50 = (dq.get("desc_len") or {}).get("p50")
    skills_p50 = (dq.get("skills_per_job") or {}).get("p50")
    company_rate = dq.get("company_nonnull_rate")

    dq_slo = {
        "desc_len_p50_min": cfg.dq_desc_len_p50_min,
        "company_nonnull_rate_min": cfg.dq_company_nonnull_rate_min,
        "skills_per_job_p50_min": cfg.dq_skills_per_job_p50_min,
        "observed": {
            "desc_len_p50": desc_p50,
            "company_nonnull_rate": company_rate,
            "skills_per_job_p50": skills_p50,
        },
        "met": (
            (desc_p50 is not None and desc_p50 >= cfg.dq_desc_len_p50_min) and
            (company_rate is not None and company_rate >= cfg.dq_company_nonnull_rate_min) and
            (skills_p50 is not None and skills_p50 >= cfg.dq_skills_per_job_p50_min)
        )
    }
    summary["dq_slo"] = dq_slo

    save_json(arts["run_summary"], summary)
    db.end_run(cfg.run_id, summary, elapsed_sec=run_elapsed, slo_met=slo_met)

    emit_log(cfg, "info", "run_end", elapsed_sec=round(run_elapsed, 3), counts=summary["counts"], slo=summary["slo"])
    emit_log(cfg, "info", "artifacts_written", dir=str(arts["base"]), run_summary=str(arts["run_summary"]))
    emit_log(cfg, "info", "run_end", elapsed_sec=round(run_elapsed, 3), counts=summary["counts"], slo=summary["slo"], dq_slo=summary.get("dq_slo"))


if __name__ == "__main__":
    asyncio.run(main())