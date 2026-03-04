# Reliability Statement

This document defines what the scraper guarantees and what it does not.

---

# 1. Guarantees

Under normal operating conditions:

- ≥95% ingestion success rate
- Idempotent upsert behavior
- Content-hash based incremental processing
- Structured audit trail for each job
- Run-level summary artifacts
- Data quality SLO validation

Each run produces:
- run_summary.json
- config.json
- scrape_runs DB record
- scrape_events DB record

---

# 2. Does NOT Guarantee

- Immunity to major platform layout redesign
- Unlimited scraping without rate limits
- Zero latency variance (browser-dominated)
- Perfect skill extraction (LLM heuristic dependent)

---

# 3. Failure Handling Strategy

## Parse Failure
- Retries with exponential backoff
- Event recorded in scrape_events
- Non-blocking to overall pipeline

## DB Failure
- Job marked failed
- Error persisted

## Extraction Failure
- Ingestion preserved
- Extraction logged as failure
- Does not block job persistence

## Drift Detection
- desc_len collapse
- skills_per_job drop
- field completeness degradation

DQ-SLO failure triggers investigation but not pipeline abort.