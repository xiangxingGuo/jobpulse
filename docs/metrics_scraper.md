# Scraper Metrics & SLOs

This document defines the operational and data quality metrics for the scraping pipeline.

---

# 1. Service-Level Indicators (SLIs)

## Ingestion SLIs

- links_collected_total
- jobs_parsed_ok
- jobs_upserted
- skills_jobs_written
- success_rate = jobs_upserted / jobs_parsed_ok

### Latest Run (run_id: 50eed95dea)

- Jobs processed: 62
- Success rate: 100%
- Target: ≥ 95%
- SLO met: ✅

Source: run_summary.json (50eed95dea)

---

## Latency SLIs

Per-job stage timings:

| Stage            | P50 (s) | P95 (s) |
|------------------|---------|---------|
| parse_detail     | 12.0    | 15.3    |
| db_upsert_job    | 0.008   | 0.016   |
| extract_skills   | 0.009   | 0.022   |
| per_job_total    | 15.2    | 24.8    |

Observation:
- >95% of end-to-end latency is dominated by browser rendering.
- Database and NLP extraction are negligible contributors.

# 2. Data Quality Metrics

## Description Length

- p50: 6868 characters
- p95: 10960
- min: 2985

Interpretation:
- Description blocks are large and stable.
- No truncation or obvious parsing collapse.

## Skills per Job

- p50: 1
- p95: 6
- mean: 2.0

Target p50 ≥ 2.0  
Result: ❌ Not met

This indicates potential:
- overly strict extraction rules
- noise in description block
- structural drift in page layout

## Field Completeness

- company_nonnull_rate = 1.0
- location_nonnull_rate = 1.0

This indicates selector stability for core metadata.

# 3. Defined SLOs

## Availability SLO

- success_rate ≥ 0.95

## Data Quality SLO

- desc_len_p50 ≥ 600
- company_nonnull_rate ≥ 0.85
- skills_per_job_p50 ≥ 2.0

Latest Run:
- Ingestion SLO: ✅
- DQ SLO: ❌ (skills_per_job_p50 too low)

This separation prevents silent degradation.