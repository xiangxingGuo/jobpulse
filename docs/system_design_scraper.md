# Scraper System Design

---

# 1. High-Level Architecture

List Collector → Detail Parser → Quality Gate → DB Upsert → Skills Extraction → Metrics Aggregation → Artifact Persist

---

# 2. Components

## List Collector
- Playwright-based
- URL normalization
- Deduplication

## Detail Parser
- DOM-based structured extraction
- Expander handling
- Fallback to body when necessary

## Data Quality Gate
- URL validation
- UI chrome marker filter
- Minimum description length

## Persistence Layer
- SQLite (WAL mode)
- Idempotent upsert
- content_hash for incremental detection
- scrape_runs & scrape_events audit tables

## Metrics Layer
- Stage timing histogram
- Per-job latency
- Data quality distribution
- SLO evaluation

---

# 3. Data Flow

Browser → Structured dict → Gate → DB → Skill extraction → DB update → Run summary

---

# 4. Failure Modes

| Failure | Detection | Mitigation |
|----------|-----------|------------|
| Selector drift | desc_len collapse | Fallback + DQ-SLO |
| Layout redesign | company_nonnull_rate drop | Runbook inspection |
| Anti-bot throttle | parse timeout spikes | Backoff + retry |
| Extraction regression | skills_p50 drop | DQ-SLO failure |

---

# 5. Scaling Considerations

- Browser-dominated latency (~12s/job)
- DB operations negligible
- Parallelization possible with worker pool
- Rate-limit aware

---

# 6. Observability

- Structured logs
- Run artifacts
- DB audit tables
- SLO result persisted