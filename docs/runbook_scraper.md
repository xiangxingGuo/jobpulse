# Scraper Runbook

---

# 1. Common Symptoms & Diagnosis

## skills_per_job_p50 drops

Check:
- Description quality
- parse_mode
- extract_skills changes

Likely causes:
- DOM structure changed
- Noise injected into description block

---

## desc_len_p50 collapses

Likely:
- Selector failed
- Fallback body extraction truncated

Action:
- Inspect bad_samples
- Compare raw HTML

---

## success_rate drops

Check:
- auth_state.json validity
- rate limiting
- Playwright timeouts

---

# 2. Recovery Steps

1. Re-run with lower pages
2. Enable headless mode for stability
3. Increase timeout
4. Inspect scrape_events table

---

# 3. Investigating Drift

Query:

SELECT stage, reason FROM scrape_events WHERE run_id='50eed95dea';

Look for:
- parse_failed spikes
- gate triggers
- extract failures