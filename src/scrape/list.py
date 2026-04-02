from __future__ import annotations

import re
from typing import Dict, List

from playwright.async_api import Page

BASE = "https://app.joinhandshake.com"

_JOB_RE = re.compile(r"^/jobs/(\d+)\b")


async def collect_job_links(
    page: Page, pages: int = 1, per_page: int = 25, timeout_ms: int = 15000
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()

    for p in range(1, pages + 1):
        url = f"{BASE}/job-search?page={p}&per_page={per_page}"
        await page.goto(url, wait_until="domcontentloaded")
        try:
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception:
            # Best-effort, avoid hard failure on flaky network
            pass

        # anchor hrefs
        loc = page.locator('a[href^="/jobs/"]')
        n = await loc.count()
        for i in range(n):
            href = (await loc.nth(i).get_attribute("href")) or ""
            m = _JOB_RE.match(href)
            if not m:
                continue
            job_id = m.group(1)
            clean = f"{BASE}/jobs/{job_id}"
            if clean in seen:
                continue
            seen.add(clean)
            out.append({"job_id": job_id, "url": clean})

    return out
