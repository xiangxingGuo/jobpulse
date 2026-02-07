from __future__ import annotations
import re
from typing import List
from playwright.sync_api import Page

BASE = "https://app.joinhandshake.com"

def collect_job_links(page: Page, pages: int = 1, per_page: int = 25) -> List[str]:
    out: List[str] = []
    seen = set()

    for p in range(1, pages + 1):
        url = f"{BASE}/job-search?page={p}&per_page={per_page}"
        page.goto(url, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        page.wait_for_timeout(1200)

        loc = page.locator('a[href^="/jobs/"]')
        n = loc.count()
        for i in range(n):
            href = loc.nth(i).get_attribute("href") or ""
            m = re.match(r"^/jobs/(\d+)", href)
            if not m:
                continue
            clean = f"{BASE}/jobs/{m.group(1)}"
            if clean not in seen:
                seen.add(clean)
                out.append(clean)

    return out
