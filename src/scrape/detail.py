from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import re

from playwright.async_api import Page


def _clean(s: str) -> str:
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


async def _click_expanders(page: Page) -> None:
    """
    Click "Show more / See more / More" buttons to expand truncated sections.
    Must not navigate away from the current page.
    """
    candidates = [
        page.get_by_role("button", name=re.compile(r"^Show more\b", re.IGNORECASE)),
        page.get_by_role("button", name=re.compile(r"^See more\b", re.IGNORECASE)),
        page.get_by_role("button", name=re.compile(r"^More\b", re.IGNORECASE)),
    ]

    for loc in candidates:
        try:
            n = min(await loc.count(), 6)
        except Exception:
            continue

        for i in range(n):
            try:
                btn = loc.nth(i)
                if not await btn.is_visible():
                    continue
                before_url = page.url
                await btn.click(timeout=800)
                await page.wait_for_timeout(200)
                if page.url != before_url:
                    # hard guard: expander must not navigate
                    raise RuntimeError(f"Unexpected navigation after expander click: {before_url} -> {page.url}")
            except Exception:
                continue


@dataclass
class JobDetail:
    job_id: Optional[str]
    url: str
    title: str
    company: Optional[str] = None
    posted_text: Optional[str] = None
    apply_by_text: Optional[str] = None
    pay_text: Optional[str] = None
    location_text: Optional[str] = None
    work_auth_text: Optional[str] = None
    opt_cpt_text: Optional[str] = None
    employment_type: Optional[str] = None
    date_range_text: Optional[str] = None
    description: Optional[str] = None


def _job_id_from_url(url: str) -> Optional[str]:
    m = re.search(r"/jobs/(\d+)", url)
    return m.group(1) if m else None


async def parse_job_detail(page: Page, url: str, timeout_ms: int = 20000) -> JobDetail:
    await page.goto(url, wait_until="domcontentloaded")
    try:
        await page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except Exception:
        pass

    await page.wait_for_timeout(500)
    await _click_expanders(page)
    await page.wait_for_timeout(300)

    root = page.locator('[data-hook="job-details-page"]')
    # title
    title = (await root.locator("h1").first.inner_text()).strip()

    jd = JobDetail(job_id=_job_id_from_url(url), url=url, title=title)

    # company: best-effort
    try:
        company = (await root.locator('a[aria-label][href^="/e/"] div').first.inner_text()).strip()
    except Exception:
        company = ""
    if not company:
        try:
            about_emp = root.get_by_role("heading", name="About the employer").locator(
                "xpath=ancestor::div[contains(@class,'sc-cYucNP')][1]"
            )
            company = (await about_emp.locator("h4").first.inner_text()).strip()
        except Exception:
            company = None
    jd.company = company

    # posted/apply-by line
    try:
        posted_line = (await root.locator("h1").locator(
            "xpath=following::div[contains(.,'Posted') and contains(.,'Apply by')][1]"
        ).inner_text()).strip()

        m = re.search(r"Posted\s+(.+?)(?:∙|\u2219)\s*Apply by\s+(.+)$", posted_line)
        jd.posted_text = m.group(1).strip() if m else None
        jd.apply_by_text = m.group(2).strip() if m else None
    except Exception:
        pass

    # description: prefer structured block near "At a glance", fallback to body carving
    desc = None
    try:
        at = root.get_by_role("heading", name="At a glance").locator(
            "xpath=ancestor::div[contains(@class,'sc-cYucNP')][1]"
        )
        desc_block = at.locator("xpath=following::div[contains(@class,'sc-cYucNP')][1]")
        desc = (await desc_block.inner_text()).strip()
    except Exception:
        desc = None

    if not desc:
        try:
            body = await page.locator("body").inner_text()
            # naive fallback: keep body but clean it; later we can improve by slicing around known headings
            desc = body
        except Exception:
            desc = ""

    jd.description = _clean(desc)

    # optional: parse some “At a glance” fields from body (best-effort)
    try:
        body = await page.locator("body").inner_text()
        m = re.search(r"\nAt a glance\n(.+)\n", body)
        if m:
            jd.pay_text = m.group(1).strip()

        m = re.search(r"\nAt a glance\n(.+\n){1,10}", body)
        if m:
            block = m.group(0)
            lines = [x.strip() for x in block.splitlines() if x.strip()]
            for ln in lines:
                if "Remote" in ln or "based in" in ln or "," in ln:
                    jd.location_text = ln
                    break
            for ln in lines:
                if "work authorization" in ln.lower():
                    jd.work_auth_text = ln
                if "opt/cpt" in ln.lower():
                    jd.opt_cpt_text = ln
                if ln in ("Internship", "Full-time", "Part-time", "Contract"):
                    jd.employment_type = ln
                if ln.startswith("Full-time∙From") or ln.startswith("Part-time∙From") or ln.startswith("From"):
                    jd.date_range_text = ln
    except Exception:
        pass

    return jd


def to_dict(jd: JobDetail) -> dict:
    return asdict(jd)