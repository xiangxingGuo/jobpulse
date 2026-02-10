from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import re

from playwright.sync_api import Page


# def _click_expanders(page: Page) -> None:
#     """
#     Handshake job detail often truncates sections behind 'More' buttons.
#     We'll click a few common variants if visible.
#     """
#     candidates = [
#     "button:has-text('More')",
#     "button:has-text('Show more')",
#     "button:has-text('See more')",
#     ]

#     for sel in candidates:
#         loc = page.locator(sel)
#         # click a few times if there are multiple expanders
#         n = min(loc.count(), 6)
#         for i in range(n):
#             try:
#                 btn = loc.nth(i)
#                 if btn.is_visible():
#                     btn.click(timeout=800)
#                     page.wait_for_timeout(300)
#             except Exception:
#                 continue

def _click_expanders(page: Page) -> None:
    """
    Click "Show more / See more / More" buttons to expand truncated sections.
    Must not navigate away from the current page.
    """
    # Use role-based locator to avoid accidentally clicking links
    candidates = [
        page.get_by_role("button", name=re.compile(r"^Show more\b", re.IGNORECASE)),
        page.get_by_role("button", name=re.compile(r"^See more\b", re.IGNORECASE)),
        page.get_by_role("button", name=re.compile(r"^More\b", re.IGNORECASE)),
    ]

    for loc in candidates:
        n = min(loc.count(), 6)
        for i in range(n):
            try:
                btn = loc.nth(i)
                if not btn.is_visible():
                    continue

                before_url = page.url
                btn.click(timeout=800)
                page.wait_for_timeout(250)

                # Hard guard: expanding must not navigate
                if page.url != before_url:
                    raise RuntimeError(
                        f"Unexpected navigation after clicking expander: {before_url} -> {page.url}"
                    )
            except Exception:
                continue


def _clean(s: str) -> str:
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


@dataclass
class JobDetail:
    url: str
    title: str
    company: Optional[str] = None
    posted_text: Optional[str] = None
    apply_by_text: Optional[str] = None
    pay_text: Optional[str] = None
    location_text: Optional[str] = None
    work_auth_text: Optional[str] = None
    opt_cpt_text: Optional[str] = None
    employment_type: Optional[str] = None  # Internship / Full-time etc
    date_range_text: Optional[str] = None  # From ... to ...
    description: Optional[str] = None  # full JD text


def parse_job_detail(page: Page, url: str) -> JobDetail:
    page.goto(url, wait_until="domcontentloaded")
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    page.wait_for_timeout(800)

    _click_expanders(page)
    page.wait_for_timeout(500)

    root = page.locator('[data-hook="job-details-page"]')

    title = root.locator("h1").first.inner_text().strip()

    jd = JobDetail(url=url, title=title)

    company = root.locator('a[aria-label][href^="/e/"] div').first.inner_text().strip()

    if not company:
        about_emp = root.get_by_role("heading", name="About the employer").locator("xpath=ancestor::div[contains(@class,'sc-cYucNP')][1]")
        company = about_emp.locator("h4").first.inner_text().strip()

    jd.company = company
    posted_line = root.locator("h1").locator("xpath=following::div[contains(.,'Posted') and contains(.,'Apply by')][1]").inner_text().strip()

    import re
    m = re.search(r"Posted\s+(.+?)(?:∙|\u2219)\s*Apply by\s+(.+)$", posted_line)
    posted_text = m.group(1).strip() if m else None
    apply_by_text = m.group(2).strip() if m else None

    jd.posted_text = posted_text
    jd.apply_by_text = apply_by_text

    at = root.get_by_role("heading", name="At a glance").locator("xpath=ancestor::div[contains(@class,'sc-cYucNP')][1]")

    desc_block = at.locator("xpath=following::div[contains(@class,'sc-cYucNP')][1]")
    description_text = desc_block.inner_text().strip()
    jd.description = _clean(description_text)

    # Strategy: use body text as robust fallback, then carve out fields.
    body = page.locator("body").inner_text()


    # Pay: "$40–50/hr" or similar usually under "At a glance"
    m = re.search(r"\nAt a glance\n(.+)\n", body)
    if m:
        # first line after "At a glance" is typically pay
        jd.pay_text = m.group(1).strip()

    # Remote/location line often follows pay
    # We'll grab a small block under "At a glance"
    m = re.search(r"\nAt a glance\n(.+\n){1,10}", body)
    if m:
        block = m.group(0)
        lines = [x.strip() for x in block.splitlines() if x.strip()]
        # lines[0] == "At a glance"
        # try to locate location-like line
        for ln in lines:
            if "Remote" in ln or "based in" in ln or "," in ln:
                jd.location_text = ln
                break
        for ln in lines:
            if "work authorization" in ln.lower():
                jd.work_auth_text = ln
            if "OPT/CPT" in ln or "opt/cpt" in ln.lower():
                jd.opt_cpt_text = ln
            if ln in ("Internship", "Full-time", "Part-time", "Contract"):
                jd.employment_type = ln
            if ln.startswith("Full-time∙From") or ln.startswith("Part-time∙From") or ln.startswith("From"):
                jd.date_range_text = ln

    return jd


def to_dict(jd: JobDetail) -> dict:
    return asdict(jd)
