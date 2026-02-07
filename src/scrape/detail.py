from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import re

from playwright.sync_api import Page


def _click_expanders(page: Page) -> None:
    """
    Handshake job detail often truncates sections behind 'More' buttons.
    We'll click a few common variants if visible.
    """
    candidates = [
        "button:has-text('More')",
        "button:has-text('Show more')",
        "button:has-text('See more')",
        "a:has-text('More')",
        "[role=button]:has-text('More')",
    ]
    for sel in candidates:
        loc = page.locator(sel)
        # click a few times if there are multiple expanders
        n = min(loc.count(), 6)
        for i in range(n):
            try:
                btn = loc.nth(i)
                if btn.is_visible():
                    btn.click(timeout=800)
                    page.wait_for_timeout(300)
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

    # Strategy: use body text as robust fallback, then carve out fields.
    body = page.locator("body").inner_text()

    # Title: usually appears in <title> and also as a heading on page.
    title = page.title().split("|")[0].strip()
    jd = JobDetail(url=url, title=title)

    # Heuristic extraction from body text (robust across DOM changes)
    # Company: often appears above title; but easiest is from page title after the first |
    # Example: "Data Science Intern ... | Jobfair® | Handshake"
    parts = page.title().split("|")
    if len(parts) >= 2:
        jd.company = parts[1].strip().replace("®", "®")

    # Posted / apply by line: "Posted 2 weeks ago∙Apply by February 21, 2026 at 10:59 PM"
    m = re.search(r"Posted\s+(.+?)(?:∙|\u2219)\s*Apply by\s+(.+)", body)
    if m:
        jd.posted_text = m.group(1).strip()
        jd.apply_by_text = m.group(2).strip().split("\n")[0]

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

    # Description: best effort by anchoring from "About The Role" or similar headers
    desc = None
    anchor_headers = ["About The Role", "Job Description", "The Role", "Description"]
    for h in anchor_headers:
        if h in body:
            # take text after header up to next known section
            after = body.split(h, 1)[1]
            # stop at known next sections
            stop_headers = ["What they're looking for", "Qualifications", "Required Qualifications", "Benefits", "Application"]
            end = len(after)
            for sh in stop_headers:
                idx = after.find(sh)
                if idx != -1:
                    end = min(end, idx)
            desc = after[:end]
            desc = _clean(desc)
            break

    # Fallback: use full body if anchor missing
    jd.description = desc if desc else _clean(body)

    return jd


def to_dict(jd: JobDetail) -> dict:
    return asdict(jd)
