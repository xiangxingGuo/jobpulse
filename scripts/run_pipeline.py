import re
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright

from src.db import get_conn, init_db, upsert_job, replace_job_skills
from src.extract import extract_skills
from src.scrape.list import collect_job_links
from src.scrape.detail import parse_job_detail, to_dict
from src.report import build_report

STATE_PATH = Path("data/auth_state.json")

def job_id_from_url(url: str) -> str:
    m = re.search(r"/jobs/(\d+)", url)
    if not m:
        raise ValueError(f"Cannot parse job_id from url: {url}")
    return m.group(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=1)
    ap.add_argument("--per-page", type=int, default=25)
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    if not STATE_PATH.exists():
        raise SystemExit("Missing data/auth_state.json. Run: uv run python scripts/login.py")

    conn = get_conn()
    init_db(conn)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Set to False to debug
        context = browser.new_context(storage_state=str(STATE_PATH))
        page = context.new_page()

        links = collect_job_links(page, pages=args.pages, per_page=args.per_page)
        links = links[: args.limit]
        print(f"Collected {len(links)} job links")

        for idx, url in enumerate(links, 1):
            print(f"[{idx}/{len(links)}] Scraping detail: {url}")
            jd = parse_job_detail(page, url)
            row = to_dict(jd)
            row["job_id"] = job_id_from_url(url)

            upsert_job(conn, row)

            skills = extract_skills((row.get("description") or "") + " " + (row.get("title") or ""))
            replace_job_skills(conn, row["job_id"], skills)

        browser.close()

    build_report(conn, Path("data/reports/latest.md"))
    print("✅ Report generated: data/reports/latest.md")

if __name__ == "__main__":
    main()
