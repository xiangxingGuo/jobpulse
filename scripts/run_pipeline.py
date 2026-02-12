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

def get_existing_job_ids(conn) -> set[int]:
    cur = conn.cursor()
    cur.execute("SELECT job_id FROM jobs;")
    rows = cur.fetchall()
    return list(set([int(row[0]) for row in rows]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=20)
    ap.add_argument("--per-page", type=int, default=50)
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    if not STATE_PATH.exists():
        raise SystemExit("Missing data/auth_state.json. Run: uv run python scripts/login.py")

    conn = get_conn()
    init_db(conn)

    processed_job_ids = get_existing_job_ids(conn)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Set to False to debug
        context = browser.new_context(storage_state=str(STATE_PATH))
        page = context.new_page()

        links = collect_job_links(page, pages=args.pages, per_page=args.per_page)
        links = links[: args.limit]
        print(f"Collected {len(links)} job links")

        links = [link for link in links if job_id_from_url(link) not in processed_job_ids]
        print(f"{len(links)} new job links after filtering out existing job_ids")

        for idx, url in enumerate(links, 1):
            # randomly pause every few iterations to mimic human behavior and avoid anti-scraping measures
            import random
            import time
            stop_time = random.randint(1, 10)
            time.sleep(stop_time)
            extra_flag = random.randint(1, 100)
            if extra_flag <= 40:  # 40% chance of an extra pause
                extra_time = random.randint(2, 8)
                time.sleep(extra_time) 

            print(f"[{idx}/{len(links)}] Scraping detail: {url}")
            jd = parse_job_detail(page, url)
            row = to_dict(jd)
            row["job_id"] = job_id_from_url(url)

            import hashlib

            # desc = row.get("description") or ""
            # print("\n=== PRE-DB CHECK ===")
            # print("target_url:", url)
            # print("page_url:", page.url if hasattr(page, "url") else None)
            # print("desc_len:", len(desc))
            # print("desc_head:", desc[:120].replace("\n", " "))
            # print("desc_sha1:", hashlib.sha1(desc.encode("utf-8")).hexdigest())

            # Gate 1: must still be on Handshake job detail page
            if not page.url.startswith("https://app.joinhandshake.com/jobs/"):
                print(f"❌ SKIP: navigated away from Handshake job page: {page.url}")
                continue

            # Gate 2: reject obvious UI chrome masquerading as description
            desc = row.get("description") or ""
            bad_markers = ["Explore Feed", "Inbox", "Career center", "Get the app", "Skip to content", "View all"]
            if any(m in desc for m in bad_markers):
                print("❌ SKIP: description looks like UI chrome (bad markers found)")
                continue

            # Gate 3: too short is suspicious (tune threshold later)
            if len(desc) < 800:
                print(f"❌ SKIP: description too short ({len(desc)})")
                continue


            upsert_job(conn, row)

            skills = extract_skills((row.get("description") or "") + " " + (row.get("title") or ""))
            replace_job_skills(conn, row["job_id"], skills)

        browser.close()

    # build_report(conn, Path("data/reports/latest.md"))
    # print("✅ Report generated: data/reports/latest.md")

if __name__ == "__main__":
    main()
