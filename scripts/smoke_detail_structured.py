from pathlib import Path
from playwright.sync_api import sync_playwright
from src.scrape.detail import parse_job_detail, to_dict

STATE_PATH = Path("data/auth_state.json")
LINKS_FILE = Path("data/raw/job_links_page1.txt")

def main():
    url = LINKS_FILE.read_text(encoding="utf-8").splitlines()[0].strip()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(storage_state=str(STATE_PATH))
        page = context.new_page()

        jd = parse_job_detail(page, url)
        d = to_dict(jd)

        print("=== STRUCTURED ===")
        for k in ["title","company","posted_text","apply_by_text","pay_text","location_text","employment_type","date_range_text","work_auth_text","opt_cpt_text"]:
            print(f"{k}: {d.get(k)}")

        print("\n=== DESCRIPTION SNIPPET (first 800 chars) ===")
        print((d.get("description") or "")[:800])

        browser.close()

if __name__ == "__main__":
    main()
