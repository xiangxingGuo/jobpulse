from pathlib import Path
from playwright.sync_api import sync_playwright

STATE_PATH = Path("data/auth_state.json")

BASE = "https://app.joinhandshake.com"
SEARCH_URL = "https://app.joinhandshake.com/job-search?page=1&per_page=25"


def main():
    if not STATE_PATH.exists():
        raise SystemExit("Missing data/auth_state.json. Run login first.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(storage_state=str(STATE_PATH))
        page = context.new_page()

        page.goto(SEARCH_URL, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        page.wait_for_timeout(1500)

        # Find all job detail links
        links = page.locator('a[href^="/jobs/"]')
        n = links.count()
        print("Found /jobs links:", n)
        if n == 0:
            raise SystemExit("No /jobs links found. Try running debug script again.")

        a = links.first
        title = (a.inner_text() or "").strip().replace("\n", " ")
        href = a.get_attribute("href")
        print("Title:", title)
        print("Link:", BASE + href)

        browser.close()

if __name__ == "__main__":
    main()