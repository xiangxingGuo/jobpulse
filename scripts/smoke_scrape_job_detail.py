from pathlib import Path
from playwright.sync_api import sync_playwright

STATE_PATH = Path("data/auth_state.json")

def main():
    if not STATE_PATH.exists():
        raise SystemExit("Missing data/auth_state.json. Run login first.")

    links_file = Path("data/raw/job_links_page1.txt")
    if not links_file.exists():
        raise SystemExit("Missing data/raw/job_links_page1.txt. Run smoke_collect_job_links.py first.")

    first_url = links_file.read_text(encoding="utf-8").splitlines()[0].strip()
    print("Opening:", first_url)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(storage_state=str(STATE_PATH))
        page = context.new_page()

        page.goto(first_url, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass
        page.wait_for_timeout(1200)

        # Grab the body text
        body = page.locator("body").inner_text()

        # Check if the page contains job detail indicators
        print("Page title:", page.title())
        print("Body snippet:\n", body[:1200])

        browser.close()

if __name__ == "__main__":
    main()
