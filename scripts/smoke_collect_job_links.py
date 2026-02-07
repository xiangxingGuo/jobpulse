from pathlib import Path
from playwright.sync_api import sync_playwright

STATE_PATH = Path("data/auth_state.json")
BASE = "https://app.joinhandshake.com"

def collect_one_page(page) -> list[str]:

    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    page.wait_for_timeout(1200)

    loc = page.locator('a[href^="/jobs/"]')
    n = loc.count()
    links = []
    for i in range(n):
        href = loc.nth(i).get_attribute("href")
        if href and href.startswith("/jobs/"):
            links.append(BASE + href.split("?")[0])  # Remove query parameters

    # Remove duplicates
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def main():
    if not STATE_PATH.exists():
        raise SystemExit("Missing data/auth_state.json. Run login first.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(storage_state=str(STATE_PATH))
        page = context.new_page()

        page.goto("https://app.joinhandshake.com/job-search?page=1&per_page=25", wait_until="domcontentloaded")
        links = collect_one_page(page)

        print(f"Collected {len(links)} unique job links from page 1:")
        for u in links[:20]:
            print("-", u)
        
        # Save to file for later detail scraping
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        out_path = Path("data/raw/job_links_page1.txt")
        out_path.write_text("\n".join(links), encoding="utf-8")
        print(f"Saved to {out_path}")

        browser.close()

if __name__ == "__main__":
    main()