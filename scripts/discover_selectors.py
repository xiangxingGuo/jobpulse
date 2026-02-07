from pathlib import Path
from playwright.sync_api import sync_playwright

STATE_PATH = Path("data/auth_state.json")
URL = "https://app.joinhandshake.com/job-search?page=1&per_page=25"

def main():
    if not STATE_PATH.exists():
        raise SystemExit("Missing data/auth_state.json. Run login first.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(storage_state=str(STATE_PATH))
        page = context.new_page()

        page.goto(URL, wait_until="domcontentloaded")
        page.wait_for_timeout(2500) # wait for page to load

        # find all <a> elements
        anchors = page.locator("a")
        n = anchors.count()

        candidates = []
        for i in range(min(n, 400)):
            a = anchors.nth(i)
            href = a.get_attribute("href") or ""
            text = (a.inner_text() or "").strip().replace("\n", " ")
            if not href:
                continue
        
            # find possible detail URL patterns
            looks_like_job = any(
                    pat in href
                    for pat in [
                        "/job/", "/jobs/", "/postings/", "/stu/postings/", "/employers/", "/career_center/",
                    ]
                )
            if looks_like_job and len(text) >= 6:
                candidates.append((text[:120], href))

        print(f"Total anchors: {n}")
        print(f"Candidates found: {len(candidates)}")
        print("Top 30 candidates (text | href):")
        for t, h in candidates[:30]:
            print("-", t, "|", h)
        
        # extra: print data-testid
        testids = page.locator("[data-testid]").all()
        print("\nSome data-testid samples:")
        for el in testids[:30]:
            v = el.get_attribute("data-testid")
            if v:
                print("-", v)

        browser.close()

if __name__ == "__main__":
    main()