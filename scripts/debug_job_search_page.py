from pathlib import Path
from playwright.sync_api import sync_playwright

STATE_PATH = Path("data/auth_state.json")
URL = "https://app.joinhandshake.com/job-search?page=1&per_page=25"

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def scroll_some(page, steps=6):
    for _ in range(steps):
        page.mouse.wheel(0, 1200)
        page.wait_for_timeout(800)

def main():
    if not STATE_PATH.exists():
        raise SystemExit("Missing data/auth_state.json. Run: uv run python scripts/login.py")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(storage_state=str(STATE_PATH))
        page = context.new_page()

        page.goto(URL, wait_until="domcontentloaded")
        page.wait_for_timeout(1000)

        # Wait for network to be idle
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        scroll_some(page, steps=6)
        page.wait_for_timeout(1500)

        # 1) Screenshot
        png_path = OUT_DIR / "job_search_debug.png"
        page.screenshot(path=str(png_path), full_page=True)
        print(f"Saved screenshot: {png_path}")

        # 2) Save runtime DOM HTML
        html_path = OUT_DIR / "job_search_debug.html"
        html_path.write_text(page.content(), encoding="utf-8")
        print(f"Saved runtime HTML: {html_path}")

        # 3) Print visible text (first 1200 chars)
        body_text = page.locator("body").inner_text().strip().replace("\r", "")
        print("\n=== BODY TEXT (first 1200 chars) ===")
        print(body_text[:1200])
        print("=== END BODY TEXT ===\n")

        # 4) Enumerate "possible job card" clickable elements
        selectors = [
            "a",
            "[role=link]",
            "button",
            "[role=button]",
            "[tabindex='0']",
            "[onclick]",
            "[data-testid]",
            "[data-qa]",
        ]

        seen = set()
        candidates = []

        for sel in selectors:
            loc = page.locator(sel)
            cnt = min(loc.count(), 200)
            for i in range(cnt):
                el = loc.nth(i)
                try:
                    txt = (el.inner_text() or "").strip().replace("\n", " ")
                except Exception:
                    continue
                if len(txt) < 10:
                    continue

                href = ""
                try:
                    href = el.get_attribute("href") or ""
                except Exception:
                    pass

                # Filter by keywords: job pages usually contain these words
                key_hit = any(k in txt.lower() for k in ["intern", "engineer", "data", "software", "machine", "research", "analyst", "apply"])
                if not key_hit:
                    continue

                sig = (txt[:80], href, sel)
                if sig in seen:
                    continue
                seen.add(sig)
                candidates.append(sig)

        print(f"Clickable-ish candidates: {len(candidates)}")
        print("Top 40 (text | href | matched_selector):")
        for t, h, s in candidates[:40]:
            print("-", t, "|", h, "|", s)

        print("\nCurrent URL:", page.url)
        print("Title:", page.title())

        browser.close()

if __name__ == "__main__":
    main()
