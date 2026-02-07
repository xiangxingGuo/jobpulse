from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

STATE_PATH = Path("data/auth_state.json")

def main():
    if not STATE_PATH.exists():
        raise SystemExit(f"Missing {STATE_PATH}. Run: uv run python scripts/login.py")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(storage_state=str(STATE_PATH))
        page = context.new_page()

        page.goto("https://app.joinhandshake.com/", wait_until="domcontentloaded")
        page.wait_for_timeout(1000)

        url = page.url
        print(f"Current URL: {url}")

        # Check if the user is logged in
        # if not logged in, the URL will contain "login" or "signin"
        if "login" in url or "signin" in url:
            raise SystemExit("Looks logged out. Re-run login to refresh auth_state.json")

        print("✅ Logged-in state looks OK.")
        browser.close()

if __name__ == "__main__":
    main()
