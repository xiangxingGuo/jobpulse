from pathlib import Path
from playwright.sync_api import sync_playwright

STATE_PATH = Path("data/auth_state.json")

def main():
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        page.goto("https://app.joinhandshake.com/")
        print("Navigated to Handshake login page")
        input()
        context.storage_state(path=STATE_PATH)
        print(f"Saved authentication state to {STATE_PATH}")
        browser.close()

if __name__ == "__main__":
    main()
