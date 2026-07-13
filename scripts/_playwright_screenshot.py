"""Isolated Playwright screenshot worker (stdin JSON -> stdout PNG)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _configure_playwright_browsers_path() -> None:
    current = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if current:
        root = Path(current)
        if not any(root.rglob("chrome-headless-shell.exe")):
            os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
    if os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
        return
    for root in [
        Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Temp" / "cursor-sandbox-cache",
    ]:
        if not root.is_dir():
            continue
        for exe in root.rglob("chrome-headless-shell.exe"):
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(exe.parents[2])
            return


def main() -> int:
    payload = json.loads(sys.stdin.read())
    html = payload["html"]
    width = int(payload["width"])
    height = int(payload["height"])

    _configure_playwright_browsers_path()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        try:
            page.set_viewport_size({"width": width, "height": height})
            page.set_content(html)
            # Capture the complete document, not just the requested viewport.
            # Verovio glyphs and long systems can extend beyond the SVG's nominal
            # dimensions; viewport-only screenshots silently cut that content off.
            png = page.screenshot(type="png", omit_background=False, full_page=True)
        finally:
            page.close()
            browser.close()

    sys.stdout.buffer.write(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
