#!/usr/bin/env python
"""Capture the README's demo images.

Every shot is driven through the real UI — click the model, press the button,
wait for the result — rather than posed, so the pictures in the README are the
app as it actually behaves. Writes into docs/images/.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from preview import find_browser                      # noqa: E402

from playwright.sync_api import sync_playwright         # noqa: E402


def chip(pg, key: str):
    el = pg.wait_for_selector(f'.nav [data-unit="{key}"]', timeout=30000)
    el.click()


def type_into(pg, text: str):
    box = pg.wait_for_selector(".in-text", timeout=30000)
    pg.fill(".in-text", text)
    return box


def wait_engine(pg, engine: str, limit_s: int = 240):
    for _ in range(limit_s // 4):
        loaded = pg.evaluate(
            "async () => (await (await fetch('/api/engines')).json()).loaded")
        if loaded == engine:
            return True
        pg.wait_for_timeout(4000)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "docs" / "images"))
    ap.add_argument("--only", default=None, help="one shot name, or all")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = os.environ.get("INFERSCOPE_URL", "http://127.0.0.1:8090")

    shots: list[tuple[str, callable]] = []

    def shot_empty(pg):
        pg.evaluate("() => fetch('/api/engines/offload', {method:'POST'})")
        pg.evaluate("() => fetch('/api/chat/server/stop', {method:'POST'})")
        pg.wait_for_timeout(3000)
        pg.reload(wait_until="domcontentloaded")
        pg.wait_for_selector(".stack-empty", timeout=30000)
        pg.wait_for_timeout(700)

    def shot_asr(pg):
        chip(pg, "asr/ggml-base.en.bin")
        pg.wait_for_selector('[data-testid="asr-task"]', timeout=120000)
        pg.wait_for_timeout(800)
        pg.locator('[data-testid="asr-task"]').scroll_into_view_if_needed()

    def shot_tts(pg):
        chip(pg, "tts/en_US-lessac-medium.onnx")
        pg.wait_for_selector(".in-text", timeout=120000)
        wait_engine(pg, "tts")
        type_into(pg, "Every model needs an input.")
        pg.click(".input-box button.primary")
        pg.wait_for_selector('[data-testid="tts-result"]', timeout=120000)
        pg.wait_for_timeout(400)
        pg.locator('[data-testid="tts-result"]').scroll_into_view_if_needed()

    def shot_image(pg):
        chip(pg, "image/sd-turbo")
        pg.wait_for_selector(".in-text", timeout=180000)
        wait_engine(pg, "image")
        type_into(pg, "a green isometric cube on a pale mint background")
        pg.click(".input-box button.primary")
        pg.wait_for_selector('[data-testid="image-result"]', timeout=300000)
        pg.wait_for_timeout(400)
        pg.locator('[data-testid="image-result"]').scroll_into_view_if_needed()

    def shot_arch(pg):
        chip(pg, "llm/Qwen3.8-27B-UD-Q4_K_M.gguf")
        pg.wait_for_selector("svg.scene-svg .tier-model", timeout=180000)
        pg.wait_for_timeout(1500)
        el = pg.query_selector("svg.scene-svg .tier-model")
        el.click()
        pg.wait_for_selector(".arch-svg .arch-boxes > g", timeout=120000)
        pg.wait_for_timeout(900)
        pg.locator(".arch-svg").scroll_into_view_if_needed()

    for name, fn in (("empty", shot_empty), ("asr", shot_asr), ("tts", shot_tts),
                     ("image", shot_image), ("architecture", shot_arch)):
        if a.only and a.only != name:
            continue
        shots.append((name, fn))

    with sync_playwright() as pw:
        exe = find_browser()
        b = pw.chromium.launch(executable_path=exe,
                               args=["--no-sandbox", "--use-fake-device-for-media-stream",
                                     "--use-fake-ui-for-media-stream"]) if exe \
            else pw.chromium.launch(args=["--no-sandbox"])
        pg = b.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=2)
        for name, fn in shots:
            print(f"--- {name}")
            try:
                pg.goto(base, wait_until="domcontentloaded", timeout=90000)
                pg.wait_for_selector(".topbar button.brand", timeout=45000)
                pg.wait_for_timeout(2500)
                fn(pg)
                pg.screenshot(path=str(out / f"{name}.png"))
                print(f"    {out / (name + '.png')}")
            except Exception as e:                          # noqa: BLE001
                print(f"    FAILED: {str(e)[:120]}")
        b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())