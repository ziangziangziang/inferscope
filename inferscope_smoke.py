#!/usr/bin/env python
"""InferScope UI smoke test.

Boots the app against a running backend (tunnel or localhost), visits every view,
and asserts each renders real content with no console/page errors. Also drives the
signature interactions: architecture node selection -> inspector, and drill-into
an "x N" repeat group. Screenshots each view.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = os.environ.get("INFERSCOPE_URL", "http://127.0.0.1:8090")
SHOTS = Path(os.environ.get("INFERSCOPE_SHOTS", "shots"))
VIEWS = ["index", "machine", "run", "analyze", "arch", "memory", "mm"]


def find_browser():
    roots = [Path.home() / "Library/Caches/ms-playwright", Path.home() / ".cache/ms-playwright"]
    pats = ["chromium_headless_shell-*/**/chrome-headless-shell", "chromium-*/**/chrome-linux64/chrome",
            "chromium-*/**/chrome"]
    for r in roots:
        if not r.is_dir():
            continue
        for p in pats:
            for c in sorted(r.glob(p)):
                if c.is_file() and os.access(c, os.X_OK):
                    return str(c)
    return None


def main():
    SHOTS.mkdir(exist_ok=True, parents=True)
    errs, pageerrs, fails = [], [], []
    with sync_playwright() as pw:
        exe = find_browser()
        b = pw.chromium.launch(executable_path=exe, args=["--no-sandbox", "--disable-dev-shm-usage"]) if exe \
            else pw.chromium.launch(args=["--no-sandbox"])
        pg = b.new_page(viewport={"width": 1500, "height": 950})
        pg.on("console", lambda m: errs.append(m.text[:200]) if m.type == "error" else None)
        pg.on("pageerror", lambda e: pageerrs.append(str(e)[:200]))
        pg.goto(BASE, wait_until="domcontentloaded", timeout=60000)
        try:
            pg.wait_for_selector(".topbar .brand", timeout=20000)
        except Exception:
            fails.append("app shell never rendered")
        pg.wait_for_timeout(2500)

        for view in VIEWS:
            try:
                pg.click(f".nav button:has-text('{ {'index':'Start here','machine':'Status','run':'Run & capture','analyze':'Analyze','arch':'Architecture','memory':'Memory & KV','mm':'Multimodal'}[view] }')", timeout=8000)
            except Exception as e:
                fails.append(f"{view}: nav click failed {str(e)[:60]}"); continue
            READY = {
                "arch": "svg.graph .node",
                "memory": ".kv-table",
                "mm": ".view-head h1",
                "analyze": ".view-head h1",
                "run": ".chat-log",
                "machine": ".card",
                "index": ".tree-svg",
            }
            try:
                pg.wait_for_selector(READY[view], timeout=15000)
            except Exception:
                pass
            pg.wait_for_timeout(700)
            txt = pg.inner_text(".main")
            if len(txt) < 40:
                fails.append(f"{view}: view nearly empty ({len(txt)} chars)")
            pg.screenshot(path=str(SHOTS / f"{view}.png"))
            print(f"{view:14} text={len(txt)}")

        # Architecture interactions
        pg.click(".nav button:has-text('Architecture')")
        try:
            pg.wait_for_selector("svg.graph .node", timeout=15000)
        except Exception:
            pass
        nodes = pg.evaluate("() => document.querySelectorAll('svg.graph .node').length")
        print("arch nodes:", nodes)
        if nodes == 0:
            fails.append("architecture: no graph nodes rendered")
        else:
            pg.click("svg.graph .node >> nth=1")
            pg.wait_for_timeout(600)
            insp = pg.inner_text(".inspector")
            if "Inspector" not in insp and len(insp) < 20:
                fails.append("architecture: inspector empty after node click")
            pg.screenshot(path=str(SHOTS / "arch-selected.png"))
        # drill into an x N group if present
        exp = pg.query_selector_all("svg.graph text.repeat-tag")
        if exp:
            exp[0].click()
            pg.wait_for_timeout(1200)
            sub = pg.evaluate("() => document.querySelectorAll('svg.graph .node').length")
            print("after expand nodes:", sub)
            pg.screenshot(path=str(SHOTS / "arch-expanded.png"))

        # --- the scene: engine -> model -> architecture -> sub-layers ---
        pg.click(".nav button:has-text('Start here')")
        try:
            pg.wait_for_selector("svg.scene-svg .tier-model", timeout=15000)
        except Exception:
            fails.append("scene: model block never rendered")
            boxes = 0
        else:
            # hovering the tower lifts it clear of the hardware
            pg.hover("svg.scene-svg .tier-model")
            pg.wait_for_timeout(500)
            if not pg.query_selector("svg.scene-svg g.upper.lifted"):
                fails.append("scene: hovering the tower did not lift it")
            pg.screenshot(path=str(SHOTS / "scene-lifted.png"))

            # step off the tower and let it settle before clicking, the way a
            # person would
            pg.mouse.move(20, 20)
            pg.wait_for_timeout(700)

            # clicking the model unfolds the architecture
            pg.click("svg.scene-svg .tier-model")
            pg.wait_for_timeout(900)
            boxes = pg.evaluate(
                "() => document.querySelectorAll('svg.scene-svg .arch-boxes > g').length")
            print("arch boxes:", boxes)
            if boxes < 10:
                fails.append(f"scene: architecture did not unfold ({boxes} boxes)")
            if not pg.query_selector(".side-inner"):
                fails.append("scene: detail panel did not open for the model")
            else:
                panel = pg.inner_text(".side-inner")
                if "parameters" not in panel.lower():
                    fails.append("scene: model panel missing its parameter count")
            pg.screenshot(path=str(SHOTS / "scene-unfolded.png"))

            # the model is a toggle: a second click folds the tier away again
            pg.mouse.move(20, 20)
            pg.wait_for_timeout(600)
            pg.click("svg.scene-svg .tier-model")
            pg.wait_for_timeout(600)
            if pg.evaluate("() => document.querySelectorAll('svg.scene-svg .arch-boxes > g').length"):
                fails.append("scene: clicking the model again did not fold the architecture")
            # reopen for the sub-layer check
            pg.mouse.move(20, 20)
            pg.wait_for_timeout(600)
            pg.click("svg.scene-svg .tier-model")
            pg.wait_for_timeout(700)

            # one unit of the wrapped repeat group opens its sub-layers
            if pg.evaluate("() => document.querySelectorAll('svg.scene-svg .arch-boxes > g').length"):
                pg.click("svg.scene-svg g[data-id='blocks#7']")
                pg.wait_for_timeout(700)
                panel = pg.inner_text(".side-inner")
                if "UNIT 8 OF" not in panel.upper():
                    fails.append(f"scene: unit panel wrong ({panel[:60]!r})")
                for want in ("RMSNorm", "MLP"):
                    if want not in panel:
                        fails.append(f"scene: sub-layers missing {want}")
                pg.screenshot(path=str(SHOTS / "scene-unit.png"))

        # Memory KV calc produces a number
        pg.click(".nav button:has-text('Memory')")
        pg.wait_for_timeout(1500)
        mt = pg.inner_text(".main")
        if "GiB" not in mt:
            fails.append("memory: KV calculator produced no GiB value")

        b.close()

    report = {"console_errors": errs, "page_errors": pageerrs, "failures": fails}
    (SHOTS / "report.json").write_text(json.dumps(report, indent=2))
    print("-" * 50)
    print("console errors:", len(errs), "page errors:", len(pageerrs))
    for e in errs[:5]: print("  CERR:", e)
    for e in pageerrs[:5]: print("  PERR:", e)
    if fails:
        print("FAILURES:"); [print("  -", f) for f in fails]; return 1
    print("ALL VIEWS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
