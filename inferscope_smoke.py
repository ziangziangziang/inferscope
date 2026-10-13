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
        # the scene draws from the chosen unit, so make sure one is chosen
        if not pg.query_selector("svg.scene-svg .tier-model"):
            # pick the language model the way a reader would: off the picker.
            # (the old fallback pressed a "start engine" button that now only
            #  exists once the language model is the chosen unit.)
            # The catalogue polls in, so give it a moment to arrive.
            try:
                pg.wait_for_selector('[data-unit^="llm/"]', timeout=20000)
            except Exception:
                pass
            chip = pg.query_selector('[data-unit^="llm/"]')
            if chip:
                chip.click()
                try:
                    pg.wait_for_selector("svg.scene-svg .tier-model", timeout=180000)
                except Exception:
                    fails.append("scene: model block never rendered (engine did not start)")
            else:
                fails.append("scene: model block never rendered")
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
                "() => document.querySelectorAll('.arch-svg .arch-boxes > g').length")
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
            if pg.evaluate("() => document.querySelectorAll('.arch-svg .arch-boxes > g').length"):
                fails.append("panel: clicking the model again did not close the panel")
            # reopen for the sub-layer check
            pg.mouse.move(20, 20)
            pg.wait_for_timeout(600)
            pg.click("svg.scene-svg .tier-model")
            pg.wait_for_timeout(700)

            # one unit of the wrapped repeat group opens its sub-layers
            if pg.evaluate("() => document.querySelectorAll('.arch-svg .arch-boxes > g').length"):
                pg.click(".arch-svg g[data-id='blocks#7']")
                pg.wait_for_timeout(700)
                panel = pg.inner_text(".side-inner")
                if "UNIT 8 OF" not in panel.upper():
                    fails.append(f"scene: unit panel wrong ({panel[:60]!r})")
                for want in ("RMSNorm", "MLP"):
                    if want not in panel:
                        fails.append(f"scene: sub-layers missing {want}")
                pg.screenshot(path=str(SHOTS / "scene-unit.png"))

        # --- engines + ASR, both on the first page ---
        pg.click(".nav button:has-text('Start here')")
        try:
            pg.wait_for_selector('[data-testid="unit-picker"]', timeout=15000)
        except Exception:
            fails.append("index: no unit picker on the first page")
        models = pg.evaluate("() => document.querySelectorAll('[data-unit]').length")
        print("unit chips:", models)
        if models < 2:
            fails.append(f"index: only {models} units in the catalogue")

        # Start from a known state. Residency has two owners — enginectl and
        # chat.py's language server — and the scene reads both, so both have to
        # be cleared or there is no empty->load transition to assert on.
        pg.evaluate("() => fetch('/api/engines/offload', {method:'POST'})")
        # ask chat.py directly rather than pressing "stop engine": that button
        # only exists while the language model is the chosen unit, which is
        # exactly the state this step is trying to clear.
        pg.evaluate("() => fetch('/api/chat/server/stop', {method:'POST'})")
        # Reload rather than sleep: the page holds the chosen unit in state, so
        # a reload is what actually returns it to "nothing picked".
        pg.wait_for_timeout(1500)
        pg.reload(wait_until="domcontentloaded")
        # Both shutdowns are asynchronous — llama.cpp takes tens of seconds to
        # come up and a moment to go away — so wait for the plate to actually
        # go bare rather than guessing with a fixed sleep.
        try:
            pg.wait_for_function("() => !document.querySelector('.stack [data-block]')",
                                 timeout=45000)
        except Exception:
            pass
        pg.wait_for_timeout(1200)
        if pg.query_selector(".stack [data-block]"):
            fails.append("index: scene drew blocks with nothing loaded")
        if not pg.query_selector(".stack-empty"):
            fails.append("index: empty stack gave no explanation")

        # microphone must degrade with a sentence, not a crash
        mic = pg.query_selector('.chip:has-text("microphone")')
        if mic:
            mic.click()
            pg.wait_for_timeout(400)
            rec = pg.query_selector('button:has-text("record")')
            if rec:
                rec.click()
                pg.wait_for_timeout(1500)
                hint = pg.inner_text(".sp-hint") if pg.query_selector(".sp-hint") else ""
                print("mic fallback:", hint[:40])
                if not hint.strip():
                    fails.append("index: record with no mic produced no explanation")
                else:
                    pg.screenshot(path=str(SHOTS / "index-mic.png"))
            pg.click('.chip:has-text("file")')

        # load asr from page 1, then transcribe the bundled sample
        asr_chip = pg.query_selector('[data-unit^="asr/"]:not([disabled])')
        if asr_chip:
            asr_chip.click()
            # the scene must animate in: phase passes through "loading"
            saw_loading = False
            for _ in range(40):
                ph = pg.evaluate("() => document.querySelector('.stack')?.dataset.phase || ''")
                if ph == "loading":
                    saw_loading = True
                    break
                if ph == "loaded":
                    break
                pg.wait_for_timeout(100)
            if not saw_loading:
                fails.append("index: loading a unit did not animate")
            try:
                pg.wait_for_selector('[data-testid="resident-strip"] .resident-name', timeout=180000)
            except Exception:
                fails.append("index: asr never became resident from page 1")
            blocks = pg.evaluate("() => document.querySelectorAll('.stack [data-block]').length")
            print("stack blocks:", blocks)
            if blocks < 2:
                fails.append(f"index: stack drew {blocks} blocks for a loaded unit")
            caps = pg.inner_text(".stack-caps") if pg.query_selector(".stack-caps") else ""
            try:
                pg.wait_for_selector('[data-testid="asr-task"]', timeout=15000)
            except Exception:
                fails.append("index: asr task panel did not appear")
            pg.screenshot(path=str(SHOTS / "index-asr-loaded.png"))

        sample = Path("/workspace/build/whisper.cpp/samples/jfk.wav")
        if sample.exists():
            pg.set_input_files("input[type=file]", str(sample))
            pg.wait_for_timeout(400)
            pg.click('button:has-text("transcribe")')
            try:
                pg.wait_for_selector('[data-testid="asr-result"]', timeout=180000)
            except Exception:
                fails.append("index: no transcript rendered from page 1")
            else:
                txt = pg.inner_text('[data-testid="asr-result"]')
                if "country" not in txt.lower():
                    fails.append(f"index: transcript looks wrong ({txt[:70]!r})")
                print("asr transcript ok (page 1)")
                pg.screenshot(path=str(SHOTS / "index-asr-result.png"))
        else:
            print("no sample audio on this host, skipping transcription")

        off = pg.query_selector('[data-testid="resident-strip"] button:has-text("offload")')
        if off:
            off.click()
            pg.wait_for_timeout(7000)
            if pg.query_selector("[data-testid='resident-strip'] .resident-name"):
                fails.append("index: offload from page 1 did not release the engine")
            if pg.query_selector(".stack [data-block]"):
                fails.append("index: blocks still drawn after offload")

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
