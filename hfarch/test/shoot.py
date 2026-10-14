#!/usr/bin/env python3
"""Draw real models in a real browser and check what the tests cannot.

`node test/graph.test.mjs` covers the arithmetic. Nothing in it can see whether
a card is cut off by the edge of the viewBox, whether a label has been
truncated to fit, or whether a family draws at all — every one of this
project's visual bugs has been invisible in code and obvious in a picture.

    python3 test/shoot.py                          # every known repo, both modes
    python3 test/shoot.py Qwen/Qwen3-4B            # one model
    python3 test/shoot.py --mode 3d whisper/tiny

It writes `test/shots/<repo>-<mode>.png` and asserts, in the live DOM:

  - every card is inside the viewBox with room to spare;
  - no card title or sub-line has been truncated with an ellipsis;
  - no text is wider than the card it sits in;
  - the drawing is not letterboxed: its aspect ratio is inside a usable range.

Requires playwright (`pip install playwright && playwright install chromium`).
"""
import argparse
import json
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

REPOS = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-VL-8B-Instruct",
    "openai/whisper-tiny",
    "stabilityai/stable-diffusion-xl-base-1.0/unet",
    "facebook/DiT-XL-2-256/transformer",
    "PixArt-alpha/PixArt-XL-2-1024-MS/transformer",
    "runwayml/stable-diffusion-v1-5/text_encoder",
    "openai/clip-vit-large-patch14",
]

CHECKS = """
() => {
  const svg = document.getElementById('svg');
  const vb = svg.getAttribute('viewBox').split(/\\s+/).map(Number);
  const out = { vb, clipped: [], truncated: [], overflow: [], cards: 0, text: 0 };
  const inside = (x, y) => x >= vb[0] && x <= vb[0] + vb[2] && y >= vb[1] && y <= vb[1] + vb[3];
  for (const node of document.querySelectorAll('#svg g.node')) {
    out.cards++;
    const b = node.getBBox();
    out.text += node.querySelectorAll('text').length;
    // every corner of every card, not sampled points: a shadow or a label
    // hanging off one edge is how these got missed before
    for (const [x, y] of [[b.x, b.y], [b.x + b.width, b.y],
                          [b.x, b.y + b.height], [b.x + b.width, b.y + b.height]]) {
      if (!inside(x, y)) {
        out.clipped.push({ card: node.getAttribute('aria-label'),
          at: [Math.round(x), Math.round(y)], box: [b.x, b.y, b.width, b.height].map(Math.round) });
        break;
      }
    }
    const card = node.querySelector('rect:not([opacity])');
    if (card) {
      const cw = parseFloat(card.getAttribute('width'));
      for (const t of node.querySelectorAll('text.card-t, text.card-s, text.band-t')) {
        if (/\u2026$/.test(t.textContent)) {
          out.truncated.push({ card: node.getAttribute('aria-label'), text: t.textContent });
        }
        // for text inside the card right now / fix me
        if (t.getBBox().width > cw) {
          out.overflow.push({ card: node.getAttribute('aria-label'),
            text: t.textContent, over: Math.round(t.getBBox().width - cw) });
        }
      }
    }
  }
  out.count = out.cards;
  out.aspect = +(vb[2] / Math.max(1, vb[3])).toFixed(2);
  out.state = document.getElementById('status').dataset.state;
  out.head = document.getElementById('sum').textContent;
  return out;
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="*", default=REPOS)
    ap.add_argument("--url", default="http://127.0.0.1:8811")
    ap.add_argument("--mode", default="2d", choices=["2d", "3d"])
    ap.add_argument("--out", default=str(Path(__file__).parent / "shots"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failures = []

    with sync_playwright() as pw:
        # --no-proxy-server: the sandbox exports HTTP_PROXY, which would put
        # localhost through a proxy that answers for it
        browser = pw.chromium.launch(args=["--no-proxy-server"])
        for repo in args.repos:
            ctx = browser.new_context(viewport={"width": 1500, "height": 940},
                                      device_scale_factor=2)
            page = ctx.new_page()
            msgs = []
            page.on("console", lambda m: msgs.append(f"{m.type}: {m.text}"))
            page.on("pageerror", lambda e: msgs.append(f"pageerror: {e}"))
            try:
                page.goto(f"{args.url}/#{repo}", wait_until="load", timeout=30000)
                if args.mode == "3d":
                    page.evaluate("localStorage.setItem('hfarch.mode','3d')")
                    page.reload(wait_until="load")
                page.wait_for_function(
                    """() => ['ready','error'].includes(
                         document.getElementById('status').dataset.state)""",
                    timeout=60000)
            except Exception as e:
                failures.append(f"{repo}: never settled ({e})")
                print(f"  FAIL  {repo}: never settled — {e}")
                ctx.close()
                continue
            page.wait_for_timeout(350)
            stem = re.sub(r"[^A-Za-z0-9._-]+", "_", repo)
            shot = out / f"{stem}-{args.mode}.png"
            page.screenshot(path=str(shot))
            r = page.evaluate(CHECKS)
            name = f"{repo} [{args.mode}]"
            print(f"\n{name}")
            print(f"  {r['cards']} cards, {r['text']} labels, viewBox aspect {r['aspect']}")
            print(f"  {r['head']}")
            if r["state"] != "ready":
                failures.append(f"{repo}: status {r['state']}")
                print(f"  FAIL  status is {r['state']}")
                for m in msgs[-6:]:
                    print(f"        {m}")
            for c in r["clipped"]:
                failures.append(f"{repo}: card '{c['card']}' outside the viewBox")
                print(f"  FAIL  card outside the viewBox: {c['card']} at {c['at']} box {c['box']}")
            for c in r["truncated"]:
                failures.append(f"{repo}: label truncated '{c['text']}'")
                print(f"  FAIL  label truncated: {c['card']} → {c['text']}")
            for c in r["overflow"]:
                failures.append(f"{repo}: text wider than its card '{c['text']}'")
                print(f"  FAIL  text wider than card: {c['card']} → {c['text']} (+{c['over']}px)")
            if r["aspect"] > 4.2:
                failures.append(f"{repo}: aspect {r['aspect']} — still letterboxed")
                print(f"  FAIL  aspect {r['aspect']}: a thin band with empty margins")
            if not failures or failures[-1].startswith("_ok_"):
                pass
            if not (r["clipped"] or r["truncated"] or r["overflow"]) and r["state"] == "ready":
                print("  ok    every card inside the viewBox, no label truncated")
            ctx.close()
        browser.close()

    print()
    fails = [f for f in failures if not f.startswith("_ok_")]
    if fails:
        print(f"{len(fails)} FAILED")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("all visual checks passed")


if __name__ == "__main__":
    main()
