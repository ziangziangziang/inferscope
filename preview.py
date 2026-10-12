#!/usr/bin/env python3
"""Local visual-review harness for the InferScope scene.

Builds the UI, serves `dist/` on localhost, and drives Chromium with
`/api/**` fulfilled from the *real* remote backend over the rc-tunnel. That
keeps the review loop honest (real model config, real measurements) without
deploying every iteration.

    python3 preview.py --view index --out shots/
    python3 preview.py --view index --click '.model-block' --out shots/

The tunnel URL is resolved automatically over SSH unless --url is given.
"""
from __future__ import annotations
import argparse, json, os, ssl, subprocess, sys, threading, time, urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UI = ROOT / "ui"
DIST = UI / "dist"

SSH = ["ssh", "-p", "31699", "-o", "StrictHostKeyChecking=no",
       "-o", "ConnectTimeout=15", "root@36.150.116.206"]

# Endpoints the scene actually needs; everything else gets a harmless stub so a
# dead backend can't stall the page on an SSE stream.
STUBS = {
    "/api/stream": "data: {\"done\":true}\n\n",
}


def resolve_url() -> str:
    try:
        out = subprocess.run(SSH + [
            "curl -s --max-time 20 http://127.0.0.1:8090/api/access"
        ], capture_output=True, text=True, timeout=60).stdout
        url = json.loads(out).get("public_url")
        if url:
            return url
    except Exception as e:                                    # noqa: BLE001
        print(f"! could not resolve tunnel url: {e}")
    raise SystemExit("no tunnel URL; pass --url")


def build(skip: bool) -> None:
    if skip and DIST.is_dir():
        print("= build skipped")
        return
    print("= building ui ...")
    r = subprocess.run(["npm", "run", "build:only"], cwd=UI,
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit("build failed")
    print("= build ok")


def serve(port: int) -> ThreadingHTTPServer:
    handler = partial(SimpleHTTPRequestHandler, directory=str(DIST))
    handler.log_message = lambda *a, **k: None            # type: ignore[method-assign]
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def fetch_remote(url: str, timeout: int = 45) -> tuple[int, bytes, str]:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.status, r.read(), r.headers.get("content-type", "application/json")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), "application/json"


def find_browser() -> str | None:
    roots = [Path.home() / "Library/Caches/ms-playwright", Path.home() / ".cache/ms-playwright"]
    pats = ["chromium_headless_shell-*/**/chrome-headless-shell",
            "chromium-*/**/chrome-linux64/chrome", "chromium-*/**/chrome"]
    for r in roots:
        if not r.is_dir():
            continue
        for p in pats:
            for c in sorted(r.glob(p)):
                if c.is_file() and os.access(c, os.X_OK):
                    return str(c)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="index")
    ap.add_argument("--out", default="shots")
    ap.add_argument("--url", default=None)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--height", type=int, default=1100)
    ap.add_argument("--full", action="store_true", help="full page screenshot")
    ap.add_argument("--click", action="append", default=[],
                    help="selector to click before shooting (repeatable)")
    ap.add_argument("--hover", action="append", default=[])
    ap.add_argument("--wait", type=float, default=2.5)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--eval", default=None, help="JS to run before shooting")
    args = ap.parse_args()

    build(args.no_build)
    base = args.url or resolve_url()
    serve(args.port)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=find_browser(),
                                    args=["--force-color-profile=srgb"])
        page = browser.new_page(viewport={"width": args.width, "height": args.height},
                                device_scale_factor=args.scale)
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

        def handler(route):
          try:
            path = route.request.url.split("127.0.0.1:%d" % args.port)[-1]
            path = path if path.startswith("/") else "/" + path
            for s, body in STUBS.items():
                if path.startswith(s):
                    return route.fulfill(status=200, body=body,
                                          content_type="text/event-stream")
            try:
                st, body, ct = fetch_remote(base + path)
            except Exception as e:                            # noqa: BLE001
                errors.append(f"fetch {path}: {e}")
                return route.fulfill(status=200, body="{}",
                                      content_type="application/json")
            return route.fulfill(status=st, body=body, content_type=ct)
          except Exception:                                  # noqa: BLE001
            return route.fulfill(status=200, body="{}",
                                 content_type="application/json")

        page.route("**/api/**", handler)
        page.goto(f"http://127.0.0.1:{args.port}/#{args.view}",
                  wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(int(args.wait * 1000))

        for sel in args.click:
            try:
                page.click(sel, timeout=8000)
                page.wait_for_timeout(900)
            except Exception as e:                            # noqa: BLE001
                errors.append(f"click {sel}: {e}")
        for sel in args.hover:
            try:
                page.hover(sel, timeout=8000)
                page.wait_for_timeout(600)
            except Exception as e:                            # noqa: BLE001
                errors.append(f"hover {sel}: {e}")
        if args.eval:
            print("eval ->", page.evaluate(args.eval))

        shot = out / f"{args.view}.png"
        page.screenshot(path=str(shot), full_page=args.full)
        print(f"= shot {shot}")

        svg = page.query_selector("svg.tree-svg")
        if svg:
            box = svg.bounding_box()
            print(f"= scene box: {box}")
            sub = out / f"{args.view}-scene.png"
            page.screenshot(path=str(sub), clip=box, scale="css") if False else \
                svg.screenshot(path=str(sub))
            print(f"= scene shot {sub}")

    if errors:
        print("\n!! page problems:")
        for e in dict.fromkeys(errors):
            print("   ", e)
    else:
        print("\n== no console/page errors")


if __name__ == "__main__":
    main()