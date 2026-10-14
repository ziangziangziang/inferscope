#!/usr/bin/env python3
"""
The smallest server that makes Model Atlas work here.

`python3 -m http.server` is enough on a normal network. It is not enough on
this one, and the reason is not visible in the browser's error message:

  - `huggingface.co` cannot be reached at all from here;
  - `hf-mirror.com` can, but the CDN it redirects to answers with
    `Access-Control-Allow-Origin: https://hf-mirror.com`, so a page served
    from http://localhost:8811 gets nothing, and the browser reports it as
    a bare `TypeError` — which is why typing a model name appeared to do
    nothing at all.

So this server adds exactly one route, `/api/hf/config`, which reads the same
file with urllib, where the network is fine, and hands it to the page that
asked. It is deliberately narrow:

  - bound to 127.0.0.1 — this is a developer's localhost, not a service;
  - three filenames, never a path chosen by the caller;
  - two allowlisted hosts;
  - `owner/name` validated against the same rule the app uses.

Nothing is fetched unless a browser asks, and no weight is ever touched: a
model's shape is declared in config.json, and that is the only file read.

    python3 serve.py 8811
    open http://localhost:8811
"""

import json
import re
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

HOSTS = ("https://huggingface.co", "https://hf-mirror.com")
FILES = ("config.json", "generation_config.json", "preprocessor_config.json")
SEG = r"[A-Za-z0-9][A-Za-z0-9._-]*"          # no leading dot: no ".." segments
REPO = re.compile(rf"^{SEG}/{SEG}(/{SEG})?$")  # owner/name, or owner/name/subfolder
TIMEOUT = 12
UA = "ModelAtlas/1.0 (+local static server; reads config.json only)"

# Repeated reads of the same repo inside a session are cached: a model's shape
# does not change, and the each request costs a round trip to whichever host
# answers first.
_seen = {}
_lock = threading.Lock()


def read_repo(path):
    """Read the three metadata files from whichever host answers.

    Returns (payload, errors). Each exception is named — a bare `except` here
    used to report a JSON bug as a network failure on the wrong host.
    """
    parts = path.split("/")
    owner, name = parts[0], parts[1]
    sub = parts[2] if len(parts) > 2 else ""
    problems = {}
    for host in HOSTS:
        files = {}
        for fname in FILES:
            url = f"{host}/{owner}/{name}/resolve/main/{sub + '/' if sub else ''}{fname}"
            try:
                req = Request(url, headers={"User-Agent": UA})
                with urlopen(req, timeout=TIMEOUT) as r:
                    if r.status != 200:
                        if fname == "config.json":
                            problems[host] = f"HTTP {r.status}"
                        continue
                    body = r.read(1_000_000).decode("utf-8", "replace")
                try:
                    files[fname] = json.loads(body)
                except ValueError as e:
                    if fname == "config.json":
                        problems[host] = f"{fname} is not JSON: {e}"
            except HTTPError as e:
                # 401/403 is what a gated repo says; anything else is a miss
                if fname == "config.json":
                    problems[host] = ("gated" if e.code in (401, 403) else f"HTTP {e.code}")
            except (URLError, TimeoutError, ConnectionError) as e:
                problems[host] = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
            if not files and fname == "config.json":
                break        # no config here: the other two are optional, skip the wait
        if files.get("config.json") is not None:
            return {"repo": path, "host": host, "files": files}, problems
    return None, problems


def _cached(repo):
    with _lock:
        hit = _seen.get(repo)
    if hit and "files" in hit:
        out = dict(hit)
        out["cached"] = True
        return out
    if hit:
        return hit
    payload, problems = read_repo(repo)
    if payload is None:
        gated = [h for h, why in (problems or {}).items() if why == "gated"]
        payload = {
            "repo": repo,
            "error": "this repo is gated — the Hub answers 401/403 without a token"
                     if gated else "no host could read this repo",
            "detail": problems,
        }
    with _lock:
        _seen[repo] = payload
    return payload


class Handler(SimpleHTTPRequestHandler):
    extensions_map = SimpleHTTPRequestHandler.extensions_map.copy()
    extensions_map.update({".js": "text/javascript", ".mjs": "text/javascript"})

    def do_GET(self):
        parts = urlsplit(self.path)
        if parts.path == "/api/hf/config":
            return self.hf_config(parse_qs(parts.query or ""))
        return super().do_GET()

    def reply(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def hf_config(self, query):
        repo = (query.get("repo") or [""])[0].strip()
        if not REPO.match(repo):
            return self.reply(400, {"detail": "give a model name like Qwen/Qwen3-4B"})
        try:
            payload = _cached(repo)
        except Exception as e:                      # never leak a traceback
            return self.reply(502, {"detail": f"{type(e).__name__}: {e}"})
        if "error" in payload:
            return self.reply(502, {"detail": payload["error"], "detail_by_host": payload.get("detail", {})})
        return self.reply(200, payload)

    def log_message(self, fmt, *args):
        if "/api/hf/config" in (args[0] if args else ""):
            super().log_message(fmt, *args)


class Server(ThreadingHTTPServer):       # threading already built in
    daemon_threads = True
    allow_reuse_address = True


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8811
    host = "127.0.0.1"
    print(f"Model Atlas on http://{host}:{port}  (Ctrl-C to stop)")
    print(f"  static files from {__import__('os').getcwd()}")
    print("  /api/hf/config reads config.json through the shell's network")
    Server((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
