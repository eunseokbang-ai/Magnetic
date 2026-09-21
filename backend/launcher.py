"""Entry point for the packaged desktop build.

Starts the same FastAPI application the development server runs, on a port
chosen at startup, and opens the browser at it once it is actually
answering.

Why a port is chosen rather than fixed at 8000: the app is a local web
program, so a second copy (or a stale one left running from a previous
launch) would otherwise fail to bind and the browser would silently keep
talking to the old process - which is exactly the confusion this build is
meant to remove for someone who did not set the program up themselves.
The frontend calls its API at the relative path /api, so it does not care
which port it ended up on.
"""
from __future__ import annotations

import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

import uvicorn

from app.main import app

# One place for the product name, so the console banner, the API title
# and the browser tab cannot drift apart.
APP_NAME = "DroneMag Studio"

HOST = "127.0.0.1"
PREFERRED_PORT = 8000


def pick_port() -> int:
    for port in (PREFERRED_PORT, 8001, 8002, 8003, 8010):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((HOST, port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))  # let the OS pick anything free
        return probe.getsockname()[1]


def open_browser_when_ready(url: str, timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5):
                break
        except urllib.error.HTTPError:
            break  # answering, even if with an error status
        except Exception:
            time.sleep(0.4)
    webbrowser.open(url)


def main() -> int:
    port = pick_port()
    url = f"http://{HOST}:{port}/"

    print("=" * 62)
    print(f"  {APP_NAME} - 드론 자력탐사 자료처리")
    print("=" * 62)
    print(f"  주소: {url}")
    print("  브라우저가 자동으로 열립니다. 이 창을 닫으면 프로그램이 종료됩니다.")
    print("=" * 62, flush=True)

    threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True).start()
    uvicorn.run(app, host=HOST, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
