"""Start MineMag on a free local port and open the browser. Same scheme
as backend/launcher.py of DroneMag Studio; a different preferred port so
the two programs can run side by side."""
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

APP_NAME = "MineMag"
HOST = "127.0.0.1"
PREFERRED_PORT = 8100


def pick_port() -> int:
    for port in (PREFERRED_PORT, 8101, 8102, 8103, 8110):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((HOST, port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


def open_browser_when_ready(url: str, timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5):
                break
        except urllib.error.HTTPError:
            break
        except Exception:
            time.sleep(0.4)
    webbrowser.open(url)


def main() -> int:
    port = pick_port()
    url = f"http://{HOST}:{port}/"
    print("=" * 62)
    print(f"  {APP_NAME} - 드론 자력탐사 지뢰·불발탄 탐지")
    print("=" * 62)
    print(f"  주소: {url}")
    print("  브라우저가 자동으로 열립니다. 이 창을 닫으면 프로그램이 종료됩니다.")
    print("=" * 62, flush=True)
    threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True).start()
    uvicorn.run(app, host=HOST, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
