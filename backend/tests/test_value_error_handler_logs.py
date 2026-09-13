"""main.py's blanket ValueError -> 400 exception_handler is needed because
processing/* deliberately raises plain ValueError for user-facing
validation messages, but a bare-type catch like this also swallows any
*unintentional* ValueError a real bug happens to raise, converting it into
the same "your fault" 400 as a real validation error - with no trace of it
in the server's own logs. Confirms the handler now logs the exception
(with traceback) before responding, so an unexpected ValueError is still
visible server-side even though the client still gets a clean 400.

Calls the handler function directly rather than through a live HTTP
request: app.main mounts a catch-all StaticFiles("/") for the frontend
build as its last route, which (being a Starlette Mount matching any
path prefix) would swallow any new test-only route added to the shared
app afterward before it ever reached FastAPI's exception-handling layer."""
import asyncio
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.main import value_error_handler


def test_value_error_response_and_server_log(caplog):
    # Raise-and-catch first (rather than just constructing the ValueError)
    # so sys.exc_info() is actually populated inside the handler, matching
    # how Starlette really invokes exception handlers - from within the
    # except block that caught the original exception.
    try:
        raise ValueError("의도적인 테스트 오류")
    except ValueError as exc:
        with caplog.at_level(logging.WARNING, logger="app.main"):
            response = asyncio.run(value_error_handler(None, exc))

    assert response.status_code == 400
    assert b"\xec\x9d\x98\xeb\x8f\x84\xec\xa0\x81\xec\x9d\xb8" in response.body  # utf-8 for "의도적인"
    assert any("의도적인 테스트 오류" in rec.message for rec in caplog.records)
    assert any(rec.exc_info is not None for rec in caplog.records)


if __name__ == "__main__":
    print("This test uses pytest's caplog fixture - run via pytest.")
