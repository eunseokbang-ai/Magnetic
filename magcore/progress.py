"""What a long operation is doing right now, and a way to stop it.

Processing, gridding, inversion and restoring a project are single
blocking requests. On a large survey they run for tens of seconds to
minutes, and until 2026-11 the screen showed a spinner and nothing else:
no way to tell a slow step from a hung one, and no way out short of
closing the program. This module is the smallest thing that fixes both
without turning every request into a queue:

- The code doing the work calls `report("측선 판별", 0.15)` at its stage
  boundaries and inside its long loops. That is all a computation has to
  know - no callback threaded through signatures. Outside a job the call
  does nothing, so library functions can report unconditionally.
- An entry point wraps the work in `with job("자료 처리"):`. The job is
  registered where another request can see it (`active_jobs`), which is
  how the screen draws a progress bar while the first request is still
  running.
- `cancel(job_id)` only raises a flag. The next `report()` in the working
  thread raises JobCancelled there, which unwinds the computation like
  any exception. Cancellation is therefore cooperative: it takes effect
  at the next reporting point, not in the middle of a matrix solve. A
  stage that reports nothing for a long time cannot be interrupted, and
  the screen says so by showing no percentage for it.

A job started inside another one (restoring a bundle re-runs the
pipeline) does not replace it: the inner `job()` becomes a stretch of the
outer one, and `span(lo, hi)` maps whatever fractions are reported inside
onto that stretch, so the bar keeps moving forward instead of restarting.

The current job travels in a ContextVar. Each request runs in its own
thread with its own context, so two requests never see each other's job;
threads a computation starts itself (numba, a thread pool) do not inherit
it and their `report()` calls are silently no-ops, which is the safe
direction.
"""
from __future__ import annotations

import contextvars
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Iterator


class JobCancelled(Exception):
    """Raised inside the working thread at the first report() after the
    user asked to stop."""

    def __init__(self, name: str = ""):
        super().__init__(f"작업을 취소했습니다{': ' + name if name else ''}.")
        self.name = name


class Job:
    def __init__(self, name: str, project_id: str | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.project_id = project_id
        self.stage = name
        self.fraction: float | None = None
        self.started_at = time.time()
        self.stage_started_at = self.started_at
        self._cancel = threading.Event()
        # the stretch of the whole bar that reports currently map onto
        self._lo, self._hi = 0.0, 1.0

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def request_cancel(self) -> None:
        self._cancel.set()

    def check(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled(self.name)

    def report(self, stage: str | None, fraction: float | None) -> None:
        self.check()
        if stage is not None and stage != self.stage:
            self.stage = stage
            self.stage_started_at = time.time()
        if fraction is None:
            self.fraction = None
        else:
            f = min(1.0, max(0.0, float(fraction)))
            self.fraction = self._lo + (self._hi - self._lo) * f

    def to_dict(self) -> dict:
        now = time.time()
        return {
            "id": self.id,
            "name": self.name,
            "project_id": self.project_id,
            "stage": self.stage,
            # None = this stage cannot say how far along it is
            "fraction": None if self.fraction is None else round(self.fraction, 4),
            "elapsed_s": round(now - self.started_at, 1),
            "stage_elapsed_s": round(now - self.stage_started_at, 1),
            "cancel_requested": self.cancel_requested,
        }


_current: contextvars.ContextVar[Job | None] = contextvars.ContextVar("magcore_job", default=None)
_active: dict[str, Job] = {}
_active_lock = threading.Lock()


def current() -> Job | None:
    return _current.get()


def report(stage: str | None = None, fraction: float | None = None) -> None:
    """Say where the work is. `fraction` is 0..1 of the enclosing job (or
    of the enclosing span); None means "in this stage, extent unknown".
    Raises JobCancelled if the user asked to stop. A no-op outside a job."""
    job_ = _current.get()
    if job_ is not None:
        job_.report(stage, fraction)


def check() -> None:
    """Only the cancellation test, for a loop with nothing new to say."""
    job_ = _current.get()
    if job_ is not None:
        job_.check()


@contextmanager
def job(name: str, project_id: str | None = None) -> Iterator[Job]:
    """Run the enclosed work as a visible, cancellable job. Inside an
    existing job this is just a named stage of it."""
    outer = _current.get()
    if outer is not None:
        outer.report(name, None)
        yield outer
        return
    job_ = Job(name, project_id)
    with _active_lock:
        _active[job_.id] = job_
    token = _current.set(job_)
    try:
        yield job_
    finally:
        _current.reset(token)
        with _active_lock:
            _active.pop(job_.id, None)


@contextmanager
def span(lo: float, hi: float, stage: str | None = None) -> Iterator[None]:
    """Fractions reported inside map onto lo..hi of the enclosing range."""
    job_ = _current.get()
    if job_ is None:
        yield
        return
    old = (job_._lo, job_._hi)
    width = old[1] - old[0]
    job_._lo, job_._hi = old[0] + width * lo, old[0] + width * hi
    try:
        job_.report(stage, 0.0)
        yield
    finally:
        job_._lo, job_._hi = old


def active_jobs() -> list[dict]:
    with _active_lock:
        jobs = list(_active.values())
    return [j.to_dict() for j in sorted(jobs, key=lambda j: j.started_at)]


def cancel(job_id: str) -> bool:
    """Ask a running job to stop. False if there is no such job (it may
    have just finished)."""
    with _active_lock:
        job_ = _active.get(job_id)
    if job_ is None:
        return False
    job_.request_cancel()
    return True
