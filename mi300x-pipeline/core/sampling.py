"""
sampling.py — cadence clock that ticks polling collectors at a fixed interval,
anchoring every sample to a common t0 (monotonic ns) for cross-layer alignment.
"""
from __future__ import annotations
import threading
import time


def now_ns() -> int:
    return time.monotonic_ns()


class SampleLoop:
    def __init__(self, collectors, interval_ms: int = 100):
        self.collectors = collectors
        self.interval = interval_ms / 1000.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.t0 = None

    def start(self):
        self.t0 = now_ns()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            t = now_ns()
            for c in self.collectors:
                try:
                    c.sample(t)
                except Exception as e:  # a flaky poll must not kill the run
                    c_errs = getattr(c, "_sample_errors", None)
                    if c_errs is None:
                        c._sample_errors = []
                    c._sample_errors.append(str(e))
            # sleep the remainder of the interval
            elapsed = (now_ns() - t) / 1e9
            self._stop.wait(max(0.0, self.interval - elapsed))

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
