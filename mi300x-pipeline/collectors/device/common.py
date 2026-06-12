"""
common.py — shared helpers for device collectors (real ROCm tools on Ubuntu).

Every device collector supports a FIXTURE path: if run_ctx["fixture_scalars"]
contains values for this layer's canonical src names, collect() returns those.
This lets the full parse/normalize/predict/publish flow be exercised on a box
without an MI300X, then run unchanged on real hardware.
"""
from __future__ import annotations
import json
import subprocess

from core.interface import BaseCollector, CollectorResult, Cadence
from normalize.layer_map import LAYER_METRICS


def run_json(cmd, timeout=10):
    """Run a command and parse stdout as JSON; return None on any failure."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True).stdout
        return json.loads(out)
    except Exception:
        return None


def run_text(cmd, timeout=10):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True).stdout
    except Exception:
        return None


def read_file(path):
    try:
        return open(path).read().strip()
    except OSError:
        return None


class DeviceCollector(BaseCollector):
    """Base for device collectors with fixture fallback + fidelity defaults."""

    fidelity_default = "measured"

    def _layer_srcs(self):
        return [m["src"] for m in LAYER_METRICS[self.layer_id]]

    def _fixture(self):
        fx = self.run_ctx.get("fixture_scalars")
        if not fx:
            return None
        srcs = self._layer_srcs()
        sub = {k: fx[k] for k in srcs if k in fx}
        return sub or None

    def collect(self) -> CollectorResult:
        res = CollectorResult(layer_id=self.layer_id, cadence=Cadence.SCALAR)
        fx = self._fixture()
        if fx is not None:
            res.scalars.update(fx)
            res.fidelity.update({k: "measured" for k in fx})
            return res
        return self.collect_real(res)

    def collect_real(self, res: CollectorResult) -> CollectorResult:
        """Override with real tool parsing. Default: emit Nones (unobserved)."""
        for src in self._layer_srcs():
            res.scalars.setdefault(src, None)
            res.fidelity[src] = "null"
        return res
