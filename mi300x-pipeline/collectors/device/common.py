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


def amd_smi_json(subcommand, gpu=0, timeout=10):
    """Run `amd-smi <subcommand> -g <gpu> --json` (ROCm 7 / AMD-SMI). amd-smi
    returns a list (one entry per GPU) or a dict; normalize to the GPU's dict."""
    import shutil
    if not shutil.which("amd-smi"):
        return None
    d = run_json(["amd-smi", subcommand, "-g", str(gpu), "--json"], timeout=timeout)
    if d is None:
        return None
    if isinstance(d, list):
        return d[0] if d else {}
    if isinstance(d, dict) and "gpu" in d and isinstance(d["gpu"], list):
        return d["gpu"][0] if d["gpu"] else {}
    return d


def deep_find(obj, *keys):
    """Find the first numeric/scalar value whose key matches any of `keys`
    (case-insensitive) anywhere in a nested dict/list. amd-smi nests values like
    {"power": {"socket_power": {"value": 412, "unit": "W"}}}."""
    targets = {k.lower() for k in keys}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.lower() in targets:
                    if isinstance(v, dict) and "value" in v:
                        return v["value"]
                    if not isinstance(v, (dict, list)):
                        return v
                r = walk(v)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = walk(v)
                if r is not None:
                    return r
        return None

    val = walk(obj)
    try:
        return float(val)
    except (TypeError, ValueError):
        return val


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
