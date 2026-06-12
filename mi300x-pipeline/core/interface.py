"""
interface.py — uniform collector contract.

Every per-layer collector (device or gem5) subclasses BaseCollector and returns
a CollectorResult. The orchestrator drives all collectors identically:

    available() -> setup() -> start(pid) -> [sample(t_ns) ...] -> stop()
                -> collect() -> teardown()

Trace-based collectors (rocprofv3, ftrace, Kineto, gem5 stats) do their work in
start/stop and ignore sample(); polling collectors (rocm-smi, sysfs,
/proc/interrupts) implement sample(). The orchestrator does not care which.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Cadence(Enum):
    SCALAR = "scalar"          # one value for the whole run (firmware, partition)
    TIMESERIES = "timeseries"  # sampled at run cadence (power, clock, temp, util)
    PERKERNEL = "perkernel"    # one row per kernel dispatch (launch latency, flops)


@dataclass
class Sample:
    t_ns: int
    key: str
    value: float | str
    unit: str | None = None


@dataclass
class CollectorResult:
    layer_id: int
    cadence: Cadence
    scalars: dict[str, Any] = field(default_factory=dict)        # canonical src -> value
    series: dict[str, list[Sample]] = field(default_factory=dict)
    perkernel: list[dict] = field(default_factory=list)
    raw_artifacts: list[str] = field(default_factory=list)       # absolute paths
    fidelity: dict[str, str] = field(default_factory=dict)       # src -> measured|derived|synthetic|null
    errors: list[str] = field(default_factory=list)

    def merge(self, other: "CollectorResult") -> None:
        self.scalars.update(other.scalars)
        self.series.update(other.series)
        self.perkernel.extend(other.perkernel)
        self.raw_artifacts.extend(other.raw_artifacts)
        self.fidelity.update(other.fidelity)
        self.errors.extend(other.errors)


class BaseCollector:
    """Base class for all layer collectors. Override what you need."""

    layer_id: int = -1
    name: str = "base"

    def __init__(self, cfg: dict, run_ctx: dict):
        self.cfg = cfg or {}
        self.run_ctx = run_ctx
        self.enabled = self.cfg.get("enabled", True)
        self._raw_dir = run_ctx["raw_dir"] / f"l{self.layer_id}"

    # --- lifecycle (all optional to override) ---
    def available(self) -> tuple[bool, str]:
        """Preflight: return (ok, reason). Default: always available."""
        return True, "ok"

    def setup(self) -> None:
        self._raw_dir.mkdir(parents=True, exist_ok=True)

    def start(self, workload_pid: int | None = None) -> None:
        pass

    def sample(self, t_ns: int) -> None:
        pass

    def stop(self) -> None:
        pass

    def collect(self) -> CollectorResult:
        return CollectorResult(layer_id=self.layer_id, cadence=Cadence.SCALAR)

    def teardown(self) -> None:
        pass

    # --- helpers ---
    def env_overrides(self) -> dict[str, str]:
        """Env vars this collector needs injected into the workload process
        (e.g. ROCBLAS_LAYER, MIOPEN_ENABLE_LOGGING, NCCL_DEBUG, profiler flags)."""
        return {}
