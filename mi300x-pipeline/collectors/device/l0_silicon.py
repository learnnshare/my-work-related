"""L0 Silicon — rocprofv3 HW counters + rocm-smi (power/temp/clock polled)."""
from __future__ import annotations
from core.interface import Cadence, Sample
from core.env import has_binary
from core.sampling import now_ns
from .common import DeviceCollector, run_json


class L0Silicon(DeviceCollector):
    layer_id = 0
    name = "l0_silicon"

    def available(self):
        if has_binary("rocm-smi"):
            return True, "ok"
        return False, "rocm-smi not found"

    def start(self, workload_pid=None):
        # rocprofv3 wraps the workload (handled by orchestrator env); here we poll SMI
        self._gpu = self.cfg.get("gpu_index", 0)

    def sample(self, t_ns):
        d = run_json(["rocm-smi", "--showpower", "--showtemp", "--showgpuclocks", "--json"])
        if not d:
            return
        card = next(iter(d.values())) if isinstance(d, dict) else {}
        def num(*keys):
            for k in keys:
                for ck, cv in card.items():
                    if k.lower() in ck.lower():
                        try:
                            return float(str(cv).split()[0])
                        except (ValueError, IndexError):
                            pass
            return None
        for src, val in (("power_w", num("average socket power", "power")),
                         ("temp_c", num("junction", "edge", "temperature")),
                         ("clock_mhz", num("sclk"))):
            if val is not None:
                self.series.setdefault(src, []).append(Sample(t_ns, src, val))

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.series = {}

    def collect_real(self, res):
        res.cadence = Cadence.TIMESERIES
        res.series.update(self.series)
        # rocprofv3 counters parsed by parsers/rocprof_csv if a counter csv exists
        csv = self.run_ctx.get("rocprof_csv")
        if csv:
            from parsers.rocprof_csv import counters_to_l0
            sc, fd = counters_to_l0(csv, self.run_ctx.get("peak", {}))
            res.scalars.update(sc)
            res.fidelity.update(fd)
        return res
