"""L0 Silicon — rocprofv3 HW counters + amd-smi/rocm-smi (power/temp/clock/util)."""
from __future__ import annotations
from core.interface import Cadence, Sample
from core.env import has_binary
from .common import DeviceCollector, run_json, amd_smi_json, deep_find


class L0Silicon(DeviceCollector):
    layer_id = 0
    name = "l0_silicon"

    def available(self):
        if has_binary("amd-smi"):
            return True, "ok (amd-smi)"
        if has_binary("rocm-smi"):
            return True, "ok (rocm-smi)"
        return False, "neither amd-smi nor rocm-smi found"

    def start(self, workload_pid=None):
        self._gpu = self.cfg.get("gpu_index", 0)
        self._use_amd = has_binary("amd-smi")

    def _sample_amd_smi(self, t_ns):
        m = amd_smi_json("metric", gpu=self._gpu)
        if not m:
            return
        vals = {
            "power_w": deep_find(m, "socket_power", "average_socket_power", "power"),
            "temp_c": deep_find(m, "hotspot", "junction", "edge"),
            "clock_mhz": deep_find(m, "gfx", "gfx_0", "sclk"),
            "hbm_util_pct": deep_find(m, "umc_activity", "memory_activity", "mem_usage"),
            "mfma_util_pct": deep_find(m, "gfx_activity", "graphics_activity"),
        }
        for src, v in vals.items():
            if isinstance(v, (int, float)):
                self.series.setdefault(src, []).append(Sample(t_ns, src, float(v)))

    def _sample_rocm_smi(self, t_ns):
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

    def sample(self, t_ns):
        if getattr(self, "_use_amd", False):
            self._sample_amd_smi(t_ns)
        else:
            self._sample_rocm_smi(t_ns)

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
