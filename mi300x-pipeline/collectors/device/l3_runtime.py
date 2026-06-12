"""L3 Runtime (ROCr/HSA) — rocprofv3 HIP/HSA/kernel trace."""
from __future__ import annotations
from core.env import has_binary
from .common import DeviceCollector


class L3Runtime(DeviceCollector):
    layer_id = 3
    name = "l3_runtime"

    def available(self):
        ok = has_binary("rocprofv3") or has_binary("rocprof")
        return ok, "ok" if ok else "rocprofv3/rocprof not found"

    def collect_real(self, res):
        trace = self.run_ctx.get("rocprof_trace")
        if trace:
            from parsers.rocprof_csv import trace_to_l3
            sc, fd = trace_to_l3(trace)
            res.scalars.update(sc)
            res.fidelity.update(fd)
            return res
        return super().collect_real(res)
