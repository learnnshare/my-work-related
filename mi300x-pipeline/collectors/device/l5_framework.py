"""L5 Framework (PyTorch + HIP) — torch.profiler / Kineto trace."""
from __future__ import annotations
from .common import DeviceCollector


class L5Framework(DeviceCollector):
    layer_id = 5
    name = "l5_framework"

    def env_overrides(self):
        # The workload should wrap its model in torch.profiler and export a
        # Kineto trace to run_ctx["kineto_trace"]; we pass the target path.
        return {"MI300X_KINETO_OUT": str(self.run_ctx["raw_dir"] / "l5" / "kineto.json")}

    def collect_real(self, res):
        trace = self.run_ctx.get("kineto_trace")
        if trace:
            from parsers.kineto import parse_kineto
            sc, fd = parse_kineto(trace)
            res.scalars.update(sc)
            res.fidelity.update(fd)
            return res
        return super().collect_real(res)
