"""L4 Math libraries — rocBLAS/MIOpen/RCCL logs (env-enabled) + kernel flops."""
from __future__ import annotations
from .common import DeviceCollector


class L4MathLibs(DeviceCollector):
    layer_id = 4
    name = "l4_mathlibs"

    def env_overrides(self):
        ov = {"ROCBLAS_LAYER": str(self.cfg.get("rocblas_layer", 2))}
        if self.cfg.get("miopen_logging", True):
            ov["MIOPEN_ENABLE_LOGGING"] = "1"
            ov["MIOPEN_LOG_LEVEL"] = "6"
        if self.cfg.get("nccl_debug"):
            ov["NCCL_DEBUG"] = str(self.cfg["nccl_debug"])
        return ov

    def collect_real(self, res):
        logs = {
            "rocblas": self.run_ctx.get("rocblas_log"),
            "miopen": self.run_ctx.get("miopen_log"),
            "rccl": self.run_ctx.get("rccl_log"),
        }
        if any(logs.values()):
            from parsers.rccl_log import parse_lib_logs
            sc, fd = parse_lib_logs(logs)
            res.scalars.update(sc)
            res.fidelity.update(fd)
            return res
        return super().collect_real(res)
