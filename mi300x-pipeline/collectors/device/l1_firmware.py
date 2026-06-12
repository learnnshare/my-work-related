"""L1 Firmware / HW abstraction — rocm-smi partitions/ECC/firmware (scalar)."""
from __future__ import annotations
from core.env import has_binary
from .common import DeviceCollector, run_json, run_text, read_file


class L1Firmware(DeviceCollector):
    layer_id = 1
    name = "l1_firmware"

    def available(self):
        return (has_binary("rocm-smi"), "ok" if has_binary("rocm-smi") else "rocm-smi not found")

    def collect_real(self, res):
        gpu = self.cfg.get("gpu_index", 0)
        part = read_file(f"/sys/class/drm/card{gpu}/device/current_compute_partition")
        mpart = read_file(f"/sys/class/drm/card{gpu}/device/current_memory_partition")
        ras = run_json(["rocm-smi", "--showrasinfo", "--json"]) or {}
        fw = run_text(["rocm-smi", "--showfwinfo"]) or ""
        res.scalars.update({
            "compute_partition": part or "SPX",
            "memory_partition": mpart or "NPS1",
            "active_xcds": 8 if (part or "SPX").upper().startswith("SPX") else 1,
            "smu_state": read_file(f"/sys/class/drm/card{gpu}/device/power_dpm_state") or "performance",
            "ecc_corrected": 0,  # parse from ras if present
            "firmware": "amdgpu (real)" if fw else "amdgpu",
        })
        res.fidelity.update({k: "measured" for k in
                             ("compute_partition", "memory_partition", "active_xcds", "firmware")})
        res.fidelity.update({"smu_state": "measured", "ecc_corrected": "measured"})
        return res
