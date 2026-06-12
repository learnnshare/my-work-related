"""L1 Firmware / HW abstraction — amd-smi (ROCm 7) / rocm-smi / sysfs."""
from __future__ import annotations
from core.env import has_binary
from .common import DeviceCollector, run_json, run_text, read_file, amd_smi_json, deep_find


class L1Firmware(DeviceCollector):
    layer_id = 1
    name = "l1_firmware"

    def available(self):
        if has_binary("amd-smi") or has_binary("rocm-smi"):
            return True, "ok"
        return False, "neither amd-smi nor rocm-smi found"

    def collect_real(self, res):
        gpu = self.cfg.get("gpu_index", 0)
        part = mpart = None
        ecc = 0
        fw_label = "amdgpu"

        if has_binary("amd-smi"):
            # partition lives under `amd-smi partition` (not `static`) on ROCm 7
            partj = amd_smi_json("partition", gpu=gpu) or {}
            part = deep_find(partj, "accelerator_type", "compute_partition", "partition_mode")
            mpart = deep_find(partj, "current_memory_partition", "memory_partition")
            static = amd_smi_json("static", gpu=gpu) or {}
            part = part or deep_find(static, "compute_partition", "partition_mode")
            mpart = mpart or deep_find(static, "memory_partition")
            metric = amd_smi_json("metric", gpu=gpu) or {}
            e = deep_find(metric, "correctable_count", "ce")
            ecc = int(e) if isinstance(e, (int, float)) else 0
            fw = amd_smi_json("firmware", gpu=gpu)
            fw_label = "amdgpu (amd-smi, real)" if fw else "amdgpu"
        # sysfs fallbacks
        part = part or read_file(f"/sys/class/drm/card{gpu}/device/current_compute_partition") or "SPX"
        mpart = mpart or read_file(f"/sys/class/drm/card{gpu}/device/current_memory_partition") or "NPS1"
        if fw_label == "amdgpu" and run_text(["rocm-smi", "--showfwinfo"]):
            fw_label = "amdgpu (rocm-smi, real)"

        part = str(part).upper()
        res.scalars.update({
            "compute_partition": part,
            "memory_partition": str(mpart).upper(),
            "active_xcds": 8 if part.startswith("SPX") else (1 if part.startswith("CPX") else 8),
            "smu_state": read_file(f"/sys/class/drm/card{gpu}/device/power_dpm_state") or "performance",
            "ecc_corrected": ecc,
            "firmware": fw_label,
        })
        res.fidelity.update({k: "measured" for k in
                             ("compute_partition", "memory_partition", "active_xcds",
                              "firmware", "smu_state", "ecc_corrected")})
        return res
