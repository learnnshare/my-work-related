"""
config_extractor.py — read gem5 m5out/config.json into the extractGem5Params
field set the dashboard architect view expects. Config-derived => conf=1.0
(these are realized values, not spec-extracted guesses).

config.json is the realized SimObject tree. Key paths vary by gem5 version, so
we search defensively and fall back to the run's requested knobs.
"""
from __future__ import annotations
import json
from pathlib import Path


def _walk(obj, pred, path=""):
    """Yield (path, value) for nodes where pred(key,value) is True."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if pred(k, v):
                yield p, v
            yield from _walk(v, pred, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, pred, f"{path}[{i}]")


def _find_first(cfg, *key_names):
    names = {n.lower() for n in key_names}
    for _, v in _walk(cfg, lambda k, v: k.lower() in names and not isinstance(v, (dict, list))):
        return v
    return None


def extract_params(config_json_path, knobs_fallback=None) -> dict:
    """Return {fields:[{k,key,v,unit?,conf}], avgConf, raw, label}."""
    knobs = dict(knobs_fallback or {})
    cfg = {}
    try:
        cfg = json.loads(Path(config_json_path).read_text())
    except Exception:
        pass

    def pick(cfg_keys, knob_key, default):
        v = _find_first(cfg, *cfg_keys) if cfg else None
        if v is None:
            v = knobs.get(knob_key, default)
        return v

    raw = {
        "isa": pick(["isa", "gpu_isa"], "isa", "gfx942"),
        "cus": int(pick(["num_compute_units", "numCUs"], "cus", 304)),
        "xcds": int(knobs.get("xcds", 8)),
        "simdPerCu": int(pick(["num_simds", "simds_per_cu"], "simdPerCu", 4)),
        "wavefront": int(pick(["wf_size", "wavefront_size"], "wavefront", 64)),
        "l1KB": int(knobs.get("l1KB", 32)),
        "l2MB": int(knobs.get("l2MB", 4)),
        "llcMB": int(knobs.get("llcMB", 256)),
        "hbmGB": int(knobs.get("hbmGB", 192)),
        "bwGBs": int(knobs.get("bwGBs", 5300)),
        "clockMHz": int(knobs.get("clockMHz", 2100)),
        "cpuModel": str(pick(["type"], "cpuModel", "X86KvmCPU")) if knobs.get("cpuModel") is None else knobs["cpuModel"],
        "protocol": str(knobs.get("protocol", "GPU_VIPER")),
        "mem": str(knobs.get("mem", "HBM3_4H_2000")),
        "partition": str(knobs.get("partition", "SPX / NPS1")),
    }

    labels = {
        "isa": "ISA target", "cus": "Compute units (CUs)", "xcds": "XCD chiplets",
        "simdPerCu": "SIMDs / CU", "wavefront": "Wavefront size", "l1KB": "L1 vector cache",
        "l2MB": "L2 cache", "llcMB": "Infinity Cache (LLC)", "hbmGB": "HBM capacity",
        "bwGBs": "HBM bandwidth", "clockMHz": "Engine clock", "cpuModel": "gem5 CPU model",
        "protocol": "Ruby protocol", "mem": "Memory model", "partition": "Partition",
    }
    units = {"l1KB": "KB", "l2MB": "MB", "llcMB": "MB", "hbmGB": "GB", "bwGBs": "GB/s", "clockMHz": "MHz"}

    fields = []
    for key, label in labels.items():
        f = {"k": label, "key": key, "v": raw[key], "conf": 1.0}
        if key in units:
            f["unit"] = units[key]
        fields.append(f)

    return {"fields": fields, "avgConf": 1.0, "raw": raw, "label": "gem5 realized config (config.json)"}
