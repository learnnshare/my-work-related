"""rccl_log.py — parse rocBLAS / MIOpen / RCCL logs for L4 library metrics."""
from __future__ import annotations
import re
from pathlib import Path


def _read(path):
    try:
        return Path(path).read_text(errors="replace") if path else ""
    except OSError:
        return ""


def parse_lib_logs(logs: dict):
    rocblas = _read(logs.get("rocblas"))
    miopen = _read(logs.get("miopen"))
    rccl = _read(logs.get("rccl"))
    sc, fd = {}, {}

    # rocBLAS / hipBLASLt: library + autotune variant from solution names
    if rocblas:
        sc["library"] = "hipBLASLt" if "hipblaslt" in rocblas.lower() else "rocBLAS"
        fd["library"] = "measured"
        m = re.search(r"(MFMA[_ ]?\d+x\d+|split[-_]?K|GSU\d+)", rocblas, re.I)
        if m:
            sc["autotune_variant"] = m.group(1); fd["autotune_variant"] = "measured"

    # MIOpen kernel cache hit rate
    if miopen:
        hits = len(re.findall(r"cache hit", miopen, re.I))
        miss = len(re.findall(r"cache miss", miopen, re.I))
        if hits + miss:
            sc["kernel_cache_hit_pct"] = round(100 * hits / (hits + miss), 1)
            fd["kernel_cache_hit_pct"] = "measured"

    # RCCL bus bandwidth (busbw GB/s)
    if rccl:
        m = re.search(r"busbw[:\s]+([\d.]+)", rccl, re.I)
        if m:
            sc["rccl_gbs"] = float(m.group(1)); fd["rccl_gbs"] = "measured"
    return sc, fd
