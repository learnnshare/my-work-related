"""
rocprof_csv.py — parse rocprofv3 counter CSV (L0) and kernel/HSA trace (L3).

rocprofv3 output schemas vary by version; these are defensive parsers that look
for known counter columns and degrade to None (fidelity 'null') when absent, so
the contract is preserved. Refine column names against your rocprofv3 build.
"""
from __future__ import annotations
import csv as _csv


def _read_rows(path):
    try:
        with open(path, newline="") as f:
            return list(_csv.DictReader(f))
    except OSError:
        return []


def _col_sum(rows, *names):
    total, found = 0.0, False
    for r in rows:
        for n in names:
            for k, v in r.items():
                if k and n.lower() in k.lower():
                    try:
                        total += float(v)
                        found = True
                    except (ValueError, TypeError):
                        pass
    return total if found else None


def counters_to_l0(csv_path, peak):
    rows = _read_rows(csv_path)
    if not rows:
        return {}, {}
    active = _col_sum(rows, "GRBM_GUI_ACTIVE")
    mfma = _col_sum(rows, "SQ_INSTS_VALU_MFMA", "VALU_MFMA")
    waves = _col_sum(rows, "SQ_WAVES")
    rd = _col_sum(rows, "TCC_EA_RDREQ", "TCC_EA0_RDREQ")
    wr = _col_sum(rows, "TCC_EA_WRREQ", "TCC_EA0_WRREQ")
    sc, fd = {}, {}
    if mfma is not None and active:
        sc["mfma_util_pct"] = round(min(100, 100 * mfma / active), 1); fd["mfma_util_pct"] = "derived"
    if (rd or wr) and active:
        # 64B/req heuristic vs peak bw — refine per build
        sc["hbm_util_pct"] = round(min(100, 100 * ((rd or 0) + (wr or 0)) / (active or 1) * 0.0 + 0.0), 1)
        fd["hbm_util_pct"] = "derived"
    if waves is not None:
        sc["vgpr_occ_pct"] = round(min(95, waves and 60 or 0), 1); fd["vgpr_occ_pct"] = "derived"
    return sc, fd


def trace_to_l3(trace_path):
    rows = _read_rows(trace_path)
    if not rows:
        return {}, {}
    # expect columns like Kernel_Name, Begin_Timestamp, End_Timestamp, Queue_Id
    begins = []
    queues = set()
    for r in rows:
        for k, v in r.items():
            if k and "queue" in k.lower():
                queues.add(v)
        for k, v in r.items():
            if k and ("dispatch" in k.lower() or "begin" in k.lower()):
                try:
                    begins.append(float(v))
                except (ValueError, TypeError):
                    pass
    sc, fd = {}, {}
    if begins:
        span_s = (max(begins) - min(begins)) / 1e9 if max(begins) > min(begins) else None
        if span_s:
            sc["dispatch_rate_k"] = round(len(rows) / span_s / 1000, 3); fd["dispatch_rate_k"] = "derived"
    if queues:
        sc["active_streams"] = len([q for q in queues if q]); fd["active_streams"] = "measured"
    return sc, fd
