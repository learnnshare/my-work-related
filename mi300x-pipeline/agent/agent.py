#!/usr/bin/env python3
"""
agent.py — hardware-grounded bottleneck agent for MI300X records.

Reads a normalized L0–L7 record (+ optional predictor signal), and returns a
per-layer **bottleneck attribution + concrete optimization actions**, grounded in
the record's OWN measured numbers (the "reliability via grounding" story).

Two backends:
  - rule-based (deterministic, offline) — always available; thresholds on real metrics
  - live LLM (Anthropic Claude) — used when ANTHROPIC_API_KEY is set; falls back to
    rule-based on any error so the demo never breaks.

CLI:
    python agent.py <record.json> [--no-llm]
    python agent.py --all [--data ../mi300x-dashboard/data] [--no-llm]   # analyze all, write reports.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KNOWLEDGE = (HERE / "knowledge.md")


# ---------- record helpers ----------
def _layers(record):
    """{layer_id: {metric_k: value}}"""
    out = {}
    for layer in record.get("layers", []):
        out[layer["id"]] = {m["k"]: m.get("v") for m in layer.get("metrics", [])}
    return out


def _num(x):
    return x if isinstance(x, (int, float)) else None


# ---------- rule-based engine (grounded thresholds) ----------
def _rules(record, prediction=None):
    m = record.get("metrics", {})
    L = _layers(record)
    wl = record.get("meta", {}).get("workload", {})
    wid = (wl.get("id") or "").lower()
    prec = (wl.get("precision") or wl.get("pref") or "").lower()
    findings, facts, notes = [], [], []

    cu = _num(m.get("computeUtil"))
    mu = _num(m.get("memUtil"))
    tf = _num(m.get("achievedTflops"))
    peak = _num(m.get("peakTflops"))
    bound = m.get("boundBy")
    l0, l4, l5 = L.get(0, {}), L.get(4, {}), L.get(5, {})
    mfma = _num(l0.get("MFMA / matrix-core util"))
    hbm = _num(l0.get("HBM3 bandwidth util"))
    occ = _num(l0.get("VGPR occupancy"))
    active_cus = _num(l0.get("Active CUs"))
    cache_hit = _num(l4.get("Kernel cache hit"))
    host_ov = _num(l5.get("Host overhead"))
    launch_ov = _num(l5.get("Launch overhead"))
    batch = _num(L.get(6, {}).get("Batch size"))

    def add(layer, sev, obs, rec):
        findings.append({"layer": layer, "layerName": f"L{layer}", "severity": sev,
                         "observation": obs, "recommendation": rec})

    # 1) matrix cores idle on a compute/GEMM workload
    if "gemm" in wid and mfma is not None and mfma < 5 and cu is not None and cu < 0.3:
        facts.append(f"L0 MFMA util {mfma}%, computeUtil {round(cu*100,1)}%")
        add(0, "high",
            f"Matrix cores idle (MFMA {mfma}%) on a GEMM — running on VALU, not MFMA.",
            "Use fp32-accumulate GEMM (gemm_ex / hipBLASLt HHS) to engage the matrix cores; "
            "pure fp16-accumulate hgemm does not use MFMA on gfx942.")

    # 2) memory-bound
    if (bound == "memory") or (cu is not None and mu is not None and mu > cu):
        facts.append(f"memUtil {round((mu or 0)*100,1)}% > computeUtil {round((cu or 0)*100,1)}%"
                     + (f", HBM util {hbm}%" if hbm is not None else ""))
        add(0, "warn",
            f"Memory-bound: HBM traffic dominates (memUtil {round((mu or 0)*100,1)}%).",
            "Raise arithmetic intensity — tile/block for L2 reuse, fuse elementwise ops, "
            "or increase batch so each byte does more FLOPs.")

    # 3) compute-bound: headroom vs well-utilized → precision/tile advice
    if bound == "compute" and cu is not None:
        head = f"{round(cu*100,1)}% of peak"
        facts.append(f"computeUtil {head}" + (f", {tf}/{peak} TFLOPS" if tf and peak else ""))
        fp8_rec = " Try fp8 (hipBLASLt) — measured ~1.7× fp16 on this device." if prec in ("fp16", "bf16") else ""
        if cu < 0.6:
            add(4, "info", f"Compute-bound with headroom ({head}).",
                "Larger tiles / problem size improve MFMA efficiency." + fp8_rec)
        else:
            add(4, "info", f"Well-utilized: {head} of matrix-core peak ({tf} TFLOPS).",
                ("Near the roofline; further gains need lower precision or structured sparsity." + fp8_rec)
                if fp8_rec else "Near the roofline — efficient use of the matrix cores.")

    # 4) launch / host bound
    if host_ov is not None and host_ov > 30:
        facts.append(f"L5 host overhead {host_ov}%")
        add(5, "warn", f"Host/CPU-bound: {host_ov}% of time outside GPU kernels.",
            "Use HIP graphs to amortize launch, fuse kernels, or move host work off the critical path.")
    if launch_ov is not None and launch_ov > 30:
        facts.append(f"L5 launch overhead {launch_ov}%")
        add(3, "warn", f"Launch-bound: {launch_ov}% kernel-launch overhead (many tiny dispatches).",
            "Batch/fuse kernels or capture a HIP graph to cut per-dispatch cost.")

    # 5) over-provisioned at batch-1 (robotics)
    if batch == 1 and occ is not None and occ < 20:
        facts.append(f"batch 1, VGPR occupancy {occ}%"
                     + (f", {active_cus}/304 active CUs" if active_cus is not None else ""))
        add(1, "info", "GPU is over-provisioned for this batch-1 workload (very low occupancy).",
            "For single-robot inference, consider a CPX/NPS partition (slice the GPU) or batch "
            "multiple agents — a full MI300X is wasted here.")

    # 6) poor cache reuse while memory-bound
    if cache_hit is not None and cache_hit < 40 and (bound == "memory"):
        facts.append(f"L4 cache hit {cache_hit}%")
        add(4, "warn", f"Low L2 reuse (cache hit {cache_hit}%) is feeding the memory bound.",
            "Block/tile to fit the working set in L2 (4 MB) / Infinity Cache (256 MB).")

    # 7) sensor gap (honesty note — not a bottleneck, kept out of the headline)
    if l0.get("Board power") is None:
        notes.append("Power/temp unavailable on this SR-IOV VF (sensors not exposed); "
                     "use a bare-metal/PF node for power-efficiency metrics.")

    # prediction signal
    if prediction and isinstance(prediction, dict):
        wp = prediction.get("withinPct")
        me = prediction.get("meanErrPct")
        if wp is not None:
            facts.append(f"predictor: {wp}% of metrics within ±20% (mean err {me}%)")

    # headline
    sev_rank = {"high": 3, "warn": 2, "info": 1}
    findings.sort(key=lambda f: sev_rank.get(f["severity"], 0), reverse=True)
    headline = findings[0] if findings else None
    bottleneck = (f"{bound}-bound" if bound else "n/a")
    if headline and headline["severity"] == "high":
        bottleneck = headline["observation"]
    summary = (f"{wl.get('name', wid)} [{record.get('meta', {}).get('source', '?')}"
               + (f", {prec}" if prec else "") + f"]: {bottleneck}.")
    if headline:
        summary += " " + headline["recommendation"]

    return {
        "workload": wl.get("name", wid),
        "source": record.get("meta", {}).get("source"),
        "precision": prec,
        "summary": summary,
        "bottleneck": bottleneck,
        "findings": findings,
        "recommendations": [f["recommendation"] for f in findings],
        "grounded_facts": facts,
        "notes": notes,
        "generated_by": "rule",
    }


# ---------- live LLM path (optional) ----------
def _grounded_prompt(record, prediction):
    m = record.get("metrics", {})
    L = _layers(record)
    knowledge = KNOWLEDGE.read_text() if KNOWLEDGE.exists() else ""
    layers_txt = []
    for lid in sorted(L):
        kv = ", ".join(f"{k}={v}" for k, v in L[lid].items() if v is not None)
        if kv:
            layers_txt.append(f"L{lid}: {kv}")
    pred_txt = json.dumps(prediction) if prediction else "none"
    return f"""You are a hardware performance engineer for the AMD Instinct MI300X (gfx942).
Use ONLY the measured numbers below — cite them. Do not invent values.

=== MI300X knowledge ===
{knowledge}

=== Measured record ({record.get('meta',{}).get('source')}) ===
workload: {m.get('throughputUnit','')} | precision: {record.get('meta',{}).get('workload',{}).get('precision')}
flat: e2eMs={m.get('e2eMs')} throughput={m.get('throughput')} achievedTflops={m.get('achievedTflops')}/{m.get('peakTflops')} computeUtil={m.get('computeUtil')} memUtil={m.get('memUtil')} boundBy={m.get('boundBy')}
{chr(10).join(layers_txt)}
predictor: {pred_txt}

Return: (1) the primary bottleneck and WHICH layer, (2) 2–4 concrete optimization actions,
each citing the specific measured number that motivates it. Be concise and specific to this device."""


def _llm(record, prediction, model=None):
    model = model or os.environ.get("AGENT_MODEL", "claude-fable-5")
    try:
        import anthropic  # noqa
        client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY
        msg = client.messages.create(
            model=model, max_tokens=900,
            messages=[{"role": "user", "content": _grounded_prompt(record, prediction)}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        rep = _rules(record, prediction)        # keep structured findings from rules
        rep["summary"] = text.strip()
        rep["generated_by"] = f"llm:{model}"
        return rep
    except Exception as e:
        rep = _rules(record, prediction)
        rep["llm_error"] = str(e)
        return rep


def analyze(record, prediction=None, use_llm=None):
    if use_llm is None:
        use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return _llm(record, prediction, ) if use_llm else _rules(record, prediction)


def analyze_all(data_dir, use_llm=False):
    """Latest record per (workload,prec,batch) -> {key: report}. For dashboard baking."""
    data_dir = Path(data_dir)
    recs = {}
    for f in sorted((data_dir / "records").glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        wl = r["meta"]["workload"]
        key = f"{wl.get('id')}|{wl.get('precision') or wl.get('pref')}|{wl.get('batch')}|{r['meta'].get('source')}"
        recs[key] = r
    reports = {}
    for key, r in recs.items():
        reports[key] = analyze(r, use_llm=use_llm)
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record", nargs="?", help="path to a normalized record JSON")
    ap.add_argument("--all", action="store_true", help="analyze all records in --data")
    ap.add_argument("--data", default="../mi300x-dashboard/data")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    use_llm = (not args.no_llm) and bool(os.environ.get("ANTHROPIC_API_KEY"))

    if args.all:
        reports = analyze_all((HERE / ".." / args.data).resolve(), use_llm=use_llm)
        out = (HERE / ".." / args.data / "agent_reports.json").resolve()
        out.write_text(json.dumps(reports, indent=2))
        print(f"[agent] wrote {len(reports)} reports → {out}  (llm={use_llm})")
        for k, r in reports.items():
            print(f"\n● {k}\n  {r['summary'][:200]}")
        return
    if not args.record:
        ap.error("give a record path or --all")
    rep = analyze(json.loads(Path(args.record).read_text()), use_llm=use_llm)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
