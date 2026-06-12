"""L7 Task / control loop — robotics control-cycle JSONL (sense/infer/act)."""
from __future__ import annotations
import json
from pathlib import Path
from .common import DeviceCollector
from normalize.reduce import p
from statistics import mean


class L7Task(DeviceCollector):
    layer_id = 7
    name = "l7_task"

    def env_overrides(self):
        # control-loop wrapper writes one JSON line per cycle:
        # {"sense_ms":..,"infer_ms":..,"act_ms":..,"reward":..}
        return {"MI300X_CTRL_JSONL": str(self.run_ctx["raw_dir"] / "l7" / "control.jsonl")}

    def collect_real(self, res):
        path = self.run_ctx.get("control_jsonl") or (self.run_ctx["raw_dir"] / "l7" / "control.jsonl")
        rows = []
        try:
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        except OSError:
            return super().collect_real(res)
        if not rows:
            return super().collect_real(res)
        cyc = [(r.get("sense_ms", 0) + r.get("infer_ms", 0) + r.get("act_ms", 0)) for r in rows]
        target_hz = self.run_ctx.get("workload", {}).get("target_hz")
        deadline = (1000.0 / target_hz) if target_hz else None
        adherence = (100.0 * sum(1 for c in cyc if deadline and c <= deadline) / len(cyc)) if (deadline and cyc) else None
        res.scalars.update({
            "control_hz": round(1000.0 / mean(cyc)) if cyc and mean(cyc) else None,
            "deadline_adherence_pct": round(adherence, 1) if adherence is not None else None,
            "cycle_jitter_ms": round((p(cyc, 99) - p(cyc, 50)), 3) if cyc else None,
            "sense_ms": round(mean([r.get("sense_ms", 0) for r in rows]), 3),
            "infer_ms": round(mean([r.get("infer_ms", 0) for r in rows]), 3),
            "act_ms": round(mean([r.get("act_ms", 0) for r in rows]), 3),
            "episode_reward": round(mean([r["reward"] for r in rows if "reward" in r]), 1) if any("reward" in r for r in rows) else None,
            "sim2real_err_pct": None,  # filled by predictor comparison
        })
        res.fidelity.update({k: "measured" for k in
                             ("control_hz", "deadline_adherence_pct", "cycle_jitter_ms",
                              "sense_ms", "infer_ms", "act_ms")})
        res.fidelity["sim2real_err_pct"] = "null"
        return res
