"""L6 Application — app-reported per-iteration JSONL (e2e/throughput)."""
from __future__ import annotations
import json
from pathlib import Path
from .common import DeviceCollector
from normalize.reduce import p


class L6Application(DeviceCollector):
    layer_id = 6
    name = "l6_application"

    def env_overrides(self):
        # Workload writes one JSON line per iteration: {"iter_ms":..,"items":..}
        return {"MI300X_APP_JSONL": str(self.run_ctx["raw_dir"] / "l6" / "app.jsonl")}

    def collect_real(self, res):
        path = self.run_ctx.get("app_jsonl") or (self.run_ctx["raw_dir"] / "l6" / "app.jsonl")
        rows = []
        try:
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        except OSError:
            return super().collect_real(res)
        if not rows:
            return super().collect_real(res)
        lat = [r["iter_ms"] for r in rows if "iter_ms" in r]
        items = sum(r.get("items", 1) for r in rows)
        total_s = sum(lat) / 1000.0 if lat else None
        res.scalars.update({
            "e2e_ms": p(lat, 50) if lat else None,
            "latency_p50": p(lat, 50) if lat else None,
            "latency_p99": p(lat, 99) if lat else None,
            "throughput": (items / total_s) if total_s else None,
            "batch": self.run_ctx.get("workload", {}).get("batch"),
            "num_gpus": self.run_ctx.get("workload", {}).get("num_gpus", 1),
        })
        res.fidelity.update({k: "measured" for k in
                             ("e2e_ms", "latency_p50", "latency_p99", "throughput")})
        return res
