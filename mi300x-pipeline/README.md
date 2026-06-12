# MI300X / gem5 Metrics Pipeline

Modular, two-mode capture → normalize → predict → publish pipeline that feeds the
`mi300x-dashboard`. It captures fine-grained metrics across the **full stack
(Layer 0 → Layer 7, incl. workload/task level)** from either a real AMD Instinct
**MI300X** (Ubuntu/ROCm) or **gem5**, normalizes both into one schema that matches
the dashboard's data contract byte-for-byte, and exposes hooks for a predictive
engine (sim/gem5 features → real-hardware estimate, ±20% target).

```
capture  ──►  raw artifacts  ──►  normalize  ──►  (predict)  ──►  publish  ──►  dashboard
 device                          one schema      hooks+baseline   bundle.js     loads real data
 gem5                            two producers   file+dashboard   /records      (sim.js = fallback)
```

## Quickstart (runs locally with bundled fixtures — no MI300X/gem5 needed)

```bash
cd mi300x-pipeline
python3 orchestrator.py --config pipeline.yaml
```

This runs `mode: demo`: it ingests gem5 fixtures (features) + device fixtures
(labels) for `gemm` and `ppo_infer`, validates every record against the dashboard
contract, trains the baseline predictor, builds predicted-vs-measured
`predictionSet`s, and writes `../mi300x-dashboard/data/bundle.js`. Open any
dashboard page and it now shows **real captured data**; delete `data/` and it
falls back to the simulator.

## Modes (set `mode:` in pipeline.yaml)

| Mode | What runs | Output |
|---|---|---|
| `gem5` | gem5 (or m5out fixtures) → gem5 layer mapping | normalized `gem5` records |
| `device` | device collectors L0–L7 (real ROCm tools, or fixtures) | normalized `device` records |
| `demo` | both from fixtures + predictor + publish | bundle.js for the dashboard |

## Layout

```
core/         interface (BaseCollector), manifest, sampling, env preflight
collectors/
  device/     L0..L7 collectors (rocprofv3, rocm-smi, ftrace, torch.profiler, app/control JSONL)
  gem5/       run_gem5, stats_parser, config_extractor, map_layers (fidelity-tagged), gpu config template
parsers/      device raw → canonical scalars (rocprof_csv, ftrace, kineto, rccl_log, rocm_smi_json)
normalize/    schema (+contract guard), layer_map (single source of truth), reduce, normalizer
predict/      featurize, predictor (Protocol + baseline), sinks (file + dashboard predictionSet)
publish/      to_dashboard (bundle.js + records/ + predictions/ + index.json)
workloads.py  workload presets (mirror of dashboard WORKLOADS)
fixtures/     gem5 stats.txt/config.json + device scalar JSON for offline demo
orchestrator.py, pipeline.yaml
```

## The contract

`normalize/layer_map.py` is the single source of truth: the L0–L7 metric keys,
units, max, and fmt match `mi300x-dashboard/assets/sim.js` exactly. `normalize/
schema.py` guards every emitted record against that contract before publish, so
the dashboard renders real data with **zero chart/DOM changes**.

Fidelity tags (`_f ∈ measured|derived|synthetic|null`) mark where each source is
strong or blind — gem5 gives exact cache-hit rates and bytes (architectural
ground truth) but cannot observe power/thermal or a latency distribution; the
device path supplies those. Gaps are emitted as `null`, never fabricated.

See `RUNBOOK.md` for running on real MI300X hardware and a real gem5 build.
