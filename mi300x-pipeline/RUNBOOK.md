# RUNBOOK — operating the MI300X / gem5 metrics pipeline

End-to-end operation for both data sources, with verification gates `[✓]`.

## 0. Local demo (no hardware) — proves the data flow

```bash
cd mi300x-pipeline
python3 orchestrator.py --config pipeline.yaml         # mode: demo
```
- `[✓]` prints "contract OK for N records"
- `[✓]` writes `../mi300x-dashboard/data/bundle.js` + `records/` + `predictions/`
- `[✓]` open `mi300x-dashboard/developer.html` → layers L0–L7 render from real data
        (console: `[data.js] MI300X_DATA loaded`)
- `[✓]` `rm -rf mi300x-dashboard/data` → pages still render via sim.js fallback

## 1. Real MI300X (Ubuntu) — `mode: device`

Prereqs: ROCm 6.x; user in `render,video` groups. Privileged collectors
(L0 counters, L2 ftrace) need root/`CAP_SYS_ADMIN` — without them the run
degrades (collector marked `skipped` in the manifest), it does not fail.

```bash
sudo usermod -aG render,video $USER     # once; re-login
# enable privileged layers if you have root:
sudo python3 orchestrator.py --config pipeline.device.yaml
```
Per-layer tooling: L0 `rocprofv3` HW counters + `rocm-smi`; L1 `rocm-smi`
partitions/ECC/firmware; L2 ftrace amdgpu/KFD + `/proc/interrupts`; L3 `rocprofv3
--hip-trace --hsa-trace --kernel-trace`; L4 `ROCBLAS_LAYER`/`MIOPEN_*`/`NCCL_DEBUG`
logs; L5 `torch.profiler` (workload wraps its model and writes the Kineto trace to
`$MI300X_KINETO_OUT`); L6 app writes per-iter JSONL to `$MI300X_APP_JSONL`; L7
control loop writes per-cycle JSONL to `$MI300X_CTRL_JSONL`.
- `[✓]` device records have non-null `powerW`, `tempC`, `latencyP99`.

## 2. gem5 — `mode: gem5`

Prereqs: a built gem5 with the GPU model, e.g.
`scons build/VEGA_X86/gem5.opt` (use the AMD GPU docker image from
gem5-resources — the ROCm/kernel toolchain is fussy). Start in **GPUSE** to
validate a microkernel, then **GPUFS** (needs KVM on an x86 host) for the real
driver/runtime layers.

```bash
# point pipeline.yaml gem5.binary at your gem5.opt, set knobs, run:
python3 orchestrator.py --config pipeline.yaml         # mode: gem5
```
Wire `collectors/gem5/configs/gpu_mi300x.py` to your gem5 tree's GPUFS/GPUSE
builders, and bracket the kernel of interest with `m5.workbegin()/workend()` so
`stats.txt` isolates the kernel region.
- `[✓]` `stats.txt` contains the kernel region; `config.json` present.
- `[✓]` gem5 records carry measured Ruby cache-hit rate (L4) and exact HBM bytes;
        power/thermal and `latencyP99` are `null` (gem5 can't observe them).

## 3. Predict + publish (demo wires this automatically)

The predictor joins gem5/sim **features** to device **labels** on
`(workload, precision, batch, gpus)`, trains, and writes both sinks:
file (`runs/pred_<wl>.json`) and dashboard (`data/predictions/<wl>.json`,
predictionSet shape). Swap the model by implementing the `Predictor` protocol in
`predict/predictor.py` (`train`/`predict`) and setting `predict.predictor`.
- `[✓]` `predictions/<wl>.json` has `pairs[*].within`, `withinPct`, `targetPct:20`.

## 4. Contract guard (run before trusting any publish)

```bash
python3 -c "import json,glob,sys; sys.path.insert(0,'.'); from normalize.schema import assert_contract; \
[print(f, assert_contract(json.load(open(f))) or 'OK') for f in glob.glob('../mi300x-dashboard/data/records/*.json')]"
```
- `[✓]` every record prints `OK` (keys ⊆ the sim.js contract, layers 0..7).

## Notes
- No KVM needed for device mode (bare metal). KVM only for gem5 GPUFS boot.
- `file://` + `fetch` is unreliable; the dashboard uses `data/bundle.js`
  (`window.MI300X_DATA`) so it works offline. Optional: `python -m http.server`
  inside `mi300x-dashboard/` to serve it.
