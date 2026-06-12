# MI300X captured results (real hardware)

| workload | precision | size | TFLOPS | bandwidth_TBs | MFMA_pct | L2_hit_pct | bound_by | e2e_ms | source |
|---|---|---|---|---|---|---|---|---|---|
| GEMM bf16 2048^3 | bf16 | 2048 | 527.891 | 1.4477 | 40.4 | 67.6 | compute | 0.6834 | device |
| GEMM bf16 4096^3 | bf16 | 4096 | 730.42 | 1.7142 | 55.9 | 65.2 | compute | 3.9515 | device |
| GEMM bf16 8192^3 | bf16 | 8192 | 877.61 | 1.0299 | 67.1 | 72.9 | compute | 26.3098 | device |
| GEMM fp16 2048^3 | fp16 | 2048 | 527.395 | 1.4461 | 40.3 | 67.5 | compute | 0.6841 | device |
| GEMM fp16 4096^3 | fp16 | 4096 | 693.181 | 1.6061 | 53.0 | 65.6 | compute | 4.1637 | device |
| GEMM fp16 8192^3 | fp16 | 8192 | 851.419 | 1.0819 | 65.1 | 70.7 | compute | 27.1191 | device |
| PPO Policy Inference (batch-1 robot control) | bf16 | None | 0.3 | 0.64 | 0.0 | 70.0 | memory | 0.05 | gem5 |
| gemm (Cijk_Ailk_Bljk_HHS_BH_MT256x112x) | fp16 | None | 730.799 | 1.5943 | 55.9 | 67.6 | compute | 9.5914 | device |
| rocBLAS GEMM (8192³) | bf16 | None | 650.6 | 0.71 | 49.8 | 90.0 | compute | 1.7125 | gem5 |
| vectoradd (vadd(float const*, float const*, float*, int)) | fp16 | None | 41.07 | 1.9258 | 0.0 | 25.6 | memory | 0.6536 | device |
