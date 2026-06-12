// gemm_fp8.cpp — FP8 (E4M3 FNUZ) GEMM via hipBLASLt, the MI300X fp8 matrix path.
// rocBLAS on ROCm 7 dropped fp8, so fp8 lives in hipBLASLt.
//   usage: gemm_fp8 [M N K] [iters]
// Build: hipcc gemm_fp8.cpp -o gemm_fp8 -lhipblaslt -I/opt/rocm/include -L/opt/rocm/lib
//
// NOTE: untested at author time (no hipBLASLt locally). If it doesn't build or
// the heuristic returns 0, see the troubleshooting notes in INSTALLATION.md;
// the most likely tweak is the fp8 enum (E4M3_FNUZ vs E4M3) or the TN layout.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdio>
#include <cstdlib>

#define CK(x)  do { hipError_t e=(x); if(e){printf("HIP err %d %s @%d\n",e,hipGetErrorString(e),__LINE__); return 2;} } while(0)
#define LT(x)  do { hipblasStatus_t s=(x); if(s!=HIPBLAS_STATUS_SUCCESS){printf("hipBLASLt err %d @%d\n",(int)s,__LINE__); return 3;} } while(0)

int main(int argc, char** argv) {
    int m = 4096, n = 4096, k = 4096, iters = 50;
    if (argc > 3) { m = atoi(argv[1]); n = atoi(argv[2]); k = atoi(argv[3]); }
    if (argc > 4) iters = atoi(argv[4]);

    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) { printf("NO HIP DEVICE\n"); return 4; }
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
    printf("device: %s  %s  CUs=%d\n", p.name, p.gcnArchName, p.multiProcessorCount);

    const hipDataType IN  = HIP_R_8F_E4M3_FNUZ;   // gfx942 fp8
    const hipDataType OUT = HIP_R_16F;
    const hipblasComputeType_t COMP = HIPBLAS_COMPUTE_32F;

    // TN layout (transA=T) is the supported fp8 config on MI300:
    //   physical A: [k, m] (ld=k), B: [k, n] (ld=k), C/D: [m, n] (ld=m)
    void *A, *B, *D;
    CK(hipMalloc(&A, (size_t)m * k * 1));   // 1 byte / fp8 elem
    CK(hipMalloc(&B, (size_t)k * n * 1));
    CK(hipMalloc(&D, (size_t)m * n * 2));   // 2 bytes / f16 out
    CK(hipMemset(A, 0, (size_t)m * k));
    CK(hipMemset(B, 0, (size_t)k * n));
    void *C = D;

    hipblasLtHandle_t h; LT(hipblasLtCreate(&h));
    hipblasLtMatrixLayout_t lA, lB, lC, lD;
    LT(hipblasLtMatrixLayoutCreate(&lA, IN,  k, m, k));
    LT(hipblasLtMatrixLayoutCreate(&lB, IN,  k, n, k));
    LT(hipblasLtMatrixLayoutCreate(&lC, OUT, m, n, m));
    LT(hipblasLtMatrixLayoutCreate(&lD, OUT, m, n, m));

    hipblasLtMatmulDesc_t op;
    LT(hipblasLtMatmulDescCreate(&op, COMP, HIP_R_32F));
    hipblasOperation_t tA = HIPBLAS_OP_T, tB = HIPBLAS_OP_N;
    LT(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
    LT(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));

    uint64_t ws_size = 64ull * 1024 * 1024;
    void* ws; CK(hipMalloc(&ws, ws_size));
    hipblasLtMatmulPreference_t pref;
    LT(hipblasLtMatmulPreferenceCreate(&pref));
    LT(hipblasLtMatmulPreferenceSetAttribute(pref,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size)));

    hipblasLtMatmulHeuristicResult_t heur[1];
    int ret = 0;
    LT(hipblasLtMatmulAlgoGetHeuristic(h, op, lA, lB, lC, lD, pref, 1, heur, &ret));
    if (ret == 0) { printf("UNSUPPORTED fp8 gemm (no algorithm) %dx%dx%d\n", m, n, k); return 5; }

    float alpha = 1.0f, beta = 0.0f;
    auto run = [&]() {
        return hipblasLtMatmul(h, op, &alpha, A, lA, B, lB, &beta, C, lC, D, lD,
                               &heur[0].algo, ws, ws_size, 0);
    };
    LT(run()); CK(hipDeviceSynchronize());        // warmup
    for (int i = 0; i < iters; ++i) LT(run());
    CK(hipDeviceSynchronize());
    printf("OK ran %d hipblaslt fp8 gemm %dx%dx%d\n", iters, m, n, k);

    hipblasLtDestroy(h);
    hipFree(A); hipFree(B); hipFree(D); hipFree(ws);
    return 0;
}
