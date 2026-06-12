// gemm.cpp — rocBLAS GEMM (4096^3), fp16 inputs with fp32 accumulate via
// rocblas_gemm_ex (HHS). The fp32-accumulate path is what engages the MI300X
// matrix cores (MFMA) — a compute-bound contrast to the memory-bound vectoradd.
// Build: hipcc gemm.cpp -o gemm -lrocblas -I/opt/rocm/include -L/opt/rocm/lib
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>

#define CK(x) do { hipError_t e=(x); if(e){printf("HIP err %d %s @%d\n",e,hipGetErrorString(e),__LINE__); return 2;} } while(0)
#define RB(x) do { rocblas_status s=(x); if(s!=rocblas_status_success){printf("rocBLAS err %d @%d\n",s,__LINE__); return 3;} } while(0)

int main() {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) { printf("NO HIP DEVICE\n"); return 4; }
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
    printf("device: %s  %s  CUs=%d\n", p.name, p.gcnArchName, p.multiProcessorCount);

    const int m = 4096, n = 4096, k = 4096, iters = 50;
    size_t szA = (size_t)m * k, szB = (size_t)k * n, szC = (size_t)m * n;

    rocblas_handle h; RB(rocblas_create_handle(&h));
    rocblas_half *A, *B, *C;
    CK(hipMalloc(&A, szA * sizeof(rocblas_half)));
    CK(hipMalloc(&B, szB * sizeof(rocblas_half)));
    CK(hipMalloc(&C, szC * sizeof(rocblas_half)));
    CK(hipMemset(A, 1, szA * sizeof(rocblas_half)));
    CK(hipMemset(B, 1, szB * sizeof(rocblas_half)));

    float alpha = 1.0f, beta = 0.0f;          // f32 compute (HHS) -> MFMA path
    auto gemm = [&]() {
        return rocblas_gemm_ex(h, rocblas_operation_none, rocblas_operation_none,
            m, n, k, &alpha,
            A, rocblas_datatype_f16_r, m,
            B, rocblas_datatype_f16_r, k, &beta,
            C, rocblas_datatype_f16_r, m,     // C (input)
            C, rocblas_datatype_f16_r, m,     // D (output, in-place)
            rocblas_datatype_f32_r,           // compute type -> fp32 accumulate
            rocblas_gemm_algo_standard, 0, 0);
    };

    RB(gemm()); CK(hipDeviceSynchronize());   // warmup (pick/cache kernel)
    for (int i = 0; i < iters; ++i) RB(gemm());
    CK(hipDeviceSynchronize());
    printf("OK ran %d gemm_ex(HHS) %dx%dx%d\n", iters, m, n, k);

    rocblas_destroy_handle(h);
    hipFree(A); hipFree(B); hipFree(C);
    return 0;
}
