// gemm.cpp — rocBLAS half-precision GEMM (4096^3) to exercise the MI300X matrix
// cores (MFMA). Compute-bound contrast to the memory-bound vectoradd.
// Build: hipcc gemm.cpp -o gemm -lrocblas -I/opt/rocm/include -L/opt/rocm/lib
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstring>
#include <cstdint>

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

    rocblas_half alpha, beta;
    uint16_t one = 0x3C00, zero = 0x0000;     // 1.0 and 0.0 in IEEE half
    memcpy(&alpha, &one, 2); memcpy(&beta, &zero, 2);

    // warmup (lets rocBLAS pick/cache a kernel) then timed loop
    RB(rocblas_hgemm(h, rocblas_operation_none, rocblas_operation_none,
                     m, n, k, &alpha, A, m, B, k, &beta, C, m));
    CK(hipDeviceSynchronize());
    for (int i = 0; i < iters; ++i)
        RB(rocblas_hgemm(h, rocblas_operation_none, rocblas_operation_none,
                         m, n, k, &alpha, A, m, B, k, &beta, C, m));
    CK(hipDeviceSynchronize());
    printf("OK ran %d hgemm %dx%dx%d\n", iters, m, n, k);

    rocblas_destroy_handle(h);
    hipFree(A); hipFree(B); hipFree(C);
    return 0;
}
