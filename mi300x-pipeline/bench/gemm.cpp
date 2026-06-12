// gemm.cpp — parameterized rocBLAS GEMM via rocblas_gemm_ex with fp32 accumulate
// (the matrix-core / MFMA path). Supports fp16, bf16, fp8 inputs.
//   usage: gemm [M N K] [precision] [iters]
//   e.g.:  gemm 4096 4096 4096 fp16 50     (defaults if omitted)
// Build: hipcc gemm.cpp -o gemm -lrocblas -I/opt/rocm/include -L/opt/rocm/lib
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <string>

#define CK(x) do { hipError_t e=(x); if(e){printf("HIP err %d %s @%d\n",e,hipGetErrorString(e),__LINE__); return 2;} } while(0)
#define RB(x) do { rocblas_status s=(x); if(s!=rocblas_status_success){printf("rocBLAS err %d @%d\n",(int)s,__LINE__); return 3;} } while(0)

static rocblas_datatype in_type(const std::string& p) {
    if (p == "bf16") return rocblas_datatype_bf16_r;
    if (p == "fp8")  return rocblas_datatype_f8_r;
    return rocblas_datatype_f16_r;
}
static rocblas_datatype out_type(const std::string& p) {
    if (p == "bf16") return rocblas_datatype_bf16_r;
    return rocblas_datatype_f16_r;        // fp8/fp16 inputs -> f16 output
}
static size_t in_bytes(const std::string& p) { return p == "fp8" ? 1 : 2; }
static size_t out_bytes(const std::string&) { return 2; }

int main(int argc, char** argv) {
    int m = 4096, n = 4096, k = 4096, iters = 50;
    std::string prec = "fp16";
    if (argc > 3) { m = atoi(argv[1]); n = atoi(argv[2]); k = atoi(argv[3]); }
    if (argc > 4) prec = argv[4];
    if (argc > 5) iters = atoi(argv[5]);

    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) { printf("NO HIP DEVICE\n"); return 4; }
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
    printf("device: %s  %s  CUs=%d\n", p.name, p.gcnArchName, p.multiProcessorCount);

    size_t szA = (size_t)m * k, szB = (size_t)k * n, szC = (size_t)m * n;
    rocblas_handle h; RB(rocblas_create_handle(&h));
    void *A, *B, *C;
    CK(hipMalloc(&A, szA * in_bytes(prec)));
    CK(hipMalloc(&B, szB * in_bytes(prec)));
    CK(hipMalloc(&C, szC * out_bytes(prec)));
    CK(hipMemset(A, 1, szA * in_bytes(prec)));
    CK(hipMemset(B, 1, szB * in_bytes(prec)));

    float alpha = 1.0f, beta = 0.0f;          // f32 compute (MFMA path)
    auto gemm = [&]() {
        return rocblas_gemm_ex(h, rocblas_operation_none, rocblas_operation_none,
            m, n, k, &alpha,
            A, in_type(prec), m,
            B, in_type(prec), k, &beta,
            C, out_type(prec), m,
            C, out_type(prec), m,
            rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard, 0, 0);
    };

    rocblas_status warm = gemm();
    if (warm != rocblas_status_success) {     // precision/size combo unsupported
        printf("UNSUPPORTED gemm_ex(%s) %dx%dx%d (status %d)\n", prec.c_str(), m, n, k, (int)warm);
        return 5;
    }
    CK(hipDeviceSynchronize());
    for (int i = 0; i < iters; ++i) RB(gemm());
    CK(hipDeviceSynchronize());
    printf("OK ran %d gemm_ex(%s) %dx%dx%d\n", iters, prec.c_str(), m, n, k);

    rocblas_destroy_handle(h);
    hipFree(A); hipFree(B); hipFree(C);
    return 0;
}
