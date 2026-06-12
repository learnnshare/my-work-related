// vectoradd.cpp — tiny HIP microkernel used to probe whether this box can run
// GPU kernels and be profiled by rocprofv3. Compile: hipcc vectoradd.cpp -o vectoradd
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float acc = 0.0f;
        // a little arithmetic so the kernel is measurable, not instant
        for (int r = 0; r < 64; ++r) acc = acc * 1.0001f + a[i] * b[i];
        c[i] = acc;
    }
}

#define CK(x) do { hipError_t e = (x); if (e) { printf("HIP error %d: %s at %s:%d\n", \
    e, hipGetErrorString(e), __FILE__, __LINE__); return 2; } } while (0)

int main(int argc, char** argv) {
    // Optional args: n iters  — keep big defaults for real capture; pass a tiny
    // n (e.g. 2048) and iters=1 for gem5 simulation (which is ~1e4-1e5x slower).
    int n = 1 << 22, iters = 50;
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);

    int dev = 0, count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
        printf("NO HIP DEVICE VISIBLE (count=%d)\n", count); return 3;
    }
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, dev));
    printf("device: %s  gcnArch=%s  CUs=%d\n", p.name, p.gcnArchName, p.multiProcessorCount);

    size_t sz = (size_t)n * sizeof(float);
    float *a, *b, *c;
    CK(hipMalloc(&a, sz)); CK(hipMalloc(&b, sz)); CK(hipMalloc(&c, sz));
    CK(hipMemset(a, 1, sz)); CK(hipMemset(b, 1, sz));

    dim3 bs(256), gs((n + 255) / 256);
    for (int it = 0; it < iters; ++it)
        hipLaunchKernelGGL(vadd, gs, bs, 0, 0, a, b, c, n);
    CK(hipDeviceSynchronize());
    printf("OK ran %d vadd launches over n=%d\n", iters, n);
    hipFree(a); hipFree(b); hipFree(c);
    return 0;
}
