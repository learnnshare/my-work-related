// vectoradd.cpp — tiny HIP microkernel used to probe whether this box can run
// GPU kernels and be profiled by rocprofv3. Compile: hipcc vectoradd.cpp -o vectoradd
#include <hip/hip_runtime.h>
#include <cstdio>

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

int main() {
    int dev = 0, count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
        printf("NO HIP DEVICE VISIBLE (count=%d)\n", count); return 3;
    }
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, dev));
    printf("device: %s  gcnArch=%s  CUs=%d\n", p.name, p.gcnArchName, p.multiProcessorCount);

    int n = 1 << 22; size_t sz = (size_t)n * sizeof(float);
    float *a, *b, *c;
    CK(hipMalloc(&a, sz)); CK(hipMalloc(&b, sz)); CK(hipMalloc(&c, sz));
    CK(hipMemset(a, 1, sz)); CK(hipMemset(b, 1, sz));

    dim3 bs(256), gs((n + 255) / 256);
    for (int it = 0; it < 50; ++it)
        hipLaunchKernelGGL(vadd, gs, bs, 0, 0, a, b, c, n);
    CK(hipDeviceSynchronize());
    printf("OK ran 50 vadd launches over n=%d\n", n);
    hipFree(a); hipFree(b); hipFree(c);
    return 0;
}
