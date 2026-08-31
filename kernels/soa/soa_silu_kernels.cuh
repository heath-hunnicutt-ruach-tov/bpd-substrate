// soa_silu_kernels.cuh — SiLU and SiLU+mul inplace kernels for stock path fusion
#pragma once
#include <math.h>

static __global__ void silu_inplace_f32(float *data, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = data[i];
        data[i] = x / (1.0f + expf(-x));
    }
}

static __global__ void silu_mul_inplace_f32(float *data, const float *up, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = data[i];
        data[i] = (x / (1.0f + expf(-x))) * (up ? up[i] : 1.0f);
    }
}

// In-place residual add: dst[i] += residual[i]
static __global__ void add_inplace_f32(float *data, const float *residual, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && residual) {
        data[i] += residual[i];
    }
}
