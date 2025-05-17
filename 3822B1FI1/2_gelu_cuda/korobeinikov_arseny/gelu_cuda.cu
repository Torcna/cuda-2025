#include "gelu_cuda.h"
#include <cuda_runtime.h>

__device__ __forceinline__ float approx_tanh(float x) {
    const float a = 1.0f;
    const float b = 0.5f;
    float x2 = x * x;
    return x * (a + x2) / (a + b * x2 + x2 * x2);
}

__global__ void gelu_device_kernel(const float* __restrict__ in, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = in[idx];
    float arg = 0.79788456f * (x + 0.044715f * x * x * x);
    float t = (x <= 0.0f) ? tanhf(arg)
             : (arg >= 4.0f) ? 1.0f
             : approx_tanh(arg);
    out[idx] = 0.5f * x * (1.0f + t);
}

std::vector<float> GeluCUDA(const std::vector<float>& host_input) {
    const int total = static_cast<int>(host_input.size());
    const int half = total / 2;

    float *dev_in[2], *dev_out[2];
    cudaStream_t streams[2];

    for (int i = 0; i < 2; ++i) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&dev_in[i], half * sizeof(float));
        cudaMalloc(&dev_out[i], half * sizeof(float));
    }

    cudaMemcpyAsync(dev_in[0], host_input.data(), half * sizeof(float), cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(dev_in[1], host_input.data() + half, half * sizeof(float), cudaMemcpyHostToDevice, streams[1]);

    int block_size, grid_size;
    cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, gelu_device_kernel, 0, 0);
    grid_size = (half + block_size - 1) / block_size;

    for (int i = 0; i < 2; ++i) {
        gelu_device_kernel<<<grid_size, block_size, 0, streams[i]>>>(dev_in[i], dev_out[i], half);
    }

    std::vector<float> result(total);
    cudaMemcpyAsync(result.data(), dev_out[0], half * sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
    cudaMemcpyAsync(result.data() + half, dev_out[1], half * sizeof(float), cudaMemcpyDeviceToHost, streams[1]);

    for (int i = 0; i < 2; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaFree(dev_in[i]);
        cudaFree(dev_out[i]);
        cudaStreamDestroy(streams[i]);
    }

    return result;
}
