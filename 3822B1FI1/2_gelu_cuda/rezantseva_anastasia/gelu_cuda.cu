#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

// 0.7978845608f = std::sqrt(2.0f / M_PI)

__global__ void geluKernel(float* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        float x2 = x * x;
        float x3 = x2 * x;
        float term = x + 0.044715f * x3;
        float arg = 0.7978845608f * term;
        data[idx] = 0.5f * x * (1.0f + tanhf(arg));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) {
        return std::vector<float>();
    }
	
    size_t n = input.size();
    std::vector<float> output(n);
    float* d_data = nullptr;

    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    geluKernel<<<gridSize, blockSize>>>(d_data, n);
    cudaMemcpy(output.data(), d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
	
    return output;
}