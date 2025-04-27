#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

// CUDA kernel for GELU activation
__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
        float c = 0.044715f;
        float sqrt_2_over_pi = 0.7978845608f; // ≈ sqrt(2/pi)
        float x3 = x * x * x;
        float t = sqrt_2_over_pi * (x + c * x3);
        output[idx] = 0.5f * x * (1.0f + tanhf(t));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = static_cast<int>(input.size());
    if (n == 0) return {};

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    gelu_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);

    std::vector<float> output(n);
    cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
