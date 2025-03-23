#include "gelu_cuda.h"
#include <cmath>
#include <cuda_runtime.h>

#define PI 0.797884f
#define BLOCK_SIZE 256

__global__ void GeluKernel(const float* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = in[idx];
        float tanh_in = PI * (x + 0.044715f * x * x * x);
        out[idx] = 0.5f * x * (1.0f + tanh(tanh_in));
    }
}

__host__ std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = input.size(), memory = size * sizeof(float);
    float* in, *out;
	std::vector<float> result(size);

    cudaMalloc(&in, memory);
    cudaMalloc(&out, memory);

    cudaMemcpy(in, input.data(), memory, cudaMemcpyKind::cudaMemcpyHostToDevice);
    GeluKernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>> (in, out, size);
    cudaMemcpy(result.data(), out, memory, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(in);
    cudaFree(out);

    return result;
}
