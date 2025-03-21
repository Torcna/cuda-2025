#include "gelu_cuda.h"
#include <cuda_runtime.h>

#define PI 0.797884f
#define BLOCK_SIZE 256

__global__ void GeluKernel(const float* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = in[idx];
        float x_cubed = x * x * x;
        float tanh_input = PI * (x + 0.044715f * x_cubed);
        out[idx] = 0.5f * x * (1.f + tanhf(tanh_input));
    }
}

__host__ std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = input.size(), memory = size * sizeof(float);
    int half_size = size >> 1;
    float* in, *out;
	std::vector<float> result(size);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    cudaMalloc(&in, memory);
    cudaMalloc(&out, memory);

    cudaMemcpyAsync(in, input.data(), size * 2, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(in, input.data(), size * 2, cudaMemcpyHostToDevice, stream2);

    GeluKernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream1>>> (in, out, half_size);
    GeluKernel<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream2>>> (in + half_size, out + half_size, half_size);
    
    cudaMemcpyAsync(result.data(), out, half_size * sizeof(float), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(result.data() + half_size, out + half_size, half_size * sizeof(float), cudaMemcpyDeviceToHost, stream2);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(in);
    cudaFree(out);

    return result;
}
