#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int size = input.size();
    float* d_input;
    float* d_output;

    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    gelu_kernel<<<num_blocks, block_size>>>(d_input, d_output, size);

    std::vector<float> output(size);
    cudaMemcpy(output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
