#include "gelu_cuda.h"
#include <cmath>
#include <cuda_runtime.h>

#define PI 0.797884f

using namespace std;

__global__ void GELU(float* X, float* res, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = X[idx];
        res[idx] = 0.5f * x * (1.f + tanhf(PI * (x + 0.044715f * (x * x * x))));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = input.size();
    vector<float> res(n);
    float *cuda_input, *cuda_res;

    cudaMalloc(&cuda_input, n * sizeof(float));
    cudaMalloc(&cuda_res, n * sizeof(float));

    cudaMemcpy(cuda_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256;
    int blocks = (n + block - 1) / block;

    GELU<<<blocks, block>>>(cuda_input, cuda_res, n);
    cudaDeviceSynchronize(); // Ensure the kernel finishes executing

    cudaMemcpy(res.data(), cuda_res, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(cuda_input);
    cudaFree(cuda_res);
    return res;
}
