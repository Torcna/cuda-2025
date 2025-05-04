#include "fft_cufft.h"
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>

#define BLOCK_SIZE 256

__global__ void normalizeKernel(float* data, int total_size, float normFactor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        data[idx] *= normFactor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int totalFloats = input.size();
    int n = totalFloats / (2 * batch);
    int totalComplex = totalFloats / 2;

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    size_t memory = totalFloats * sizeof(float);
    cufftComplex* d_data = nullptr;
    cudaMalloc((void**)&d_data, sizeof(cufftComplex) * totalComplex);
    cudaMemcpy(d_data, input.data(), memory, cudaMemcpyHostToDevice);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (totalComplex + threadsPerBlock - 1) / threadsPerBlock;
    float normFactor = 1.0f / n;
    normalizeKernel << <blocksPerGrid, threadsPerBlock >> > (reinterpret_cast<float*>(d_data), totalFloats, normFactor);

    std::vector<float> result(totalFloats);
    cudaMemcpy(result.data(), d_data, memory, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cufftDestroy(plan);

    return result;
}