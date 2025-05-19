#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>

#define BLOCK_SIZE 256

__global__ void normalize_complex(cufftComplex* data, int totalComplex, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalComplex) {
        data[idx].x /= n;
        data[idx].y /= n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    const int totalFloats = input.size();
    const int n = totalFloats / (2 * batch);
    const int totalComplex = totalFloats / 2;

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftComplex* gpu_data;
    cudaMalloc(&gpu_data, totalComplex * sizeof(cufftComplex));
    cudaMemcpy(gpu_data, input.data(), totalFloats * sizeof(float), cudaMemcpyHostToDevice);

    cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_FORWARD);
    cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_INVERSE);

    int blocksPerGrid = (totalComplex + BLOCK_SIZE - 1) / BLOCK_SIZE;
    normalize_complex << <blocksPerGrid, BLOCK_SIZE >> > (gpu_data, totalComplex, n);

    std::vector<float> result(totalFloats);
    cudaMemcpy(result.data(), gpu_data, totalFloats * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpu_data);
    cufftDestroy(plan);

    return result;
}