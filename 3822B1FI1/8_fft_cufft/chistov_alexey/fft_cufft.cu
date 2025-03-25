#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
constexpr int threadsPerBlock = 512;

__global__ void normalize_kernel(float* data, int n, float normalizationFactor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] *= normalizationFactor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int totalSize = input.size();
    int fftSize = totalSize / (2 * batch);
    int byteSize = totalSize * sizeof(cufftComplex);

    cufftComplex* d_input;
    cudaMalloc(&d_input, byteSize);
    cudaMemcpy(d_input, input.data(), byteSize, cudaMemcpyHostToDevice);

    cufftHandle fftPlan;
    cufftPlan1d(&fftPlan, fftSize, CUFFT_C2C, batch);

    cufftExecC2C(fftPlan, d_input, d_input, CUFFT_FORWARD);
    cufftExecC2C(fftPlan, d_input, d_input, CUFFT_INVERSE);

    int blocksPerGrid = (totalSize + threadsPerBlock - 1) / threadsPerBlock;
    float normalizationFactor = 1.0f / static_cast<float>(fftSize);
    normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(reinterpret_cast<float*>(d_input), totalSize, normalizationFactor);

    std::vector<float> result(totalSize);
    cudaMemcpy(result.data(), d_input, byteSize, cudaMemcpyDeviceToHost);

    cufftDestroy(fftPlan);
    cudaFree(d_input);

    return result;
}
