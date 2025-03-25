#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>

__global__ void normalize_kernel(float* data, int n, float normalizationFactor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] *= normalizationFactor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int totalSize  = input.size();
    int n = (totalSize  / batch) >> 1;

    int byteSize  = sizeof(cufftComplex) * n * batch;
    cufftHandle fftPlan;
    cufftPlan1d(&fftPlan, n, CUFFT_C2C, batch);
    cufftComplex* d_input;

    cudaMalloc(&d_input, byteSize );
    cudaMemcpy(d_input, input.data(), byteSize , cudaMemcpyHostToDevice);
    cufftExecC2C(fftPlan, d_input, d_input, CUFFT_FORWARD);
    cufftExecC2C(fftPlan, d_input, d_input, CUFFT_INVERSE);
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, 0);
    size_t threadsPerBlock = deviceProp.maxThreadsPerBlock;
    size_t blocksPerGrid = (totalSize  + threadsPerBlock - 1) / threadsPerBlock;
    float norm = 1.0f / static_cast<float>(n);
    normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        reinterpret_cast<float*>(d_input), totalSize , norm);
    
    std::vector<float> result(totalSize);
    cudaMemcpy(result.data(), d_input, byteSize , cudaMemcpyDeviceToHost);

    cufftDestroy(fftPlan);
    cudaFree(d_input);

    return result;
}