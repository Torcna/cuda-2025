#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>

__global__ void normalize_kernel(float* buffer, int length, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        buffer[idx] *= factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& srcData, int numBatches) {
    const int dataSize = srcData.size();
    const int fftLen = (dataSize / numBatches) >> 1;
    const int bufferBytes = sizeof(cufftComplex) * fftLen * numBatches;

    cufftComplex* gpuBuffer;
    cufftHandle plan;
    cufftPlan1d(&plan, fftLen, CUFFT_C2C, numBatches);
    cudaMalloc(&gpuBuffer, bufferBytes);
    cudaMemcpy(gpuBuffer, srcData.data(), bufferBytes, cudaMemcpyHostToDevice);

    cufftExecC2C(plan, gpuBuffer, gpuBuffer, CUFFT_FORWARD);
    cufftExecC2C(plan, gpuBuffer, gpuBuffer, CUFFT_INVERSE);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int threads = prop.maxThreadsPerBlock;
    const int blocks = (dataSize + threads - 1) / threads;
    const float scale = 1.0f / static_cast<float>(fftLen);

    normalize_kernel<<<blocks, threads>>>(
        reinterpret_cast<float*>(gpuBuffer), dataSize, scale);

    std::vector<float> output(dataSize);
    cudaMemcpy(output.data(), gpuBuffer, bufferBytes, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(gpuBuffer);

    return output;
}
