#include "fft_cufft.h"
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>

#define BLOCK_SIZE 256

__global__ void normalizeKernel(cufftComplex* data, int totalComplex, float normFactor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalComplex) {
        data[idx].x *= normFactor;
        data[idx].y *= normFactor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int totalFloats = input.size();
    int n = totalFloats / (2 * batch);
    int totalComplex = totalFloats / 2;

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cufftSetStream(plan, stream);

    size_t memory = totalFloats * sizeof(float);
    float* h_input_pinned, * h_output_pinned;
    cudaHostAlloc(&h_input_pinned, memory, cudaHostAllocDefault);
    cudaHostAlloc(&h_output_pinned, memory, cudaHostAllocDefault);
    std::memcpy(h_input_pinned, input.data(), memory);

    cufftComplex* d_data = nullptr;
    cudaMalloc((void**)&d_data, sizeof(cufftComplex) * totalComplex);
    cudaMemcpyAsync(d_data, h_input_pinned, memory, cudaMemcpyHostToDevice, stream);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (totalComplex + threadsPerBlock - 1) / threadsPerBlock;
    float normFactor = 1.0f / n;
    normalizeKernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (d_data, totalComplex, normFactor);

    cudaMemcpyAsync(h_output_pinned, d_data, memory, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<float> result(totalFloats);
    std::memcpy(result.data(), h_output_pinned, memory);

    cudaFree(d_data);
    cudaFreeHost(h_input_pinned);
    cudaFreeHost(h_output_pinned);
    cudaStreamDestroy(stream);
    cufftDestroy(plan);

    return result;
}