#include "gemm_cublas.h"
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    size_t required_size = static_cast<size_t>(n) * n;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    size_t memory = required_size * sizeof(float);
    float* h_A_pinned, * h_B_pinned, * h_C_pinned;
    cudaHostAlloc(&h_A_pinned, memory, cudaHostAllocDefault);
    cudaHostAlloc(&h_B_pinned, memory, cudaHostAllocDefault);
    cudaHostAlloc(&h_C_pinned, memory, cudaHostAllocDefault);

    std::memcpy(h_A_pinned, a.data(), memory);
    std::memcpy(h_B_pinned, b.data(), memory);

    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, memory);
    cudaMalloc(&d_B, memory);
    cudaMalloc(&d_C, memory);
    cudaMemcpyAsync(d_A, h_A_pinned, memory, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B_pinned, memory, cudaMemcpyHostToDevice, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, &alpha, d_B, n, d_A, n, &beta, d_C, n);
    cudaMemcpyAsync(h_C_pinned, d_C, memory, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<float> result(n * n);
    std::memcpy(result.data(), h_C_pinned, memory);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A_pinned);
    cudaFreeHost(h_B_pinned);
    cudaFreeHost(h_C_pinned);
    cudaStreamDestroy(stream);
    cublasDestroy(handle);

    return result;
}