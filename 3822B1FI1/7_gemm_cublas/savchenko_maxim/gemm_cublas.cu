#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

void transpose(const std::vector<float>& in, std::vector<float>& out, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            out[j * n + i] = in[i * n + j];
}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float* d_A;
    float* d_B;
    float* d_C;

    size_t size = n * n * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    std::vector<float> aT(n * n), bT(n * n);
    transpose(a, aT, n);
    transpose(b, bT, n);

    cudaMemcpy(d_A, aT.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, bT.data(), size, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // C = alpha * A * B + beta * C
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_B, n,
                d_A, n,
                &beta,
                d_C, n);

    std::vector<float> cT(n * n);
    cudaMemcpy(cT.data(), d_C, size, cudaMemcpyDeviceToHost);

    std::vector<float> result(n * n);
    transpose(cT, result, n);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return result;
}
