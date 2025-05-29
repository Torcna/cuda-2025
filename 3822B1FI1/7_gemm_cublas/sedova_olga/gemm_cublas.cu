#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n == 0) return {};

    cublasHandle_t handle;
    cublasCreate(&handle);

    size_t size = n * n * sizeof(float);
    std::vector<float> c(n * n);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_ct = nullptr;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMalloc(&d_ct, size);

    cublasSetMatrix(n, n, sizeof(float), a.data(), n, d_a, n);
    cublasSetMatrix(n, n, sizeof(float), b.data(), n, d_b, n);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_T, 
                CUBLAS_OP_T,
                n, n, n,
                &alpha,
                d_a, n,
                d_b, n,
                &beta,
                d_c, n);

    cublasSgemm(handle, CUBLAS_OP_T, 
                CUBLAS_OP_N,
                n, n, &alpha,
                d_c, n,
                &beta,
                nullptr, n,
                d_ct, n);

    cublasGetMatrix(n, n, sizeof(float), d_ct, n, c.data(), n));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_ct);
    cublasDestroy(handle);

    return c;
}
