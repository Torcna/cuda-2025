#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    int size = n * n * sizeof(float);
    std::vector<float> c(n * n);

    float *d_A, *d_B, *d_C, *d_C_transposed;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMalloc(&d_C_transposed, size);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetMatrix(n, n, sizeof(float), a.data(), n, d_A, n);
    cublasSetMatrix(n, n, sizeof(float), b.data(), n, d_B, n);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n);
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, d_C, n, &beta, nullptr, n, d_C_transposed, n);
    cublasGetMatrix(n, n, sizeof(float), d_C_transposed, n, c.data(), n);

    cublasDestroy(handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_transposed);

    return c;
}
