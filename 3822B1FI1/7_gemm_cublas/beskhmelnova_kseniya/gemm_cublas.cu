#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    int m_size = n * n * sizeof(float);
    std::vector<float> res(n * n);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m_size);
    cudaMalloc(&d_B, m_size);
    cudaMalloc(&d_C, m_size);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetMatrix(n, n, sizeof(float), a.data(), n, d_A, n);
    cublasSetMatrix(n, n, sizeof(float), b.data(), n, d_B, n);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n);

    cublasGetMatrix(n, n, sizeof(float), d_C, n, res.data(), n);

    cublasDestroy(handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return res;
}
