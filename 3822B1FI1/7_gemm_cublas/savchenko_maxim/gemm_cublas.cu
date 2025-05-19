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
	float* d_CT;

    size_t size = n * n * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
	cudaMalloc(&d_CT, size);

    std::vector<float> c(n * n);

    cublasSetMatrix(n, n, sizeof(float), a.data(), n, d_A, n);
    cublasSetMatrix(n, n, sizeof(float), b.data(), n, d_B, n);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // C = alpha * A * B + beta * C
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n);
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, d_C, n, &beta, nullptr, n, d_CT, n);
    cublasGetMatrix(n, n, sizeof(float), d_CT, n, c.data(), n);

    cublasDestroy(handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_CT);

    return c;
}
