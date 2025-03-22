#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    std::vector<float> C(n*n);
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    float *d_A, *d_B, *d_C, *d_C_T;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_B, n * n * sizeof(float));
    cudaMalloc(&d_C, n * n * sizeof(float));
    cudaMalloc(&d_C_T, n * n * sizeof(float));

    cudaMemcpy(d_A, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n);
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, d_C, n, &beta, nullptr, n, d_C_T, n);

    cudaMemcpy(C.data(), d_C_T, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_T);
    cublasDestroy(handle);
    return C;
}