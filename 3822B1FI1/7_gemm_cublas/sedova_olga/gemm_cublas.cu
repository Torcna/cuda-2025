#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n == 0) return {};

    cublasHandle_t handle;
    cublasCreate(&handle);

    size_t size = n * n * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t stat = cublasSgemm(handle,
                                     CUBLAS_OP_T, CUBLAS_OP_T,
                                     n, n, n,
                                     &alpha,
                                     d_b, n,
                                     d_a, n,
                                     &beta,
                                     d_c, n);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cublasDestroy(handle);
        throw std::runtime_error("cuBLAS sgemm failed");
    }

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);

    return c;
}
