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
    CUBLAS_CHECK(cublasCreate(&handle));

    size_t size = n * n * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    CUDA_CHECK(cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t stat = cublasSgemm(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
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
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);

    return c;
}