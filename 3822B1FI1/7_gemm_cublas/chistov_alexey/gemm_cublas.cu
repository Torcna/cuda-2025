#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b, int size) {
    std::vector<float> result(size * size);

    size_t size = size * size * sizeof(float);

    float* d_a;
    float* d_b;
    float* d_c;

    if (cudaMalloc(&d_a, size) != cudaSuccess) {
        std::cerr << "Error allocating device memory for d_a" << std::endl;
        return result;
    }
    if (cudaMalloc(&d_b, size) != cudaSuccess) {
        std::cerr << "Error allocating device memory for d_b" << std::endl;
        cudaFree(d_a);
        return result;
    }
    if (cudaMalloc(&d_c, size) != cudaSuccess) {
        std::cerr << "Error allocating device memory for d_c" << std::endl;
        cudaFree(d_a);
        cudaFree(d_b);
        return result;
    }

    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    cublasStatus_t status = cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 size, size, size,
                 &alpha,
                 d_b, CUDA_R_32F, size,
                 d_a, CUDA_R_32F, size,
                 &beta,
                 d_c, CUDA_R_32F, size,
                 CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Error during cuBLAS GEMM operation" << std::endl;
    }

    cudaMemcpy(result.data(), d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);

    return result;
}