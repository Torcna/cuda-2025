#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    std::vector<float> result(n * n, 0.f);
    const size_t data_size = n * n * sizeof(float);

    float* d_matrix_a, * d_matrix_b, * d_matrix_c, * d_matrix_ct;
    cudaMalloc(&d_matrix_a, data_size);
    cudaMalloc(&d_matrix_b, data_size);
    cudaMalloc(&d_matrix_c, data_size);
    cudaMalloc(&d_matrix_ct, data_size);

    cudaMemcpy(d_matrix_a, a.data(), data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, b.data(), data_size, cudaMemcpyHostToDevice);

    cublasHandle_t cublas_handle;
    cublasCreate_v2(&cublas_handle);

    const float scale_factor = 1.0f;
    const float zero_factor = 0.0f;

    cublasSgemm_v2(cublas_handle,
                   CUBLAS_OP_T,
                   CUBLAS_OP_T,
                   n, n, n,
                   &scale_factor,
                   d_matrix_a, n,
                   d_matrix_b, n,
                   &zero_factor,
                   d_matrix_c, n);


    cublasSgeam(cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n, n,
                &scale_factor,
                d_matrix_c, n,
                &zero_factor,
                nullptr, n,
                d_matrix_ct, n);

    cublasDestroy_v2(cublas_handle);

    cudaMemcpy(result.data(), d_matrix_ct, data_size, cudaMemcpyDeviceToHost);

    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_c);
    cudaFree(d_matrix_ct);

    return result;
}