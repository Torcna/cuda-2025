#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& input_matrix_a,
                              const std::vector<float>& input_matrix_b,
                              int size) {
    size_t total_bytes = size * size * sizeof(float);
    std::vector<float> output_matrix(size * size);

    float* device_matrix_a;
    float* device_matrix_b;
    float* device_matrix_c;

    cudaMalloc(&device_matrix_a, total_bytes);
    cudaMalloc(&device_matrix_b, total_bytes);
    cudaMalloc(&device_matrix_c, total_bytes);

    cudaMemcpy(device_matrix_a, input_matrix_a.data(), total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix_b, input_matrix_b.data(), total_bytes, cudaMemcpyHostToDevice);
    
    cublasHandle_t cublas_context;
    cublasCreate_v2(&cublas_context);

    const float scalar_alpha = 1.0f;
    const float scalar_beta = 0.0f;
    
    cublasSgemm(cublas_context, CUBLAS_OP_T, CUBLAS_OP_T, size, size, size,
                &scalar_alpha, device_matrix_a, size, device_matrix_b, size, &scalar_beta, device_matrix_c, size);

    cudaMemcpy(output_matrix.data(), device_matrix_c, total_bytes, cudaMemcpyDeviceToHost);

    cudaFree(device_matrix_a);
    cudaFree(device_matrix_b);
    cudaFree(device_matrix_c);

    cublasDestroy_v2(cublas_context);

    return output_matrix;
}
