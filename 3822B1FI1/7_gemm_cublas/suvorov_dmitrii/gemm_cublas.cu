#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& matrix_a,
                               const std::vector<float>& matrix_b,
                               int dimension) {
  std::vector<float> matrix_result(dimension * dimension);
  size_t buffer_size = dimension * dimension * sizeof(float);

  float *device_matrix_a, *device_matrix_b, *device_result, *device_result_transposed;

  cudaMalloc(&device_matrix_a, buffer_size);
  cudaMalloc(&device_matrix_b, buffer_size);
  cudaMalloc(&device_result, buffer_size);
  cudaMalloc(&device_result_transposed, buffer_size);

  cublasSetMatrix(dimension, dimension, sizeof(float),
                  matrix_a.data(), dimension,
                  device_matrix_a, dimension);
  cublasSetMatrix(dimension, dimension, sizeof(float),
                  matrix_b.data(), dimension,
                  device_matrix_b, dimension);

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
              dimension, dimension, dimension,
              &alpha,
              device_matrix_a, dimension,
              device_matrix_b, dimension,
              &beta,
              device_result, dimension);

  cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
              dimension, dimension,
              &alpha, device_result, dimension,
              &beta, nullptr, dimension,
              device_result_transposed, dimension);

  cublasGetMatrix(dimension, dimension, sizeof(float),
                  device_result_transposed, dimension,
                  matrix_result.data(), dimension);

  cudaFree(device_matrix_a);
  cudaFree(device_matrix_b);
  cudaFree(device_result);
  cudaFree(device_result_transposed);

  cublasDestroy(cublas_handle);

  return matrix_result;
}
