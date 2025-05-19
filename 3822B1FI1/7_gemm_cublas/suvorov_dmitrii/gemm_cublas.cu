#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& matrix_a, const std::vector<float>& matrix_b, int dimension) {
  int buffer_size = dimension * dimension * sizeof(float);
  std::vector<float> matrix_result(dimension * dimension);

  float *device_matrix_a, *device_matrix_b, *device_result, *device_result_transposed;
  cudaMalloc(&device_matrix_a, buffer_size);
  cudaMalloc(&device_matrix_b, buffer_size);
  cudaMalloc(&device_result, buffer_size);
  cudaMalloc(&device_result_transposed, buffer_size);

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  cublasSetMatrix(dimension, dimension, sizeof(float), matrix_a.data(), dimension, device_matrix_a, dimension);
  cublasSetMatrix(dimension, dimension, sizeof(float), matrix_b.data(), dimension, device_matrix_b, dimension);

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, dimension, dimension, dimension, &alpha, device_matrix_a, dimension, device_matrix_b, dimension, &beta, device_result, dimension);
  cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, dimension, dimension, &alpha, device_result, dimension, &beta, nullptr, dimension, device_result_transposed, dimension);

  cublasGetMatrix(dimension, dimension, sizeof(float), device_result_transposed, dimension, matrix_result.data(), dimension);

  cublasDestroy(cublas_handle);

  cudaFree(device_matrix_a);
  cudaFree(device_matrix_b);
  cudaFree(device_result);
  cudaFree(device_result_transposed);

  return matrix_result;
}
