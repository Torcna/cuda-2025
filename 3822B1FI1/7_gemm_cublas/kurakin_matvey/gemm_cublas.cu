#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
  int m_size = n * n * sizeof(float);
  std::vector<float> res(n * n);

  float *a_m, *b_m, *res_m, *res_m_t;
  cudaMalloc(&a_m, m_size);
  cudaMalloc(&b_m, m_size);
  cudaMalloc(&res_m, m_size);
  cudaMalloc(&res_m_t, m_size);

  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSetMatrix(n, n, sizeof(float), a.data(), n, a_m, n);
  cublasSetMatrix(n, n, sizeof(float), b.data(), n, b_m, n);

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, &alpha, a_m, n, b_m, n, &beta, res_m, n);
  cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, res_m, n, &beta, nullptr, n, res_m_t, n);

  cublasGetMatrix(n, n, sizeof(float), res_m_t, n, res.data(), n);

  cublasDestroy(handle);

  cudaFree(a_m);
  cudaFree(b_m);
  cudaFree(res_m);
  cudaFree(res_m_t);

  return res;
}