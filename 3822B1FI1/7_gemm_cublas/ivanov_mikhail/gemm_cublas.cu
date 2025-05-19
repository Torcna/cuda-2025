#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
  const std::vector<float>& b,
  int n) {
  std::vector<float> result(n * n, 0.f);
  int size = n * n * sizeof(float);
  float* A, *B, *C,* Ct;  // Ct - transposed matrix C

  cudaMalloc(&A, size);
  cudaMalloc(&B, size);
  cudaMalloc(&C, size);
  cudaMalloc(&Ct, size);

  cudaMemcpy(A, a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(B, b.data(), size, cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate_v2(&handle);

  const float alpha = 1.f;
  const float beta = 0.f;

  cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, &alpha, A, n, B, n, &beta, C, n);
  cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &alpha, C, n, &beta, nullptr, n, Ct, n);

  cublasDestroy_v2(handle);

  cudaMemcpy(result.data(), Ct, size, cudaMemcpyDeviceToHost);

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(Ct);

  return result;
}