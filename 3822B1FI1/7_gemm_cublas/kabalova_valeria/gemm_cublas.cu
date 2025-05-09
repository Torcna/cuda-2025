#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
  const std::vector<float>& b,
  int n) {

  const float alpha = 1.0f;
  const float beta = 0.0f;
  std::vector<float> c(n * n);

  float* a_gpu, * b_gpu;
  cudaMalloc((void**)&a_gpu, n * n * sizeof(float));
  cudaMalloc((void**)&b_gpu, n * n * sizeof(float));

  float* c_gpu;
  cudaMalloc((void**)&c_gpu, n * n * sizeof(float));

  cublasSetMatrix(n, n, sizeof(float), a.data(), n, a_gpu, n);
  cublasSetMatrix(n, n, sizeof(float), b.data(), n, b_gpu, n);

  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, b_gpu, n, a_gpu, n, &beta, c_gpu, n);
 
  cublasGetMatrix(n, n, sizeof(float), c_gpu, n, c.data(), n);
  cublasDestroy(handle);

  cudaFree(a_gpu); cudaFree(b_gpu); cudaFree(c_gpu);

  return c;

}