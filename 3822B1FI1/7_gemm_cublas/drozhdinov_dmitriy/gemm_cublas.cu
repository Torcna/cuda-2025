#include "gemm_cublas.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
  std::vector<float> c(n * n, 0.0f);
  size_t bytes = n * n * sizeof(float);
  float* device_a;
  float* device_b;
  float* device_c;
  cudaMalloc(&device_a, bytes);
  cudaMalloc(&device_b, bytes);
  cudaMalloc(&device_c, bytes);
  cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, device_b, n, device_a, n, &beta, device_c, n);
  cublasDestroy(handle);

  cudaMemcpy(c.data(), device_c, bytes, cudaMemcpyDeviceToHost);
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  return c;
}