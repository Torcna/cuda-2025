#include <cuda_runtime.h>
#include "naive_gemm_cuda.h"

constexpr int BLOCK_SIZE = 16;

__global__ void NaiveGemmKernel(float* _A, float* _B, float* _C, size_t _n) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < _n && y < _n) {
    float sum = 0.f;
    for (size_t k = 0; k < _n; ++k)
      sum += _A[y * _n + k] * _B[k * _n + x];
    _C[y * _n + x] = sum;
  }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
  const std::vector<float>& b,
  int n) {
  size_t size = n * n * sizeof(float);
  float *A, *B, *C;
  std::vector<float> result(n * n, 0.f);

  cudaMalloc(&A, size);
  cudaMalloc(&B, size);
  cudaMalloc(&C, size);

  cudaMemcpy(A, a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(B, b.data(), size, cudaMemcpyHostToDevice);

  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 num_blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  NaiveGemmKernel << <num_blocks, block_dim >> > (A, B, C, n);

  cudaMemcpy(result.data(), C, size, cudaMemcpyDeviceToHost);

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return result;
}