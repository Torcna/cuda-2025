#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <random>
#include <iostream>
#include <chrono>
#include <cstdlib>


#define BLOCK_SIZE 32


__global__ void kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < n) {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
      sum += a[i * n + k] * b[k * n + j];
    }
    c[i * n + j] = sum;
  }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
  const std::vector<float>& b,
  int n) {

  std::vector<float> c(n * n);

  float* a_gpu, * b_gpu;
  cudaMalloc((void**)&a_gpu, n * n * sizeof(float));
  cudaMalloc((void**)&b_gpu, n * n * sizeof(float));

  float* c_gpu;
  cudaMalloc((void**)&c_gpu, n * n * sizeof(float));

  cudaMemcpy(a_gpu, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_gpu, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

  int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dim_num_blocks(num_blocks, num_blocks);
  dim3 dim_block_size(BLOCK_SIZE, BLOCK_SIZE);

  kernel << < dim_num_blocks, dim_block_size >> > (a_gpu, b_gpu, c_gpu, n);

  cudaMemcpy(c.data(), c_gpu, n * n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(a_gpu); cudaFree(b_gpu); cudaFree(c_gpu);

  return c;
}