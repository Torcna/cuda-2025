#include "block_gemm_cuda.h"

#include <cuda_runtime.h>


#define BLOCK_SIZE 32


__global__ void kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {

  __shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int i = blockIdx.y * BLOCK_SIZE + threadIdx.y;

  float sum = 0;
  int idx;

  for (int phase = 0; phase < gridDim.x; ++phase) {
    idx = phase * BLOCK_SIZE + threadIdx.x;
    if (i < n && idx < n) {
      tile_a[threadIdx.y][threadIdx.x] = a[i * n + idx];
    }

    idx = phase * BLOCK_SIZE + threadIdx.y;

    if (idx < n && j < n) {
      tile_b[threadIdx.y][threadIdx.x] = b[idx * n + j];
    }
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];

    }
    __syncthreads();

  }
  if ((i < n) && (j < n)) {
    c[i * n + j] = sum;
  }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
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
