#include "block_gemm_cuda.h"

const int block_size = 16;

__global__ void BlockGemm(float *a, float *b, float *res, int n) {
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  int row = threadIdx.y;
  int col = threadIdx.x;

  int x = n * row + col;
  int a_ind = x;
  float *a_p = &a[n * block_size * block_row];
  int b_ind = x;
  float *b_p = &b[block_size * block_col];

  float *res_p = &res[n * block_size * block_row + block_size * block_col];
  float r = 0.0f;

  __shared__ float a_shared[block_size][block_size];
  __shared__ float b_shared[block_size][block_size];

  for (int k = 0; k < (n + block_size - 1) / block_size; ++k) {
    a_shared[row][col] = a_p[a_ind];
    b_shared[row][col] = b_p[b_ind];
    __syncthreads();

    for (int i = 0; i < block_size; ++i) {
      r += a_shared[row][i] * b_shared[i][col];
    }
    res_p[x] = r;
    __syncthreads();

    a_ind += block_size;
    b_ind += n * block_size;
  }
}

std::vector<float> BlockGemmCUDA(const std::vector<float> &a, const std::vector<float> &b, int n) {
  int size = n * n;
  std::vector<float> res(size);

  float *p_a, *p_b, *p_res;
  cudaMalloc(&p_a, size * sizeof(float));
  cudaMalloc(&p_b, size * sizeof(float));
  cudaMalloc(&p_res, size * sizeof(float));

  cudaMemcpy(p_a, a.data(), size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(p_b, b.data(), size * sizeof(float), cudaMemcpyHostToDevice);

  int num_blocks = (n + block_size - 1) / block_size;

  dim3 dim_num_blocks(num_blocks, num_blocks);
  dim3 dim_block_size(block_size, block_size);
  BlockGemm<<<dim_num_blocks, dim_block_size>>>(p_a, p_b, p_res, n);

  cudaMemcpy(res.data(), p_res, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(p_a);
  cudaFree(p_b);
  cudaFree(p_res);

  return res;
}