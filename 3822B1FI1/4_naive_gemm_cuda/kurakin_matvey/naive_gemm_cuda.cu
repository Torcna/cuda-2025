#include "naive_gemm_cuda.h"

__global__ void NaiveGemm(float *a, float *b, float *res, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < n && j < n){
    float sum = 0.;
    for(int k = 0; k < n; ++k){
      sum += a[i * n + k] * b[k * n + j];
    }
    res[i * n + j] = sum;
  }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
  const int size = n * n;
  std::vector<float> res(size);

  float *p_a, *p_b, *p_res;
  cudaMalloc(&p_a, size * sizeof(float));
  cudaMalloc(&p_b, size * sizeof(float));
  cudaMalloc(&p_res, size * sizeof(float));

  cudaMemcpy(p_a, a.data(), size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(p_b, b.data(), size * sizeof(float), cudaMemcpyHostToDevice);

  const int block_size = 16;
	int num_blocks = (n + block_size - 1) / block_size;

  dim3 dim_num_blocks(num_blocks,num_blocks);
  dim3 dim_block_size(block_size, block_size);
	NaiveGemm <<< dim_num_blocks, dim_block_size >>> (p_a, p_b, p_res, n);

  cudaMemcpy(res.data(), p_res, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(p_a);
  cudaFree(p_b);
  cudaFree(p_res);

  return res;
}
