#include "gelu_cuda.h"

#include <cuda_runtime.h>

__device__ __forceinline__ float my_tanh(float x) {
  constexpr float coeff1 = 0.0001f;
  constexpr float coeff2 = 10395.0f;
  constexpr float coeff3 = 1260.0f;
  constexpr float coeff4 = 21.0f;
  constexpr float coeff5 = 10395.0f;
  constexpr float coeff6 = 4725.0f;
  constexpr float coeff7 = 210.0f;

  if (x < coeff1) return x;

  float x_2 = x * x;

  return x * (coeff2 + x_2 * (coeff3 + x_2 * coeff4)) / (coeff5 + x_2 * (coeff6 + x_2 * (coeff7 + x_2)));
}

__global__ void GeluKernel(const float* __restrict__ input, float* __restrict__ result, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n){
    float x = input[i];
    float arg1 = 0.79788458347320556640625f * (x + 0.044715f * x * x * x);
    float tmp;
    if (x <= 0.0f) tmp = tanh(arg1);
    else if (arg1 >= 4.1f) tmp = 1.0f;
    else tmp = my_tanh(arg1);
    result[i] = 0.5f * x * (1.0f + tmp);
  }
}


std::vector<float> GeluCUDA(const std::vector<float>& input) {
  const size_t n = input.size();
  const size_t n_2 = n/2;
  std::vector<float> resultVector(n);
  
  float *input_gpu1, *input_gpu2;
  cudaMalloc((void**)&input_gpu1, n_2 * sizeof(float));
  cudaMalloc((void**)&input_gpu2, n_2 * sizeof(float));
  float *result_gpu1, *result_gpu2;
  cudaMalloc((void**)&result_gpu1, n_2 * sizeof(float));
  cudaMalloc((void**)&result_gpu2, n_2 * sizeof(float));
  
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  cudaMemcpyAsync(input_gpu1, input.data(), n_2 * sizeof(float), cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(input_gpu2, input.data() + n_2, n_2 * sizeof(float), cudaMemcpyHostToDevice, stream2);
  
  int block_size;
  int num_blocks;
  cudaOccupancyMaxPotentialBlockSize(&num_blocks, &block_size, GeluKernel, 0, 0);
  num_blocks = (n + block_size - 1) / block_size;
  
  GeluKernel <<< num_blocks, block_size, 0, stream1 >>> (input_gpu1, result_gpu1, n_2);
  GeluKernel <<< num_blocks, block_size, 0, stream2 >>> (input_gpu2, result_gpu2, n_2);
  
  cudaMemcpyAsync(resultVector.data(), result_gpu1, n_2  * sizeof(float), cudaMemcpyDeviceToHost, stream1);
  cudaMemcpyAsync(resultVector.data() + n_2, result_gpu2, n_2 * sizeof(float), cudaMemcpyDeviceToHost, stream2);
  
  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);
  
  
  cudaFree(input_gpu1); cudaFree(result_gpu1);
  cudaFree(input_gpu2); cudaFree(result_gpu2);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  
  return resultVector;
}