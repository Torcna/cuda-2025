#include "gelu_cuda.h"

#include <cuda_runtime.h>

const float arg = 0.7978845608028653558f; //sollya

__global__ void GeluKernel(const float* input, float* result, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n){
    float x = input[i];
		result[i] = 0.5f * x * (1.0f + tanh(arg * (x + 0.044715f * x * x * x)));
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  const size_t n = input.size();
  std::vector<float> resultVector(n);
  
  float* input_gpu;
  cudaMalloc((void**) &input_gpu, n * sizeof(float));
  float* result_gpu;
  cudaMalloc((void**) &result_gpu, n * sizeof(float));
 
  cudaMemcpy(input_gpu, input.data(), n*sizeof(float), cudaMemcpyHostToDevice);
  
  const size_t block_size = 256;
  size_t num_blocks = (n + block_size - 1) / block_size;
  
  GeluKernel <<< num_blocks, block_size >>> (input_gpu, result_gpu, n);
  
  cudaMemcpy(resultVector.data(), result_gpu, n*sizeof(float), cudaMemcpyDeviceToHost);
  
  cudaFree(input_gpu); cudaFree(result_gpu);
  
  return resultVector;
}
