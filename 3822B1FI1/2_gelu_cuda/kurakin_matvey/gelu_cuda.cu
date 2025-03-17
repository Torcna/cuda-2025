#include "gelu_cuda.h"

__global__ void Gelu(float *input, float *output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < size){
    float x = input[i];
    float in_tanh = 0.797885f * (x + 0.044715f * x * x * x);
    output[i] = 0.5f * x * (1.0f + std::tanh(in_tanh));
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  const int size = input.size();
  std::vector<float> output(size);

  float *p_input, *p_output;
  cudaMalloc(&p_input, size * sizeof(float));
  cudaMalloc(&p_output, size * sizeof(float));

  cudaMemcpy(p_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

  const int block_size = 256;
	int num_blocks = (input.size() + block_size - 1) / block_size;
	Gelu <<< num_blocks, block_size >>> (p_input, p_output, size);

  cudaMemcpy(output.data(), p_output, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(p_input);
  cudaFree(p_output);

  return output;
}