#include <cuda_runtime.h>
#include "gelu_cuda.h"
#include <cmath>

constexpr float SQRT2DIVPI = 0.7978845f;

__global__ void GeluKernel(const float* _input, float* _result, int _size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < _size) {
    float x = _input[i];
    _result[i] = 0.5f * x * (1.f + tanhf(SQRT2DIVPI * (x + 0.044715f * (x * x * x))));
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  size_t size = input.size();
  std::vector<float> result(size);
  float* input_gpu, * result_gpu;

  int block_size = 256;
  int num_blocks = (size + block_size - 1) / block_size;

  cudaMalloc(&input_gpu, size * sizeof(float));
  cudaMemcpy(input_gpu, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&result_gpu, size * sizeof(float));

  GeluKernel <<<num_blocks, block_size >>> (input_gpu, result_gpu, size);

  cudaMemcpy(result.data(), result_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(input_gpu);
  cudaFree(result_gpu);

  return result;
}
