#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>

__global__ void gelu_kernel(const float* input, float* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float x = input[idx];
    float x3 = x * x * x;
    float arg = x + 0.044715f * x3;
    float tanh_val = 0.797885f * arg;
    float gelu_result = 0.5f * x * (1.0f + tanhf(tanh_val));
    output[idx] = gelu_result;
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  const int size = input.size();
  std::vector<float> output(size);

  float* d_input;
  float* d_output;

  cudaError_t err = cudaMalloc((void**)&d_input, size * sizeof(float));
  if (err != cudaSuccess) {
    return output;
  }

  err = cudaMalloc((void**)&d_output, size * sizeof(float));
  if (err != cudaSuccess) {
    cudaFree(d_input);
    return output;
  }

  err = cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_input);
    cudaFree(d_output);
    return output;
  }

  const int block_size = 256;
  const int num_blocks = (size + block_size - 1) / block_size;

  gelu_kernel<<<num_blocks, block_size>>>(d_input, d_output, size);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(d_input);
    cudaFree(d_output);
    return output;
  }

  err = cudaMemcpy(output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    cudaFree(d_input);
    cudaFree(d_output);
    return output;
  }

  cudaFree(d_input);
  cudaFree(d_output);

  return output;
}
