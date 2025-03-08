#include "gelu_cuda.h"
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const float sqrt2pi = 0.797884f;

__global__ void kernel(const float* sample, float* result, size_t elemCount) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < elemCount) {
    const float num = sample[i];
    result[i] = 0.5f * num * (1.0f + tanhf(sqrt2pi * num * (1.0f + 0.044715f * num * num)));
  }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
  const size_t size = input.size();
  std::vector<float> output(size);

  size_t Bytes = size * sizeof(*input.data());

  float* d_input;
  float* d_output;
  cudaMalloc(&d_input, Bytes);
  cudaMalloc(&d_output, Bytes);

  cudaMemcpy(d_input, input.data(), Bytes, cudaMemcpyHostToDevice);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  size_t threadsPerBlock = deviceProp.maxThreadsPerBlock;
  size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);

  cudaMemcpy(output.data(), d_output, Bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
  return output;
}