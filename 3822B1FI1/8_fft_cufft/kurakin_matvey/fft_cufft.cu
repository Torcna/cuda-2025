#include "fft_cufft.h"

__global__ void norm_kernel(float* data, int size, float norm) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    data[i] *= norm;
  }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
  int size = input.size();
  std::vector<float> output(size);

  int n = (size >> 1) / batch;
  int cufft_complex_sizeof = sizeof(cufftComplex) * n * batch;

  cufftHandle plan;
  cufftPlan1d(&plan, n, CUFFT_C2C, batch);

  cufftComplex* data;
  cudaMalloc((void**)&data, cufft_complex_sizeof);

  cudaMemcpy(data, input.data(), cufft_complex_sizeof, cudaMemcpyHostToDevice);

  cufftExecC2C(plan, data, data, CUFFT_FORWARD);
  cufftExecC2C(plan, data, data, CUFFT_INVERSE);

  float norm = 1.0f / (float)n;
  int block_size = 256;
  int num_blocks = (size + block_size - 1) / block_size;
  norm_kernel<<<num_blocks, block_size>>>((float*)data, size, norm);

  cudaMemcpy(output.data(), data, cufft_complex_sizeof, cudaMemcpyDeviceToHost);

  cufftDestroy(plan);
  cudaFree(data);

  return output;
}