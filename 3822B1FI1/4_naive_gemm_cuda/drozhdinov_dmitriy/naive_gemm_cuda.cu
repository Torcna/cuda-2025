#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda.h>

__global__ void kernel(const float* a, const float* b, float* const c,
                         const size_t size) {
  size_t mIdx = blockIdx.y * blockDim.y + threadIdx.y;
  size_t nIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (mIdx < size && nIdx < size) {
    float elem = 0.0f;
    for (size_t k = 0; k < size; ++k)
      elem += a[mIdx * size + k] * b[size * k + nIdx];
    c[mIdx * size + nIdx] = elem;
  }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int n) {
  std::vector<float> c(n * n);

  size_t bytes = n * n * sizeof(*a.data());

  float* device_a;
  float* device_b;
  float* device_c;
  cudaMalloc(&device_a, bytes);
  cudaMalloc(&device_b, bytes);
  cudaMalloc(&device_c, bytes);

  cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice);

  const size_t sizeAxis = 32u;
  dim3 threadsPerBlock(sizeAxis, sizeAxis);
  dim3 numBlocks((n + sizeAxis - 1) / sizeAxis,
                 (n + sizeAxis - 1) / sizeAxis);

  kernel<<<numBlocks, threadsPerBlock>>>(device_a, device_b, device_c, n);

  cudaMemcpy(c.data(), device_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  return c;
}