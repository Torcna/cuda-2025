#include "fft_cufft.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <cstdlib>
#include <vector>

__global__ void kernel(float* input, int size, float norma) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size) {
	  input[i] *= norma;
  }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
	const int size = input.size();
	std::vector<float> result(size);
	int n = (size / batch) >> 1;
	int bytes = sizeof(cufftComplex) * n * batch;
	cufftComplex* data;
	cudaMalloc(&data, bytes);
	cudaMemcpy(data, input.data(), bytes, cudaMemcpyHostToDevice);
	cufftHandle handle;
	cufftPlan1d(&handle, n, CUFFT_C2C, batch);
	cufftExecC2C(handle, data, data, CUFFT_FORWARD);
	cufftExecC2C(handle, data, data, CUFFT_INVERSE);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	size_t threadsPerBlock = deviceProp.maxThreadsPerBlock;
	size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	float norma = 1.0f / static_cast<float>(n);
	kernel<<<blocksPerGrid, threadsPerBlock>>>(reinterpret_cast<float*>(data), size, norma);
	cudaMemcpy(result.data(), data, bytes, cudaMemcpyDeviceToHost);
	cufftDestroy(handle);
	cudaFree(data);
	return result;
}