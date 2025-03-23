#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>

__global__ void normalize_kernel(float* data, int size, float norm_factor) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		data[i] *= norm_factor;
	}
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
	int size = input.size();

	int n = size / (2 * batch);
	int bytes = size * sizeof(float);

	cufftComplex* d_data;
	cudaMalloc(&d_data, bytes);
	cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

	cufftHandle plan;
	cufftPlan1d(&plan, n, CUFFT_C2C, batch);

	cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
	cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	float norm_factor = 1.0f / static_cast<float>(n);
	normalize_kernel << <blocksPerGrid, threadsPerBlock >> > (reinterpret_cast<float*>(d_data), size, norm_factor);

	std::vector<float> result(size);
	cudaMemcpy(result.data(), d_data, bytes, cudaMemcpyDeviceToHost);

	cufftDestroy(plan);
	cudaFree(d_data);

	return result;
}