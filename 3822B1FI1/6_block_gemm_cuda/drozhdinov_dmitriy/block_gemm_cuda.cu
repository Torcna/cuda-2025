#include "block_gemm_cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>

const int block_size = 16;

__global__ void kernel(const float* a, const float* b, float* const c, const int size) {
	__shared__ float sa[16][16];
	__shared__ float sb[16][16];
	int row = blockIdx.y * block_size + threadIdx.y;
	int col = blockIdx.x * block_size + threadIdx.x;
	float elem = 0.0f;
	for (int k = 0; k < n; k+= block_size) {
		if (row < n && k + threadIdx.y < n) {
			sa[threadIdx.y][threadIdx.x] = a[row * n + k + threadIdx.x];
		}
		else {
			sa[threadIdx.y][threadIdx.x] = 0.0f;
		}
		if (col < n && k + threadIdx.x < n) {
			sb[threadIdx.y][threadIdx.x] = b[col + n * (k + threadIdx.y)];
		}
		else {
			sb[threadIdx.y][threadIdx.x] = 0.0f;
		}
		__syncthreads();
		for (int i = 0; i < block_size; ++i) {
			elem += sa[threadIdx.y][i] * sb[i][threadIdx.x];
		}
		__syncthreads();
	}
	if (row < n && col < n) {
        c[row * n + col] = elem;
    }
}


std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(n * n, 0.0f);
	size_t bytes = n * n * sizeof(float);
	float* device_a;
	float* device_b;
	float* device_c;
	cudaMalloc(&device_a, bytes);
	cudaMalloc(&device_b, bytes);
	cudaMalloc(&device_c, bytes);
	cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice);
	dim3 dimBlock(block_size, block_size);
	dim3 dimGrid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);
	kernel<<<dimGrid, dimBlock>>>(device_a, device_b, device_c, n, block_size);
	cudaMemcpy(c.data(), device_c, bytes, cudaMemcpyDeviceToHost);
	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);
	return c;
}