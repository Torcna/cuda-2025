#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
	const std::vector<float>& b,
	int n) {
	int size = n * n * sizeof(float);
	std::vector<float> c(n * n, 0.0f);

	float* d_A, * d_B, * d_C;
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);

	const float alpha = 1.0f, beta = 0.0f;

	cublasSgemm(handle, 
             CUBLAS_OP_T, CUBLAS_OP_T,
             n, n, n, 
             &alpha, 
             d_B, n,
             d_A, n, 
             &beta, 
             d_C, n);

	cudaMemcpy(c.data(), d_C, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			std::swap(c[i * n + j], c[j * n + i]);
		}
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cublasDestroy(handle);

	return c;
}