#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void BlockGemmKernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= n || col >= n) return;

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    float value = 0.0f;

    for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++k) {
        int a_idx = row * n + (k * BLOCK_SIZE + threadIdx.x);
        if (row < n && (k * BLOCK_SIZE + threadIdx.x) < n) {
			As[threadIdx.y * BLOCK_SIZE + threadIdx.x] = A[a_idx];
		} else {
			As[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0f;
		}

        int b_idx = (k * BLOCK_SIZE + threadIdx.y) * n + col;
		if (col < n && (k * BLOCK_SIZE + threadIdx.y) < n) {
			Bs[threadIdx.y * BLOCK_SIZE + threadIdx.x] = B[b_idx];
		} else {
			Bs[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0f;
		}

        __syncthreads();

        for (int t = 0; t < BLOCK_SIZE; ++t) {
            value += As[threadIdx.y * BLOCK_SIZE + t] * Bs[t * BLOCK_SIZE + threadIdx.x];
        }

        __syncthreads();
    }
	
    C[row * n + col] = value;
}

__host__ std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                          const std::vector<float>& b,
                                          int n) {

    size_t size = n * n * sizeof(float);
    
    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    BlockGemmKernel <<<blocks, threads>>> (d_a, d_b, d_c, n);

    std::vector<float> result(n * n);
    cudaMemcpy(result.data(), d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return result;
}
