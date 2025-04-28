#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void NaiveGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float s_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < n; k += BLOCK_SIZE) {
        s_a[threadIdx.y][threadIdx.x] = (row < n && k + threadIdx.x < n)
            ? a[row * n + k + threadIdx.x]
            : 0.0f;

        s_b[threadIdx.y][threadIdx.x] = (col < n && k + threadIdx.y < n)
            ? b[(k + threadIdx.y) * n + col]
            : 0.0f;
        __syncthreads();

        for (int t = 0; t < BLOCK_SIZE; ++t) {
            sum += s_a[threadIdx.y][t] * s_b[t][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t bytes = n * n * sizeof(float);
    float* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    NaiveGemmKernel << <gridDim, blockDim >> > (d_a, d_b, d_c, n);

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}