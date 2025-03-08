#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define BLOCK_SIZE 16

__global__ void BlockGemmKernel(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int n) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    for (int bk = 0; bk < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; bk++) {
        Asub[ty][tx] = (row < n && (bk* BLOCK_SIZE + tx) < n) ? A[row * n + (bk * BLOCK_SIZE + tx)] : 0.0f;
        Bsub[ty][tx] = ((bk * BLOCK_SIZE + ty) < n && col < n) ? B[(bk * BLOCK_SIZE + ty) * n + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += Asub[ty][k] * Bsub[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    size_t size = n * n * sizeof(float);
    std::vector<float> c(n * n, 0.0f);

    float* d_A, * d_B, * d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    BlockGemmKernel << <grid, block >> > (d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}
