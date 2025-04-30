#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>

#define BLOCK_SIZE 16

__global__ void block_gemm_kernel(const float* a, const float* b, float* c, int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int block = 0; block < n / BLOCK_SIZE; ++block) {
        if (row < n && block * BLOCK_SIZE + threadIdx.x < n)
            As[threadIdx.y][threadIdx.x] = a[row * n + block * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < n && block * BLOCK_SIZE + threadIdx.y < n)
            Bs[threadIdx.y][threadIdx.x] = b[(block * BLOCK_SIZE + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n == 0) return {};

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t size = n * n * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    block_gemm_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
