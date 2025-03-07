#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 32

__global__ void gemm_kernel_shared(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; k++) {
        int tile_x = k * BLOCK_SIZE + threadIdx.x;
        int tile_y = k * BLOCK_SIZE + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] = (row < n && tile_x < n) ? a[row * n + tile_x] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (col < n && tile_y < n) ? b[tile_y * n + col] : 0.0f;
        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++) {
            sum += tile_a[threadIdx.y][j] * tile_b[j][threadIdx.x];
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
    std::vector<float> c(n * n);
    float *d_a, *d_b, *d_c;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_a, n * n * sizeof(float));
    cudaMalloc(&d_b, n * n * sizeof(float));
    cudaMalloc(&d_c, n * n * sizeof(float));

    cudaMemcpyAsync(d_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice, stream);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gemm_kernel_shared<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_c, n);
    cudaStreamSynchronize(stream);

    cudaMemcpyAsync(c.data(), d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaFreeAsync(d_a, stream);
    cudaFreeAsync(d_b, stream);
    cudaFreeAsync(d_c, stream);

    cudaStreamDestroy(stream);
    return c;
}
