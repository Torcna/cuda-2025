#include "block_gemm_cuda.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cstdlib>

const int block_size = 16;

__global__ void gemmKernel(const float* a, const float* b, float* const c, int n) {
    __shared__ float sa[block_size][block_size];
    __shared__ float sb[block_size][block_size];

    int row = blockIdx.y * block_size + threadIdx.y;
    int col = blockIdx.x * block_size + threadIdx.x;
    float val = 0.0f;

    for (int k = 0; k < n; k += block_size) {
        sa[threadIdx.y][threadIdx.x] = (row < n && k + threadIdx.y < n) ? a[row * n + k + threadIdx.x] : 0.0f;
        sb[threadIdx.y][threadIdx.x] = (col < n && k + threadIdx.x < n) ? b[col + n * (k + threadIdx.y)] : 0.0f;

        __syncthreads();

        for (int i = 0; i < block_size; ++i) {
            val += sa[threadIdx.y][i] * sb[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = val;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  int n) {
    std::vector<float> c(n * n, 0.0f);
    size_t bytes = n * n * sizeof(float);
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 Block(block_size, block_size);
    dim3 Grid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

    gemmKernel<<<Grid, Block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
