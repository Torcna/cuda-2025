#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

#define BLOCK_SIZE 32

__global__ void BlockGemmKernel(const float* A,
                                const float* B,
                                float* C,
                                int matrix_dim) {
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    int global_row = blockIdx.y * BLOCK_SIZE + local_y;
    int global_col = blockIdx.x * BLOCK_SIZE + local_x;

    __shared__ float shared_tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_tile_b[BLOCK_SIZE][BLOCK_SIZE];

    float acc = 0.0f;

    for (int tile_idx = 0; tile_idx < (matrix_dim + BLOCK_SIZE - 1) / BLOCK_SIZE; tile_idx++) {
        int a_col = tile_idx * BLOCK_SIZE + local_x;
        if (global_row < matrix_dim && a_col < matrix_dim) {
            shared_tile_a[local_y][local_x] = A[global_row * matrix_dim + a_col];
        } else {
            shared_tile_a[local_y][local_x] = 0.0f;
        }

        int b_row = tile_idx * BLOCK_SIZE + local_y;
        if (b_row < matrix_dim && global_col < matrix_dim) {
            shared_tile_b[local_y][local_x] = B[b_row * matrix_dim + global_col];
        } else {
            shared_tile_b[local_y][local_x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) {
            acc += shared_tile_a[local_y][i] * shared_tile_b[i][local_x];
        }

        __syncthreads();
    }

    if (global_row < matrix_dim && global_col < matrix_dim) {
        C[global_row * matrix_dim + global_col] = acc;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& A,
                                 const std::vector<float>& B,
                                 int matrix_dim) {
    size_t bytes = matrix_dim * matrix_dim * sizeof(float);
    std::vector<float> result(matrix_dim * matrix_dim, 0.0f);

    float *device_a, *device_b, *device_c;
    cudaMalloc((void**)&device_a, bytes);
    cudaMalloc((void**)&device_b, bytes);
    cudaMalloc((void**)&device_c, bytes);

    cudaMemcpy(device_a, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((matrix_dim + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (matrix_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);

    BlockGemmKernel<<<grid_dim, block_dim>>>(device_a, device_b, device_c, matrix_dim);
    cudaDeviceSynchronize();

    cudaMemcpy(result.data(), device_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return result;
}
