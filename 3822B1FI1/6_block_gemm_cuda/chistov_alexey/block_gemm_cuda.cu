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
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int global_row = blockIdx.y * BLOCK_SIZE + thread_y;
    int global_col = blockIdx.x * BLOCK_SIZE + thread_x;

    __shared__ float shared_A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B_tile[BLOCK_SIZE][BLOCK_SIZE];

    float partial_sum = 0.0f;
    for (int tile_index = 0; tile_index < (matrix_dim + BLOCK_SIZE - 1) / BLOCK_SIZE; tile_index++) {
        shared_A_tile[thread_y][thread_x] = (global_row < matrix_dim &&
            (tile_index * BLOCK_SIZE + thread_x) < matrix_dim)
                ? A[global_row * matrix_dim + (tile_index * BLOCK_SIZE + thread_x)]
                : 0.0f;
        shared_B_tile[thread_y][thread_x] = ((tile_index * BLOCK_SIZE + thread_y) < matrix_dim &&
            global_col < matrix_dim)
                ? B[(tile_index * BLOCK_SIZE + thread_y) * matrix_dim + global_col]
                : 0.0f;

        __syncthreads();

        for (int element_index = 0; element_index < BLOCK_SIZE; element_index++) {
            partial_sum += shared_A_tile[thread_y][element_index] *
                           shared_B_tile[element_index][thread_x];
        }
        
        __syncthreads();
    }

    if (global_row < matrix_dim && global_col < matrix_dim) {
        C[global_row * matrix_dim + global_col] = partial_sum;
    }
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& A,
                                 const std::vector<float>& B,
                                 int matrix_dim) {
    size_t size = matrix_dim * matrix_dim * sizeof(float);
    std::vector<float> result(matrix_dim * matrix_dim, 0.0f);

    float *device_A, *device_B, *device_C;
    cudaMalloc((void**)&device_A, size);
    cudaMalloc((void**)&device_B, size);
    cudaMalloc((void**)&device_C, size);

    cudaMemcpy(device_A, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, B.data(), size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((matrix_dim + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (matrix_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);

    BlockGemmKernel<<<grid, block>>>(device_A, device_B, device_C, matrix_dim);
    cudaDeviceSynchronize();
    
    cudaMemcpy(result.data(), device_C, size, cudaMemcpyDeviceToHost);

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    return result;   
}