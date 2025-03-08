#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#define CUDA_BLOCK_SIZE 16

__global__ void MultKernel(const float* in1, const float* in2, float* out, int n) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    
    // shared memory
    __shared__ float s_in1[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
    __shared__ float s_in2[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];

    float res = 0.0f;

    for (int k = 0; k < (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE; ++k) {

        s_in1[threadIdx.y][threadIdx.x] = in1[row * n + k * CUDA_BLOCK_SIZE + threadIdx.x];
        s_in2[threadIdx.y][threadIdx.x] = in2[(k * CUDA_BLOCK_SIZE + threadIdx.y) * n + col];
        
        __syncthreads();
        for (int t = 0; t < CUDA_BLOCK_SIZE; ++t) {
            res += s_in1[threadIdx.y][t] * s_in2[t][threadIdx.x];
        }
        __syncthreads();  
    }

    if (row < n && col < n) {
        out[row * n + col] = res;
    }
}

__host__ std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    int memory = n * n * sizeof(float);
    float* in1, *in2, *out;
	std::vector<float> result(n * n);
    
    cudaMalloc(&in1, memory);
    cudaMalloc(&in2, memory);
    cudaMalloc(&out, memory);

    cudaMemcpy(in1, a.data(), memory, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(in2, b.data(), memory, cudaMemcpyKind::cudaMemcpyHostToDevice);
    
    int grid = (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    dim3 gridSize(grid, grid),
         blockSize(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
    
    MultKernel<<<gridSize, blockSize>>> (in1, in2, out, n);
    cudaMemcpy(result.data(), out, memory, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(in1);
    cudaFree(in2);
    cudaFree(out);

    return result;
}