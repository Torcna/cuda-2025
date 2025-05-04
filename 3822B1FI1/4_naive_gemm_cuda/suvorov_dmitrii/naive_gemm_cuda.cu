#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

#define CUDA_BLOCK_SIZE 32

__global__ void SmallMatrixMulKernel(const float* matrixA, const float* matrixB, float* matrixC, int size) {
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (rowIdx < size && colIdx < size) {
        float dotProduct = 0.0f;
        for (int k = 0; k < size; ++k) {
            dotProduct += matrixA[rowIdx * size + k] * matrixB[k * size + colIdx];
        }
        matrixC[rowIdx * size + colIdx] = dotProduct;
    }
}

__global__ void TiledMatrixMulKernel(const float* matrixA, const float* matrixB, float* matrixC, int size) {
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowIdx >= size || colIdx >= size) return;

    __shared__ float tileA[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
    __shared__ float tileB[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];

    float partialSum = 0.0f;

    for (int tileIdx = 0; tileIdx < (size + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE; ++tileIdx) {
        tileA[threadIdx.y][threadIdx.x] = matrixA[rowIdx * size + tileIdx * CUDA_BLOCK_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = matrixB[(tileIdx * CUDA_BLOCK_SIZE + threadIdx.y) * size + colIdx];

        __syncthreads();

        for (int t = 0; t < CUDA_BLOCK_SIZE; ++t) {
            partialSum += tileA[threadIdx.y][t] * tileB[t][threadIdx.x];
        }

        __syncthreads();
    }

    if (rowIdx < size && colIdx < size) {
        matrixC[rowIdx * size + colIdx] = partialSum;
    }
}

__host__ std::vector<float> NaiveGemmCUDA(const std::vector<float>& inputMatrixA,
                                          const std::vector<float>& inputMatrixB,
                                          int matrixSize) {
    int byteSize = matrixSize * matrixSize * sizeof(float);
    float* deviceMatrixA;
    float* deviceMatrixB;
    float* deviceMatrixC;
    std::vector<float> hostMatrixC(matrixSize * matrixSize);

    cudaMalloc(&deviceMatrixA, byteSize);
    cudaMalloc(&deviceMatrixB, byteSize);
    cudaMalloc(&deviceMatrixC, byteSize);

    cudaMemcpy(deviceMatrixA, inputMatrixA.data(), byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, inputMatrixB.data(), byteSize, cudaMemcpyHostToDevice);

    int numBlocks = (matrixSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    dim3 gridDim(numBlocks, numBlocks);
    dim3 blockDim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);

    if (matrixSize < CUDA_BLOCK_SIZE) {
        SmallMatrixMulKernel<<<gridDim, blockDim>>>(deviceMatrixA, deviceMatrixB, deviceMatrixC, matrixSize);
    } else {
        TiledMatrixMulKernel<<<gridDim, blockDim>>>(deviceMatrixA, deviceMatrixB, deviceMatrixC, matrixSize);
    }

    cudaMemcpy(hostMatrixC.data(), deviceMatrixC, byteSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceMatrixC);

    return hostMatrixC;
}
