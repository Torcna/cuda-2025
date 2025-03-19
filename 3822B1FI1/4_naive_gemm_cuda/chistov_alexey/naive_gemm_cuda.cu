#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void MatrixMultiply(float* A, float* B, float* C, int n) {
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (rowIndex < n && colIndex < n) {
        float resultValue = 0.0f;
        for (int k = 0; k < n; ++k) {
            resultValue += A[rowIndex * n + k] * B[k * n + colIndex];
        }
        C[rowIndex * n + colIndex] = resultValue;
    } 
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  int n) {
    std::vector<float> result(n * n);
    float *A, *B, *C;
    int size = n * n * sizeof(float);

    cudaMalloc(&A, size);
    cudaMalloc(&B, size);
    cudaMalloc(&C, size);

    cudaMemcpy(A, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(B, b.data(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((n + 31) / 32, (n + 31) / 32);

    MatrixMultiply<<<numBlocks, threadsPerBlock>>>(A, B, C, n);\
    cudaMemcpy(result.data(), C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    
    return result;
}