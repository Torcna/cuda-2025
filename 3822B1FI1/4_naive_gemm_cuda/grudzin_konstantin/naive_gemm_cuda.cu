#include "naive_gemm_cuda.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

__global__ void MatMult(float* A, float* B,float* C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if(row<n && col<n)
  {
        float value = 0.0f;
        for (int k = 0; k < n; ++k) {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
  }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> C(n*n);
    float *cuda_A,*cuda_B,*cuda_C;
    int sz=n*n*sizeof(float);
    cudaMalloc(&cuda_A,sz);
    cudaMalloc(&cuda_B,sz);
    cudaMalloc(&cuda_C,sz);
    cudaMemcpy(cuda_A,a.data(),sz,cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_B,b.data(),sz,cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(16, 16); // 16x16 threads per block
    dim3 numBlocks((n + 15) / 16, (n + 15) / 16); // Enough blocks to cover matrix
    MatMult<<<numBlocks, threadsPerBlock>>>(cuda_A, cuda_B, cuda_C, n);
    cudaMemcpy(C.data(),cuda_C,sz,cudaMemcpyDeviceToHost);
    return C;
}
