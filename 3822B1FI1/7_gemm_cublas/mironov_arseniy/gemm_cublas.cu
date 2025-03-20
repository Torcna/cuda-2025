#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    int memory = n * n * sizeof(float);
    int square = n * n;
    float* in1, *in2, *out;
	std::vector<float> result(square);
    
    cudaMalloc(&in1, memory);
    cudaMalloc(&in2, memory);
    cudaMalloc(&out, memory);

    cudaMemcpy(in1, a.data(), memory, cudaMemcpyHostToDevice);
    cudaMemcpy(in2, b.data(), memory, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    float alpha = 1.0f, beta= 0.0f;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, &alpha, in1, n, in2, n, &beta, out, n);

    cudaMemcpy(result.data(), out, memory, cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < n; ++i){
        for(int j = i + 1; j < n; ++j){
            std::swap(result[i * n + j], result[j * n + i]);
        }
    }
    cudaFree(in1);
    cudaFree(in2);
    cudaFree(out);
    cublasDestroy(handle);
    return result;
}
