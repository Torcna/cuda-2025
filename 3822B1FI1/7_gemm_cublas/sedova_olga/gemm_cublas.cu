#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

#define CUDA_CHECK(call)                                                    
    do {                                                                    
        cudaError_t err = call;                                             
        if (err != cudaSuccess) {                                           
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__    
                      << ", code: " << err << ", reason: "                  
                      << cudaGetErrorString(err) << std::endl;              
            throw std::runtime_error("CUDA call failed");                   
        }                                                                   
    } while (0)

#define CUBLAS_CHECK(call)                                                  
    do {                                                                    
        cublasStatus_t status = call;                                       
        if (status != CUBLAS_STATUS_SUCCESS) {                             
            std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__ 
                      << ", code: " << status << std::endl;                 
            throw std::runtime_error("cuBLAS call failed");                 
        }                                                                   
    } while (0)

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n == 0) return {};

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    size_t size = n * n * sizeof(float);
    std::vector<float> c(n * n);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_ct = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));
    CUDA_CHECK(cudaMalloc(&d_ct, size));

    CUBLAS_CHECK(cublasSetMatrix(n, n, sizeof(float), a.data(), n, d_a, n));
    CUBLAS_CHECK(cublasSetMatrix(n, n, sizeof(float), b.data(), n, d_b, n));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_T,
                             n, n, n,
                             &alpha,
                             d_b, n,
                             d_a, n,
                             &beta,
                             d_c, n));

     CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             n, n, &alpha,
                             d_c, n,
                             &beta,
                             d_a, n,
                             d_ct, n));


    CUBLAS_CHECK(cublasGetMatrix(n, n, sizeof(float), d_ct, n, c.data(), n));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
     CUDA_CHECK(cudaFree(d_ct));
    CUBLAS_CHECK(cublasDestroy(handle));

    return c;
}
