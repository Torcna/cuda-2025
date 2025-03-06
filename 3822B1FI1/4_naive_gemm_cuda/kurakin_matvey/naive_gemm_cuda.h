#ifndef __NAIVE_GEMM_CUDA_H
#define __NAIVE_GEMM_CUDA_H

#pragma once
#include <cuda_runtime.h>
#include <vector>

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n);

__global__ void NaiveGemm(float *a, float *b, float *res, int n);

#endif // __NAIVE_GEMM_CUDA_H
