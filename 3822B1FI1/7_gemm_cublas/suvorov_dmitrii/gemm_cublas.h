#ifndef __GEMM_CUBLAS_H
#define __GEMM_CUBLAS_H

#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& matrix_a, const std::vector<float>& matrix_b, int dimension);

#endif  // __GEMM_CUBLAS_H