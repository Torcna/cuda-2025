#ifndef __NAIVE_GEMM_CUDA_H
#define __NAIVE_GEMM_CUDA_H

#include <vector>

std::vector<float> NaiveGemmCUDA(const std::vector<float>& inputMatrixA,
                                 const std::vector<float>& inputMatrixB,
                                 int matrixSize);

#endif // __NAIVE_GEMM_CUDA_H