#ifndef __BLOCK_GEMM_CUDA_H
#define __BLOCK_GEMM_CUDA_H

#include <vector>

std::vector<float> BlockGemmCUDA(const std::vector<float>& A,
                                 const std::vector<float>& B,
                                 int matrix_dim);

#endif // __BLOCK_GEMM_CUDA_H