#ifndef __GEMM_CUBLAS_H
#define __GEMM_CUBLAS_H

#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& input_matrix_a,
							  const std::vector<float>& input_matrix_b,
							  int size);

#endif // __GEMM_CUBLAS_H