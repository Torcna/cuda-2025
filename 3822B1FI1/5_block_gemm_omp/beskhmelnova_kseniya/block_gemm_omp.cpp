#include "block_gemm_omp.h"
#include <immintrin.h>
#include <omp.h>
std::vector<float> BlockGemmOMP(const std::vector<float>& a,
	const std::vector<float>& b,
	int n) {

	int block_size = 64;
	if (n < block_size) {
		block_size = n;
	}

	std::vector<float> result(n * n, 0.0f);

#pragma omp parallel for
	for (int block_row = 0; block_row < n; block_row += block_size) {
		for (int block_col = 0; block_col < n; block_col += block_size) {
			for (int block_inner = 0; block_inner < n; block_inner += block_size) {
				for (int i = block_row; i < block_row + block_size && i < n; i++) {
					for (int j = block_col; j < block_col + block_size && j < n; j++) {
						for (int k = block_inner; k < block_inner + block_size && k < n; k++) {
							result[i * n + j] += a[i * n + k] * b[k * n + j];
						}
					}
				}
			}
		}
	}
	return result;
}
