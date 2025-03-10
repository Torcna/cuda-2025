#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n);
	constexpr int block_size = 16;
	int block_count = n / block_size;
	
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < block_count; ++i) {
		for (int j = 0; j < block_count; ++j) {
			for (int k = 0; k < block_count; ++k) {
				for (int block_i = 0; block_i < block_size; ++block_i) {
					for (int block_j = 0; block_j < block_size; ++block_j) {
						float elem = 0.0f;
						#pragma omp simd
						for (int block_k = 0; block_k < block_size; ++block_k) {
							elem += a[(i * block_size + block_i) * n + (k * block_size + block_k)] * b[(k * block_size + block_k) * n + (j * block_size + block_j)];
						}
						#pragma omp atomic
						c[(i * block_size + block_i) * n + (j * block_size + block_j)] += elem;
					}
				}
			}
		}
	}
	return c;
}