#include "block_gemm_omp.h"
#include <algorithm>
#include <omp.h>
#include <vector>
std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    const int BLOCK_SIZE = 32;
    std::vector<float> c(n * n, 0.0f);

#pragma omp parallel for collapse(2) schedule(guided)
    for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
                for (int i = ii; i < std::min(ii + BLOCK_SIZE, n); ++i) {
                    for (int k = kk; k < std::min(kk + BLOCK_SIZE, n); ++k) {
                        float a_val = a[i * n + k];
                        int j_end = std::min(jj + BLOCK_SIZE, n);
#pragma omp simd
                        for (int j = jj; j < j_end; ++j) {
                            c[i * n + j] += a_val * b[k * n + j];
                        }
                    }
                }
            }
        }
    }

    return c;
}