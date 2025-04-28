#include "block_gemm_omp.h"
#include <omp.h>
#include <vector>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    std::vector<float> c(n * n, 0.0f);
    const int blockSize = 64;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < n; jj += blockSize) {
            for (int kk = 0; kk < n; kk += blockSize) {
                int i_max = std::min(ii + blockSize, n);
                int j_max = std::min(jj + blockSize, n);
                int k_max = std::min(kk + blockSize, n);
                for (int i = ii; i < i_max; ++i) {
                    for (int j = jj; j < j_max; ++j) {
                        float sum = 0.0f;
                        for (int k = kk; k < k_max; ++k) {
                            sum += a[i * n + k] * b[k * n + j];
                        }
#pragma omp atomic
                        c[i * n + j] += sum;
                    }
                }
            }
        }
    }
    return c;
}
