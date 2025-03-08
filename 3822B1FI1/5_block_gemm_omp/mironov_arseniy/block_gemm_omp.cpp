#include "block_gemm_omp.h"
#include <immintrin.h>
#include <cmath>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> x(n * n, 0.0f);
    // cache 32 KB
    int block_n = 64;
    if(n < 64) block_n = n;
    #pragma omp parallel for
    for (int i = 0; i < n; i += block_n) {
        for (int j = 0; j < n; j += block_n) {
            for (int k = 0; k < n; k += block_n) {
                for (int k1 = k; k1 < k + block_n; k1++) {
                     for (int i1 = i; i1 < i + block_n; i1++){
                        for (int j1 = j; j1 < j + block_n; j1++)  {
                            x[i1 * n + j1] += a[i1 * n + k1] * b[k1 * n + j1];
                        }
                    }
                }
            }
        }
    }
    return x;
}
