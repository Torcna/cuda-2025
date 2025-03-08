#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> out(n * n, 0.0f), transposed(n * n);
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            transposed[j * n + i] = b[i * n + j];
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                out[i * n + j] += a[i * n + k] * transposed[j * n + k];
            }
        }
    }

    return out;
}
