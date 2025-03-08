#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> out(n * n, 0.0);
    #pragma omp parallel for
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                out[i * n + j] += a[i * n + k] * b[k * n + j]; 
            }
        }
    }
    return out;
}