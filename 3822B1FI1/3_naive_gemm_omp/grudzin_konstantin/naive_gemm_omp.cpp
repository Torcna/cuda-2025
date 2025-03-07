#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> dest(n * n, 0.0f);
    
    #pragma omp parallel for
    std::vector<float> bT(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            bT[j * n + i] = b[i * n + j];
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * bT[j * n + k];
            }
            dest[i * n + j] = sum;
        }
    }
    return dest;
}
    // Place your implementation here