#include "block_gemm_omp.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> b_trans(n * n);
    int BLOCK_SIZE = 32;
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            b_trans[j * n + i] = b[i * n + j];
        }
    }

    std::vector<float> c(n * n, 0.0f);

#pragma omp parallel for collapse(2) schedule(guided)
    for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
            for (int i = ii; i < std::min(ii + BLOCK_SIZE, n); i++) {
                for (int j = jj; j < std::min(jj + BLOCK_SIZE, n); j++) {
                    float sum = 0.0f;
#pragma omp simd reduction(+:sum)
                    for (int k = 0; k < n; k++) {
                        sum += a[i * n + k] * b_trans[j * n + k];
                    }
                    c[i * n + j] = sum;
                }
            }
        }
    }

    return c;
}
