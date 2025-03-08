#include "block_gemm_omp.h"
#include <omp.h>
#include <immintrin.h>
#include <vector>

#define BLOCK_SIZE 32

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    float* __restrict A = a.data();
    float* __restrict B = b.data();
    float* __restrict C = c.data();

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int bi = 0; bi < n; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < n; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < n; bk += BLOCK_SIZE) {

                for (int i = 0; i < BLOCK_SIZE && (bi + i) < n; i++) {
                    for (int j = 0; j < BLOCK_SIZE && (bj + j) < n; j += 8) {

                        __m256 c_vec = _mm256_loadu_ps(&C[(bi + i) * n + (bj + j)]);

                        for (int k = 0; k < BLOCK_SIZE && (bk + k) < n; k++) {
                            __m256 a_vec = _mm256_broadcast_ss(&A[(bi + i) * n + (bk + k)]);
                            __m256 b_vec = _mm256_loadu_ps(&B[(bk + k) * n + (bj + j)]);

                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                        }

                        _mm256_storeu_ps(&C[(bi + i) * n + (bj + j)], c_vec);
                    }
                }
            }
        }
    }
    return c;
}

