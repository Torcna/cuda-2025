#include "block_gemm_omp.h"
#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <immintrin.h>
#include <omp.h>

constexpr int BLOCK_SIZE = 8;

template <typename T, size_t Alignment = 32>
struct aligned_allocator {
    using value_type = T;

    T* allocate(size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_t) noexcept {
        std::free(ptr);
    }

    template <typename U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };
};

std::vector<float, aligned_allocator<float>> BlockGemmOMP(
    const std::vector<float, aligned_allocator<float>>& a,
    const std::vector<float, aligned_allocator<float>>& b,
    int n)
{
    std::vector<float, aligned_allocator<float>> c(n * n, 0.0f);
    const float* A = a.data();
    const float* B = b.data();
    float* C = c.data();

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int bi = 0; bi < n; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < n; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < n; bk += BLOCK_SIZE) {

                for (int i = 0; i < BLOCK_SIZE && (bi + i) < n; i++) {
                    int row_offset = (bi + i) * n;
                    for (int j = 0; j < BLOCK_SIZE && (bj + j + 8) <= n; j += 8) {

                        __m256 c_vec = _mm256_load_ps(&C[row_offset + bj + j]);

                        for (int k = 0; k < BLOCK_SIZE && (bk + k) < n; k++) {
                            __m256 a_vec = _mm256_broadcast_ss(&A[row_offset + bk + k]);
                            __m256 b_vec = _mm256_load_ps(&B[(bk + k) * n + (bj + j)]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                        }

                        _mm256_store_ps(&C[row_offset + bj + j], c_vec);
                    }

                    for (int j = (bj + BLOCK_SIZE) & ~7; j < BLOCK_SIZE && (bj + j) < n; j++) {
                        float sum = 0.0f;
                        for (int k = 0; k < BLOCK_SIZE && (bk + k) < n; k++) {
                            sum += A[row_offset + bk + k] * B[(bk + k) * n + (bj + j)];
                        }
                        C[row_offset + bj + j] += sum;
                    }
                }
            }
        }
    }
    return c;
}
