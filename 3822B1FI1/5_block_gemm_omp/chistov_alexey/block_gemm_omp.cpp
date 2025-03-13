#include <vector>
#include <omp.h>
#include <algorithm>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> res(n * n, 0.0f);
    int block_size = 32;

    #pragma omp parallel for
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int k = 0; k < n; k += block_size) {
                for (int ii = i; ii < std::min(i + block_size, n); ++ii) {
                    for (int kk = k; kk < std::min(k + block_size, n); ++kk) {
                        float tmp = a[ii * n + kk];
                        int min_jj = std::min(j + block_size, n);
                        #pragma omp simd
                        for (int jj = j; jj < min_jj; ++jj) {
                            res[ii * n + jj] += tmp * b[kk * n + jj];
                        }
                    }
                }
            }
        }
    }
    
    return res;
}