#include "block_gemm_omp.h"
#include "omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
  const std::vector<float>& b,
  int n) {

  constexpr int block_size = 32;
  int i = 0, ii = 0, j = 0, jj = 0, k = 0, kk = 0;
  std::vector<float>c(n * n);
  
#pragma omp for schedule (static)
  for (ii = 0; ii < n; ii += block_size) {
    for (jj = 0; jj < n; jj += block_size) {
      for (kk = 0; kk < n; kk += block_size) {

        for (i = ii; i < (ii + block_size); ++i) {
          for (k = kk; k < (kk + block_size); ++k) {
            float a_ik = a[i * n + k];

#pragma omp simd
            for (j = jj; j < (jj + block_size); ++j) {
              c[i * n + j] += a_ik * b[k * n + j];
            }

          }
        }
      }
    }
  }
  return c;
}