#include "naive_gemm_omp.h"
#include <omp.h>


std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
  const std::vector<float>& b,
  int n) {

  std::vector<float> c(n * n);
#pragma omp parallel for 
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      int j = 0;
      float a_ik = a[i * n + k];
      while (j + 7 < n) {
        c[i * n + j] += a_ik * b[k * n + j];
        c[i * n + j + 1] += a_ik * b[k * n + j + 1];
        c[i * n + j + 2] += a_ik * b[k * n + j + 2];
        c[i * n + j + 3] += a_ik * b[k * n + j + 3];
        c[i * n + j + 4] += a_ik * b[k * n + j + 4];
        c[i * n + j + 5] += a_ik * b[k * n + j + 5];
        c[i * n + j + 6] += a_ik * b[k * n + j + 6];
        c[i * n + j + 7] += a_ik * b[k * n + j + 7];
        j += 8;

      }
    }
  }

  return c;
}