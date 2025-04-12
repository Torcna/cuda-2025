#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    
    std::vector<float> c(n*n);
#pragma omp parallel for
    for(size_t i =0; i < n; ++i){
      for (size_t j =0; j < n; ++j){
        float sum = 0.0;
        for(size_t k =0; k < n; ++k){
          sum += a[i * n + k] * b[k * n + j];
        }
        c[i * n + j] = sum;
      }
    }
    
    return c;
}