#include <omp.h>
#include <cmath>
#include "gelu_omp.h"

AlignedVector Gelu(const AlignedVector& input) {
  AlignedVector result(input.size());

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < static_cast<int>(input.size()); ++i) {
    float x = input[i];
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = 0.797885f * std::fma(x3, 0.044715f, x);
    float tanh_val = std::tanh(inner);
    result[i] = 0.5f * x * (1.0f + tanh_val);
  }

  return result;
}
