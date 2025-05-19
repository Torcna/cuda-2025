#include <omp.h>
#include <cmath>
#include "gelu_omp.h"

float Gelu(float val) {
  float tanh_arg = 0.797885f * (val + 0.044715f * val * val * val);
  return 0.5f * val * (1.0f + std::tanh(tanh_arg));
}

AlignedVector GeluOMP(const AlignedVector& input) {
  AlignedVector output(input.size());
#pragma omp parallel for
  for (size_t idx = 0; idx < input.size(); ++idx) {
    output[idx] = Gelu(input[idx]);
  }
  return output;
}
