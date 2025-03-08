#include <omp.h>
#include <cmath>
#include "gelu_omp.h"

float Gelu(float x){
  float in_tanh = 0.797885f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(in_tanh));
}

AlignedVector GeluOMP(const AlignedVector& input) {
  AlignedVector result(input.size());
#pragma omp parallel for
  for (size_t i = 0; i < input.size(); ++i) {
    result[i] = Gelu(input[i]);
  }
	return result;
}
