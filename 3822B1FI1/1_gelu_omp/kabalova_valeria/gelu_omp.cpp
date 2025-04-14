#include "gelu_omp.h"
#include "omp.h"
#include <cmath>

float Gelu(const float& x) {
  return 0.5f * x * (1.0f + std::tanh(coeff3 * (x + 0.044715f * x * x * x)));
}

AlignedVector GeluOMP(const AlignedVector& input) {
  AlignedVector resultVector(input.size());
#pragma omp parallel for 
  for (size_t i = 0; i < input.size(); ++i) {
    resultVector[i] = Gelu(input[i]);
  }

  return resultVector;

}