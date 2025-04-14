#include "gelu_omp.h"
#include "omp.h"
#include <cmath>
#include <cstdint>

float Gelu(const float& x) {
  //0.5f * x * (1.0f + std::tanh(coeff3 * (x + 0.044715f * x * x * x)))
  const float coeff1 = 0.5f;
  const float coeff2 = 0.044715f;
  const float coeff3 = 0.7978845608028653558f;
  float x_3 = x * x * x;
  float tmp1 = std::fmaf(coeff2, x_3, x);
  tmp1 = coeff3 * tmp1;
  tmp1 = std::tanhf(tmp1) + 1.0f;
  uint32_t tmp2 = *reinterpret_cast<const uint32_t*>(&x);
  tmp2 = tmp2 >> 1; //0.5f * x
  float tmp3 = *reinterpret_cast<float*>(&tmp2);
  return tmp3 * tmp1;
}

AlignedVector GeluOMP(const AlignedVector& input) {
  AlignedVector resultVector(input.size());
#pragma omp parallel for 
  for (size_t i = 0; i < input.size(); ++i) {
    resultVector[i] = Gelu(input[i]);
  }

  return resultVector;

}