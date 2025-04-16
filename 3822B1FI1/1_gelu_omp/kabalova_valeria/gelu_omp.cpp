#include "gelu_omp.h"
#include "omp.h"
#include <cmath>

inline float my_tanh(float x) {
  constexpr float coeff1 = 0.0001f;
  constexpr float coeff2 = 10395.0f;
  constexpr float coeff3 = 1260.0f;
  constexpr float coeff4 = 21.0f;
  constexpr float coeff5 = 10395.0f;
  constexpr float coeff6 = 4725.0f;
  constexpr float coeff7 = 210.0f;

  if (x < coeff1) return x;

  float x_2 = x * x;

  return x * (coeff2 + x_2 * (coeff3 + x_2 * coeff4)) / (coeff5 + x_2 * (coeff6 + x_2 * (coeff7 + x_2)));
}

float Gelu(const float& x) {
  constexpr float coeff1 = 0.79788458347320556640625f;
  constexpr float coeff2 = 0.044715f;
  constexpr float coeff3 = 4.1f;
  constexpr float coeff4 = 1.0f;
  constexpr float coeff5 = 0.5f;

  float arg = coeff1 * (x + coeff2 * x * x * x);
  float tmp;
  if (x <= 0.0f) tmp = std::tanhf(arg);
  else if (arg >= coeff3) tmp = coeff4;
  else tmp = my_tanh(arg);
  return coeff5 * x * (coeff4 + tmp);

}

AlignedVector GeluOMP(const AlignedVector& input) {
  AlignedVector resultVector(input.size());
#pragma omp parallel for 
  for (size_t i = 0; i < input.size(); ++i) {
    resultVector[i] = Gelu(input[i]);
  }

  return resultVector;

}