#include "gelu_omp.h"
#include "omp.h"
#include <cmath>
#include <functional>
#include <numbers>

const double arg = std::numbers::sqrt2 * std::numbers::inv_sqrtpi;

const std::function<float(float)> Gelu = [](const float x) {
  return 0.5f * x * (1.0f + std::tanh(*reinterpret_cast<const float*>(&arg) * (x + 0.044715f * x * x * x)));
  };

AlignedVector GeluOMP(const AlignedVector& input) {
  AlignedVector resultVector(input.size());

  for (size_t i = 0; i < input.size(); ++i) {
    resultVector[i] = Gelu(input[i]);
  }

  return resultVector;

}