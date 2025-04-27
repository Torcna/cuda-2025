#include "gelu_omp.h"
#include <omp.h>
#include <cmath>

AlignedVector GeluOMP(const AlignedVector& input) {
    AlignedVector result(input.size());
    const float sqrt_2_over_pi = 0.7978845608f;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float t = sqrt_2_over_pi * (x + 0.044715f * x3);
        result[i] = 0.5f * x * (1.0f + std::tanh(t));
    }
    return result;
}
