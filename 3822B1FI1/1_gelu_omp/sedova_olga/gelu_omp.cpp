#include "gelu_omp.h"
#include <omp.h>
#include <cmath>

AlignedVector GeluOMP(const AlignedVector& input) {
    AlignedVector result(input.size());
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float tanh_arg = std::sqrt(2.0f / 3.14159265f) * (x + 0.044715f * x3);
        result[i] = 0.5f * x * (1.0f + std::tanh(tanh_arg));
    }
    return result;
}
