#include "gelu_omp.h"
#include <omp.h>
#include <cmath>

AlignedVector GeluOMP(const AlignedVector& input) {
    AlignedVector result(input.size());
    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        result[i] = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / 3.14159265f) *
            (x + 0.044715f * std::pow(x, 3))));
    }
    return result;
}
