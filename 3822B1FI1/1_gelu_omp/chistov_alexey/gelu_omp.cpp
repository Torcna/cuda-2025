#include <omp.h>
#include <cmath>
#include "gelu_omp.h"

const float sqrt_two_over_pi = 0.797885f;
const float coefficient = 0.044715f;

AlignedVector GeluOMP(const AlignedVector& input) {
    size_t size = input.size();
    AlignedVector result(size);
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        float x = input[i];
        float in_tanh = sqrt_two_over_pi * (x + coefficient * x * x * x);
        result[i] = 0.5f * x * (1.0f + std::tanh(in_tanh));
    }

    return result;
}
