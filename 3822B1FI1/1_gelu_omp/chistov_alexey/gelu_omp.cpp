#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_OVER_PI = 2.0f / PI;

AlignedVector GeluOMP(const AlignedVector& input) {
    size_t size = input.size();
    AlignedVector output(size);
    size_t i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float tanh_input = TWO_OVER_PI * (x + 0.044715f * x3);
        output[i] = 0.5f * x * (1.0f + std::tanh(tanh_input));
    }

    return output;
}