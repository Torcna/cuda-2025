#define _USE_MATH_DEFINES
#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

AlignedVector GeluOMP(const AlignedVector& input) {
    AlignedVector output(input.size());

    #pragma omp parallel for
    for (std::size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float c1 = 0.5f * x;
        float c2 = std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x);
        output[i] = c1 * (1.0f + std::tanh(c2));
    }
    return output;
}
