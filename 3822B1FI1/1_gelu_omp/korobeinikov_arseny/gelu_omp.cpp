#include "gelu_omp.h"
#include <omp.h>
#include <cmath>

inline float fast_tanh(float x) {
    const float a = 1.0f;
    const float b = 0.5f;
    float x2 = x * x;
    return x * (a + x2) / (a + b * x2 + x2 * x2);
}

float Gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    float cubic = x * x * x;
    float arg = sqrt_2_over_pi * (x + coeff * cubic);

    float tanh_val = (x <= 0.0f) ? std::tanh(arg)
                     : (arg > 4.0f) ? 1.0f
                     : fast_tanh(arg);

    return 0.5f * x * (1.0f + tanh_val);
}

AlignedVector GeluOMP(const AlignedVector& input) {
    AlignedVector output(input.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(input.size()); ++i) {
        output[i] = Gelu(input[i]);
    }

    return output;
}