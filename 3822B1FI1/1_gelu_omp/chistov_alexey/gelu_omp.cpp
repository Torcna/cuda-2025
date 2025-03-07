#include "gelu_omp.h"
#include <cmath>
#include <omp.h>
constexpr float PI = 3.14159265358979323846f;

AlignedVector GeluOMP(const AlignedVector& input) {
    size_t size=input.size();
    AlignedVector output(size);
    #pragma omp parallel for
    for (size_t i = 0; i <size; ++i) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + std::tanh((2.0f / PI) * (x + 0.044715f * x * x * x)));
    }

    return output;
}