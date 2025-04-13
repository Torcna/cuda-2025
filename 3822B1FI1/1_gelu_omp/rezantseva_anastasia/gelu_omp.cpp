#include "gelu_omp.h"
#include <cmath>
#include <omp.h>
//0.7978845608f = std::sqrt(2.0f / M_PI) 
AlignedVector GeluOMP(const AlignedVector& input) {
    AlignedVector output(input.size());
    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float arg = 0.7978845608f * (x + 0.044715f * x * x * x);
        output[i] = 0.5f * x * (1.0f + std::tanh(arg));
    }
    return output;
}