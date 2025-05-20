#define _USE_MATH_DEFINES
#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

AlignedVector GeluOMP(const AlignedVector &input)
{
    size_t size = input.size();
    AlignedVector res(size);
    float var1 = M_SQRT1_2 * M_2_SQRTPI;
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i)
    {
        float x = input[i];
        res[i] = 0.5f * x * (1.f + std::tanh(var1 * (x + 0.044715f * (x * x * x))));
    }
    return res;
}