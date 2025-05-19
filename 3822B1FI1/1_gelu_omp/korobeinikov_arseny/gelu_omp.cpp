#define _USE_MATH_DEFINES
#include "gelu_omp.h"
#include <cmath>
#include <omp.h>
#include <algorithm>

namespace
{
    constexpr float GELU_COEF = 0.044715f;
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;

    inline float fast_gelu_approx(float x) noexcept
    {

        const float x_cubed = x * x * x;
        const float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed);

        if (inner > 4.0f)
            return x;
        if (inner < -4.0f)
            return 0.0f;

        const float tanh_val = 1.0f - 2.0f / (1.0f + std::exp(2.0f * inner));
        return 0.5f * x * (1.0f + tanh_val);
    }
}

AlignedVector GeluOMP(const AlignedVector &input)
{
    if (input.empty())
    {
        return {};
    }

    const size_t size = input.size();
    AlignedVector res(size);
    const float *in_data = input.data();
    float *out_data = res.data();

    const size_t block_size = std::max<size_t>(256, size / (4 * omp_get_max_threads()));

#pragma omp parallel for schedule(dynamic, block_size)
    for (size_t i = 0; i < size; ++i)
    {
        out_data[i] = fast_gelu_approx(in_data[i]);
    }

    return res;
}