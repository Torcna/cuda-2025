#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

AlignedVector GeluOMP(const AlignedVector& input) {
    AlignedVector result(input.size());
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
	
    #pragma omp parallel for
    for (std::size_t i = 0; i < input.size(); i++) {
        float x = input[i];
        float x_cubed = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        result[i] = 0.5f * x * (1.0f + std::tanh(tanh_arg));
    }
    return result;
}
