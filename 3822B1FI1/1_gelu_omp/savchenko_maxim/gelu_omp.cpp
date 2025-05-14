#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

AlignedVector GeluOMP(const AlignedVector& input) {
    AlignedVector output(input.size());
	
	const float c1 = 0.7978845608f; // std::sqrt(2.0f / M_PI) 
	const float c2 = 044715f;
	
    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float gelu = 0.5f * x * (1.0f + std::tanh(c1 * (x + c2 * x * x * x)));
        output[i] = gelu;
    }

    return output;
}
