#include <omp.h>
#include <cmath>
#include "gelu_omp.h"

float Gelu(float x){
    return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / 3.1415) * (x + 0.044715 * std::pow(x, 3))));
}

AlignedVector GeluOMP(const AlignedVector& input) {
    AlignedVector res(input.size());
	size_t i;
#pragma omp parallel
	{
#pragma omp for
		for(i = 0; i < input.size(); ++i){
			res[i] = Gelu(input[i]);
		}
	}
	return res;
}
