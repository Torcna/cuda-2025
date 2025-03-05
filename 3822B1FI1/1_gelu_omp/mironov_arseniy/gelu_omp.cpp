#include "gelu_omp.h"
#include <cmath>
#include <omp.h>
#define PI 0.636619f

AlignedVector GeluOMP(const AlignedVector& input) {
	
	AlignedVector result(input.size());
	#pragma omp parallel for
	for (size_t it = 0; it < input.size(); ++it) {
		float x = input[it];
		result[it] = 0.5f * x * (1.f + tanh(PI * (x + 0.044715f * (x * x * x))));
	}
	return result;
}
