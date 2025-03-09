#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

AlignedVector GeluOMP(const AlignedVector& input) {
	float sqrt2pi=sqrt(2/acos(-1));
	AlignedVector result(input.size());
	auto res_ptr=result.data();
	auto inp_ptr=input.data();
	#pragma omp parallel for
	for (std::size_t it = 0; it < input.size(); ++it) {
		float x = inp_ptr[it];
		res_ptr[it] = 0.5f * x * (1.f + std::tanh(sqrt2pi * (x + 0.044715f * (x * x * x))));
	}
	return result;
}
