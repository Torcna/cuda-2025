#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <cstddef>

std::vector<float> GeluOCL(const std::vector<float>& input){
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	cl::Platform platform = platforms[0];
	
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	cl::Device device = devices[0];
	
	cl::Context context(device);
    cl::CommandQueue queue(context);
	
	const char* kernel_source = R"(
	__kernel void gelu_kernel(__global const float* in, __global float* out, int n, float spi) {
		int i = get_global_id(0);
		if (i < n) {
			float x = in[i];
			float c1 = 0.5f * x;
			float c2 = spi * (x + 0.044715f * x * x * x);
			out[i] = c1 * (1.0f + tanh(c2));
		}
	}
	)";
	
	cl::Program::Sources sources;
	sources.emplace_back(kernel_source);
	cl::Program program(context, sources);
	program.build();
	cl::Kernel kernel(program, "gelu_kernel");
	
	float s2pi = sqrt(2.0f, M_PI);
	
	size_t size = input.size();
	std::vector<float> output(size);
	size_t bytes = size * sizeof(float);
	cl::Buffer buffer_in(context, CL_MEM_READ_ONLY, bytes);
	cl::Buffer buffer_out(context, CL_MEM_WRITE_ONLY, bytes);
	queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, bytes, input.data());
	kernel.setArg(0, buffer_in);
	kernel.setArg(1, buffer_out);
	kernel.setArg(2, static_cast<int>(size));
	kernel.setArg(3, static_cast<float>(s2pi));
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);
	queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, bytes, output.data());
	return output;
}
