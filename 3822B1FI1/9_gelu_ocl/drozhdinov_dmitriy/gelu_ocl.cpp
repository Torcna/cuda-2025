#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

std::vector<float> GeluOCL(const std::vector<float>& input){
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	cl::Platform platform = platforms.front();
	
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	cl::Device device = devices.front();
	
	cl::Context context(device);
    cl::CommandQueue queue(context);
	
	std::string kernel = R"(
	__kernel void kernel(__global const float* in, __global float* out, int n, float spi) {
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
	sources.emplace_back(std::move(kernel));
	cl::Program program(context, sources);
	program.build();
	cl::Kernel _kernel(program, "kernel");
	
	size_t size = input.size();
	std::vector<float> output(size);
	size_t bytes = size * sizeof(*input.data());
	cl::Buffer buffer_in(context, CL_MEM_READ_ONLY, bytes);
	cl::Buffer buffer_out(context, CL_MEM_WRITE_ONLY, bytes);
	queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, bytes, input.data());
	_kernel.setArg(0, buffer_in);
	_kernel.setArg(1, buffer_out);
	_kernel.setArg(2, static_cast<int>(size));
	_kernel.setArg(3, static_cast<float>(std::sqrt(2.0f / M_PI)));
	queue.enqueueNDRangeKernel(_kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);
	queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, bytes, output.data());
	return output;
}