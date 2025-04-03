#include "gelu_ocl.h"

#include <CL/opencl.hpp>
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
}