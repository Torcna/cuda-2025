#include "gelu_ocl.h"

#include <CL/cl.h>
#include <cmath>

const char* KernelSource = R"CLC(
__kernel void gelu(__global const float* input, __global float* output, int n) {
	int i = get_global_id(0);
	if (i < n) {
		float x = input[i];
		float val = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI_F) * (x + 0.044715f * x * x * x)));
		output[i] = val;
	}
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input) {


    cl_int err;
    cl_uint num_platforms;
    cl_platform_id platform = nullptr;

	clGetPlatformIDs(1, &platform, &num_platforms);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    size_t dataSize = input.size() * sizeof(float);
    cl_mem input_buff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dataSize, (void*)input.data(), &err);
    cl_mem output_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, nullptr, &err);

    cl_program program = clCreateProgramWithSource(context, 1, &KernelSource, nullptr, &err);
    clBuildProgram(program, 1, &device, "-DM_PI_F=3.14159265359", nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "gelu", &err);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buff);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buff);
    int n = input.size();
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    size_t global_size = input.size();
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);

    std::vector<float> output(input.size());
    clEnqueueReadBuffer(queue, output_buff, CL_TRUE, 0, dataSize, output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(input_buff);
    clReleaseMemObject(output_buff);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}
