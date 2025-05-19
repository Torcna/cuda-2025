#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cstring>

const char* geluKernelSource = R"CLC(
__kernel void geluKernel(__global const float* input, __global float* output, const int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        output[i] = x / (1.0f + exp(-1.59577f * (x + 0.044715f * x * x * x)));
    }
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input) {
    int n = static_cast<int>(input.size());
    std::vector<float> output(n);

    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    cl_platform_id platform = platforms[0];

    cl_uint numDevices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);

    std::vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
    cl_device_id device = devices[0];

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);

    cl_queue_properties properties[] = { 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, properties, nullptr);

    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * n, nullptr, nullptr);
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n, nullptr, nullptr);

    clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, sizeof(float) * n, input.data(), 0, nullptr, nullptr);

    cl_program program = clCreateProgramWithSource(context, 1, &geluKernelSource, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "geluKernel", nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    size_t globalSize = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, sizeof(float) * n, output.data(), 0, nullptr, nullptr);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}