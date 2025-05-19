#define CL_TARGET_OPENCL_VERSION 220
#include "gelu_ocl.h"
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cstring>

static const char* gelu_kernel_source = R"CLC(
__kernel void gelu_kernel(__global const float* input, __global float* output, int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        float c = 0.044715f;
        float sqrt_2_over_pi = 0.7978845608f;
        float x3 = x * x * x;
        float t = sqrt_2_over_pi * (x + c * x3);
        output[i] = 0.5f * x * (1.0f + tanh(t));
    }
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input) {
    int n = static_cast<int>(input.size());
    if (n == 0) return {};

    cl_int err;

    // Get platform
    cl_platform_id platform_id = nullptr;
    err = clGetPlatformIDs(1, &platform_id, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to get OpenCL platform");

    // Get GPU device
    cl_device_id device_id = nullptr;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to get OpenCL GPU device");

    // Create context
    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create OpenCL context");

    // Create command queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create OpenCL command queue");

    // Create buffers
    cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), (void*)input.data(), &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create input buffer");

    cl_mem output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create output buffer");

    // Create program
    cl_program program = clCreateProgramWithSource(context, 1, &gelu_kernel_source, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create OpenCL program");

    // Build program
    err = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "OpenCL build log:\n" << log.data() << std::endl;
        throw std::runtime_error("Failed to build OpenCL program");
    }

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create OpenCL kernel");

    // Set kernel args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buf);
    clSetKernelArg(kernel, 2, sizeof(int), &n);

    // Enqueue kernel
    size_t global_work_size = n;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to enqueue OpenCL kernel");

    // Read result
    std::vector<float> output(n);
    err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, n * sizeof(float), output.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to read OpenCL buffer");

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}
