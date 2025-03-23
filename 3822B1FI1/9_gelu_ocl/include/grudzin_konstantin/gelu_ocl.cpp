#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>

const char* geluKernelSource = R"CLC(
__kernel void geluKernel(__global const float* input,
                         __global float* output,
                         const int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        const float sqrt2_div_pi = 0.7978845608028654f;
        float t = tanh(sqrt2_div_pi * (x + 0.044715f * x * x * x));
        output[i] = 0.5f * x * (1.0f + t);
    }
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    int n = static_cast<int>(input.size());
    cl_int err;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return output;
    }
    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    cl_platform_id platform = platforms[0];
    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        std::cerr << "Failed to find any GPU devices." << std::endl;
        return output;
    }
    std::vector<cl_device_id> devices(numDevices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
    cl_device_id device = devices[0];
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return output;
    }
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue." << std::endl;
        clReleaseContext(context);
        return output;
    }
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * input.size(), const_cast<float*>(input.data()), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create input buffer." << std::endl;
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return output;
    }
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                     sizeof(float) * output.size(), nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create output buffer." << std::endl;
        clReleaseMemObject(d_input);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return output;
    }
    cl_program program = clCreateProgramWithSource(context, 1, &geluKernelSource, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL program." << std::endl;
        clReleaseMemObject(d_input);
        clReleaseMemObject(d_output);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return output;
    }
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Error in kernel compilation:\n" << log.data() << std::endl;
        clReleaseProgram(program);
        clReleaseMemObject(d_input);
        clReleaseMemObject(d_output);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return output;
    }
    cl_kernel kernel = clCreateKernel(program, "geluKernel", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel." << std::endl;
        clReleaseProgram(program);
        clReleaseMemObject(d_input);
        clReleaseMemObject(d_output);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return output;
    }
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &n);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arguments." << std::endl;
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(d_input);
        clReleaseMemObject(d_output);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return output;
    }
    size_t globalSize = n;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to enqueue kernel." << std::endl;
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(d_input);
        clReleaseMemObject(d_output);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return output;
    }
    clFinish(queue);
    err = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, sizeof(float) * output.size(),
                              output.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to read output buffer." << std::endl;
    }
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return output;
}
