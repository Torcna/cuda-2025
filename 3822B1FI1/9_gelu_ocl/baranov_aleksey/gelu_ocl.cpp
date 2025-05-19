#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cmath>
#include <memory>
#include <iostream>
#include <vector>
#include <string>

std::vector<float> GeluOCL(const std::vector<float> &input)
{
    std::vector<float> output(input.size(), 0.0f);
    int n = static_cast<int>(input.size());

    auto clReleaseContextWrapper = [](cl_context ctx)
    { if (ctx) clReleaseContext(ctx); };
    auto clReleaseQueueWrapper = [](cl_command_queue q)
    { if (q) clReleaseCommandQueue(q); };
    auto clReleaseProgramWrapper = [](cl_program p)
    { if (p) clReleaseProgram(p); };
    auto clReleaseKernelWrapper = [](cl_kernel k)
    { if (k) clReleaseKernel(k); };
    auto clReleaseMemWrapper = [](cl_mem m)
    { if (m) clReleaseMemObject(m); };

    std::unique_ptr<std::remove_pointer<cl_context>::type, decltype(clReleaseContextWrapper)>
        context(nullptr, clReleaseContextWrapper);
    std::unique_ptr<std::remove_pointer<cl_command_queue>::type, decltype(clReleaseQueueWrapper)>
        queue(nullptr, clReleaseQueueWrapper);
    std::unique_ptr<std::remove_pointer<cl_program>::type, decltype(clReleaseProgramWrapper)>
        program(nullptr, clReleaseProgramWrapper);
    std::unique_ptr<std::remove_pointer<cl_kernel>::type, decltype(clReleaseKernelWrapper)>
        kernel(nullptr, clReleaseKernelWrapper);
    std::unique_ptr<std::remove_pointer<cl_mem>::type, decltype(clReleaseMemWrapper)>
        d_input(nullptr, clReleaseMemWrapper), d_output(nullptr, clReleaseMemWrapper);

    try
    {
        cl_uint numPlatforms = 0;
        clGetPlatformIDs(0, nullptr, &numPlatforms);
        std::vector<cl_platform_id> platforms(numPlatforms);
        clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
        cl_platform_id platform = platforms[0];

        cl_uint numDevices = 0;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        std::vector<cl_device_id> devices(numDevices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
        cl_device_id device = devices[0];

        cl_int err;
        context.reset(clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err));

#if CL_TARGET_OPENCL_VERSION >= 200
        queue.reset(clCreateCommandQueueWithProperties(context.get(), device, 0, &err));
#else
        queue.reset(clCreateCommandQueue(context.get(), device, 0, &err));
#endif

        d_input.reset(clCreateBuffer(context.get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(float) * input.size(), const_cast<float *>(input.data()), &err));
        d_output.reset(clCreateBuffer(context.get(), CL_MEM_WRITE_ONLY,
                                      sizeof(float) * output.size(), nullptr, &err));

        const char *geluKernelSource = R"CLC(
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

        program.reset(clCreateProgramWithSource(context.get(), 1, &geluKernelSource, nullptr, &err));
        clBuildProgram(program.get(), 1, &device, nullptr, nullptr, nullptr);

        kernel.reset(clCreateKernel(program.get(), "geluKernel", &err));
        clSetKernelArg(kernel.get(), 0, sizeof(cl_mem), d_input.get());
        clSetKernelArg(kernel.get(), 1, sizeof(cl_mem), d_output.get());
        clSetKernelArg(kernel.get(), 2, sizeof(int), &n);

        size_t globalSize = static_cast<size_t>(n);
        clEnqueueNDRangeKernel(queue.get(), kernel.get(), 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
        clFinish(queue.get());

        clEnqueueReadBuffer(queue.get(), d_output.get(), CL_TRUE, 0, sizeof(float) * output.size(),
                            output.data(), 0, nullptr, nullptr);
    }
    catch (...)
    {
        return output;
    }

    return output;
}
