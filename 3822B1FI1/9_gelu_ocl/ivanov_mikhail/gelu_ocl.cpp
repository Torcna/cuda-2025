#include "gelu_ocl.h"
#include <CL/cl.h>

const char* kernelSource = R"(
__kernel void gelu(__global const float* input, __global float* output, const int n)
{
    const float SQRT2DIVPI = 0.7978845f;
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        output[i] = 0.5f * x * (1.f + tanh(SQRT2DIVPI * (x + 0.044715f * (x * x * x))));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input) {
  size_t inputSize = input.size();
  size_t size = inputSize * sizeof(float);
  std::vector<float> result(inputSize);

  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, nullptr);

  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

  cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
  cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, nullptr);

  cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
  clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  
  cl_kernel kernel = clCreateKernel(program, "gelu", nullptr);

  cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, (void*)input.data(), nullptr);
  cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, nullptr, nullptr);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
  clSetKernelArg(kernel, 2, sizeof(int), &inputSize);

  clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &inputSize, nullptr, 0, nullptr, nullptr);

  clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, size, result.data(), 0, nullptr, nullptr);

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseMemObject(inputBuffer);
  clReleaseMemObject(outputBuffer);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return result;
}