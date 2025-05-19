#include "gelu_ocl.h"
#include <CL/cl.h>

const char* kernelSource = R"(
__kernel void gelu(__global const float* input, __global float* output, const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        float x = input[i];
        output[i] = 0.5f * x * (1.f + tanh(0.7978845f * (x + 0.044715f * (x * x * x))));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input) {
  size_t inputSize = input.size();
  size_t size = inputSize * sizeof(float);
  std::vector<float> result(inputSize);

  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;
  cl_mem inputBuffer;
  cl_mem outputBuffer;


  clGetPlatformIDs(1, &platform, nullptr);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
  queue = clCreateCommandQueueWithProperties(context, device, 0, nullptr);

  program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
  clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  
  kernel = clCreateKernel(program, "gelu", nullptr);

  inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, size, nullptr, nullptr);
  outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, nullptr, nullptr);

  clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, size, input.data(), 0, NULL, NULL);

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