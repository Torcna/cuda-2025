#include "gelu_ocl.h"

#include <CL/cl.h>

std::vector<float> GeluOCL(const std::vector<float>& input) {
  const char* gelu = R"(__kernel void gelu(
    __global float * input,
    __global float * output,
    const unsigned int size
  ) {
    int i = get_global_id (0);
    if(i<size){
      float x = input[i];
      float in_tanh = 0.797885f * (x + 0.044715f * x * x * x);
	    output[i] = 0.5f * x * (1.0f + tanh(in_tanh));
      }
    })";

  size_t size = input.size();
  size_t input_sizeof = size * sizeof(float);
  std::vector<float> output(size);

  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, NULL);

  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

  cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, 0, 0};
  cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, NULL);

  cl_program program = clCreateProgramWithSource(context, 1, &gelu, NULL, NULL);
  const char* args = "-cl-fast-relaxed-math";
  clBuildProgram(program, 1, &device, args, NULL, NULL);

  cl_kernel kernel = clCreateKernel(program, "gelu", NULL);

  cl_mem cl_input = clCreateBuffer(context, CL_MEM_READ_ONLY, input_sizeof, NULL, NULL);
  cl_mem cl_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, input_sizeof, NULL, NULL);

  clEnqueueWriteBuffer(queue, cl_input, CL_TRUE, 0, input_sizeof, input.data(), 0, NULL, NULL);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_output);
  clSetKernelArg(kernel, 2, sizeof(int), &size);

  const size_t count = (size + 256 - 1) / 256 * 256;
  const size_t group = 256;
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &count, &group, 0, NULL, NULL);
  clFinish(queue);

  clEnqueueReadBuffer(queue, cl_output, CL_TRUE, 0, input_sizeof, output.data(), 0, NULL, NULL);

  clReleaseMemObject(cl_input);
  clReleaseMemObject(cl_output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return output;
}