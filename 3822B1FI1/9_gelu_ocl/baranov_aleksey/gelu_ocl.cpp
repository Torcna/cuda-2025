#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cmath>
#include <memory>
#include <iostream>
#include <vector>
#include <string>

std::vector<float> GeluOCL(const std::vector<float>& input) {
    std::string kernelSource =R"(
    __kernel void Kernel(__global const float* input, __global float* output, int size) {
        int index = get_global_id(0);
        if (index < size) {
            const float value = input[index];
            output[index] = value / (1.0f + exp(-1.59577f * (value + 0.044715f * value * value * value)));
        }
    }
    )";

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform selectedPlatform = platforms.front();

    std::vector<cl::Device> devices;
    selectedPlatform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device selectedDevice = devices.front();

    cl::Context context(selectedDevice);
    cl::CommandQueue queue(context);
    
    cl::Program::Sources sources;
    sources.emplace_back(kernelSource);
    cl::Program program(context, sources);
    program.build();

    cl::Kernel kernel(program, "Kernel");
    size_t inputSize = input.size();
    size_t bufferSize = inputSize * sizeof(float);

    cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, bufferSize);
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, bufferSize);
    
    queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, bufferSize, input.data());
    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, outputBuffer);
    kernel.setArg(2, static_cast<int>(inputSize));
    
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputSize), cl::NullRange);
    
    std::vector<float> output(inputSize);
    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, bufferSize, output.data());
    
    return output;
}
