#include "gelu_ocl.h"
#include <CL/opencl.hpp>
#include <iostream>
#include <string>

std::string LoadKernelSrc() {
    return R"(
    __kernel void GeluApply(__global const float* in, __global float* out, int count) {
        int i = get_global_id(0);
        if (i < count) {
            float x = in[i];
            out[i] = x / (1.0f + exp(-1.59577f * (x + 0.044715f * x * x * x)));
        }
    }
    )";
}

std::vector<float> GeluOCL(const std::vector<float>& inputData) {
    std::string src = LoadKernelSrc();

    std::vector<cl::Platform> oclPlatforms;
    cl::Platform::get(&oclPlatforms);
    auto& platform = oclPlatforms[0];

    std::vector<cl::Device> gpuList;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &gpuList);
    cl::Device gpu = gpuList[0];

    cl::Context ctx(gpu);
    cl::CommandQueue cmdQueue(ctx);

    cl::Program::Sources kernelSources;
    kernelSources.emplace_back(src);
    cl::Program prog(ctx, kernelSources);
    prog.build();

    cl::Kernel kernelFunc(prog, "GeluApply");

    size_t elemCount = inputData.size();
    size_t memSize = elemCount * sizeof(float);

    cl::Buffer bufInput(ctx, CL_MEM_READ_ONLY, memSize);
    cl::Buffer bufOutput(ctx, CL_MEM_WRITE_ONLY, memSize);

    cmdQueue.enqueueWriteBuffer(bufInput, CL_TRUE, 0, memSize, inputData.data());

    kernelFunc.setArg(0, bufInput);
    kernelFunc.setArg(1, bufOutput);
    kernelFunc.setArg(2, static_cast<int>(elemCount));

    cl::NDRange globalRange(elemCount);
    cmdQueue.enqueueNDRangeKernel(kernelFunc, cl::NullRange, globalRange, cl::NullRange);

    std::vector<float> outputData(elemCount);
    cmdQueue.enqueueReadBuffer(bufOutput, CL_TRUE, 0, memSize, outputData.data());

    return outputData;
}
