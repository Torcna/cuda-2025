#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int N = static_cast<int>(input.size() / (2 * batch)); // length of one signal (complex)
    if (N == 0 || batch == 0) return {};

    cufftHandle plan;
    cufftComplex *d_data;
    size_t size = input.size() * sizeof(float);

    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, input.data(), size, cudaMemcpyHostToDevice);

    if (cufftPlan1d(&plan, N, CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        cudaFree(d_data);
        throw std::runtime_error("CUFFT plan creation failed");
    }

    // Forward transform
    if (cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("CUFFT exec forward failed");
    }

    // Inverse transform
    if (cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(d_data);
        throw std::runtime_error("CUFFT exec inverse failed");
    }

    // Normalize by N
    int total_complex = N * batch;
    int total_float = total_complex * 2;
    std::vector<float> output(total_float);

    cudaMemcpy(output.data(), d_data, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < total_float; ++i) {
        output[i] /= N;
    }

    cufftDestroy(plan);
    cudaFree(d_data);

    return output;
}
