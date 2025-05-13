#include "fft_cufft.h"

#include <cufft.h>
#include <cuda_runtime.h>

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);
    size_t complex_size = sizeof(cufftComplex);
    size_t total_size = batch * n * complex_size;

    cufftComplex* d_data;
    cudaMalloc(&d_data, total_size);
    cudaMemcpy(d_data, input.data(), total_size, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    std::vector<cufftComplex> output(batch * n);
    cudaMemcpy(output.data(), d_data, total_size, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cufftDestroy(plan);

    std::vector<float> result(batch * n * 2);
    for (int i = 0; i < batch * n; ++i) {
        result[2 * i]     = output[i].x / n;
        result[2 * i + 1] = output[i].y / n;
    }

    return result;
}
