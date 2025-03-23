#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#define BLOCK_SIZE 256

__global__ void normalizeKernel(cufftComplex* data, int totalComplex, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalComplex) {
        data[idx].x /= n;
        data[idx].y /= n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int totalFloats = input.size();               
    int n = totalFloats / (2 * batch);            
    int totalComplex = totalFloats / 2;           

    cufftHandle plan;
    // Создаём план для 1D FFT для batch сигналов
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    // Выделяем память на устройстве для комплексных данных
    cufftComplex* d_data = nullptr;
    cudaMalloc((void**)&d_data, sizeof(cufftComplex) * totalComplex);

    // Переносим данные на устройство. 
    cudaMemcpy(d_data, input.data(), totalFloats * sizeof(float), cudaMemcpyHostToDevice);

    // Выполняем прямое преобразование (CUFFT_FORWARD)
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD)

    // Выполняем обратное преобразование (CUFFT_INVERSE)
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalComplex + threadsPerBlock - 1) / threadsPerBlock;
    normalizeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, totalComplex, n);

    std::vector<float> result(totalFloats);
    cudaMemcpy(result.data(), d_data, totalFloats * sizeof(float), cudaMemcpyDeviceToHost);

    // Освобождаем ресурсы
    cudaFree(d_data);
    cufftDestroy(plan);

    return result;
}
