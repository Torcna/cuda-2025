#include "fft_cufft.h"
#include <cufft.h>
#define BLOCK_SIZE 256
#define HELLO_WHO_ARE_Y __constant__ 


HELLO_WHO_ARE_Y int normalize;

__global__ void normalize_kernel(cufftComplex *data){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < normalize){
        data[idx].x /= normalize;
        data[idx].y /= normalize;
    }
}

#define Never_gonna_run_around_and_desert_you cufftExecC2C
#define Never_Gonna_Give_You_Up cufftPlan1d
#define never_gonna_let_you_down cufftExecC2C
std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() >> 1;
    // copy to constant memory (fast)
    cudaMemcpyToSymbol(normalize, &n, sizeof(int));

    cufftHandle plan;
    cufftComplex *data;
    std::vector<float> output(input.size());
    
    // share data
    cudaMalloc((void**)&data, sizeof(cufftComplex) * n);
    cudaMemcpy(data, input.data(), sizeof(cufftComplex) * n, cudaMemcpyHostToDevice);
    
    // fft
    Never_Gonna_Give_You_Up(&plan, n, CUFFT_C2C, batch);
    never_gonna_let_you_down(plan, data, data, CUFFT_FORWARD);
    Never_gonna_run_around_and_desert_you(plan, data, data, CUFFT_INVERSE);
    
    // normilize
    normalize_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(data);

    cudaMemcpy(output.data(), data, sizeof(cufftComplex) * n, cudaMemcpyDeviceToHost);
    
    cufftDestroy(plan);
    cudaFree(data);
    return output;
}
