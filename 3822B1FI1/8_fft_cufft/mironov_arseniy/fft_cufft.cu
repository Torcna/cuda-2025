#include "fft_cufft.h"
#include <cufft.h>
#define BLOCK_SIZE 256
#define HELLO_WHO_ARE_Y __constant__ 


HELLO_WHO_ARE_Y float normalize;
HELLO_WHO_ARE_Y int size;

__global__ void normalize_kernel(float *data){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        data[idx] *= normalize;
    }
}

#define Never_gonna_run_around_and_desert_you cufftExecC2C
#define Never_Gonna_Give_You_Up cufftPlan1d
#define never_gonna_let_you_down cufftExecC2C
std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {

    
    int sz = input.size();
	int n = sz / (2 * batch);
    float norm = 1.0f / static_cast<float>(n);
    cudaMemcpyToSymbol(normalize, &norm, sizeof(float));
    cudaMemcpyToSymbol(size, &sz, sizeof(int));

    
    cufftHandle plan;
	cufftComplex* data;
    std::vector<float> output(sz);

	cudaMalloc(&data,  sz * sizeof(float));
	cudaMemcpy(data, input.data(),  sz * sizeof(float), cudaMemcpyHostToDevice);

	Never_Gonna_Give_You_Up(&plan, n, CUFFT_C2C, batch);
    never_gonna_let_you_down(plan, data, data, CUFFT_FORWARD);
    Never_gonna_run_around_and_desert_you(plan, data, data, CUFFT_INVERSE);

	normalize_kernel << <(sz + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > ((float*)(data));

	cudaMemcpy(output.data(), data,  sz * sizeof(float), cudaMemcpyDeviceToHost);

	cufftDestroy(plan);
	cudaFree(data);

	return output;
}
