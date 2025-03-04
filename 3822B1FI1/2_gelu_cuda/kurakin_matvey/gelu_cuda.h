#ifndef __GELU_CUDA_H
#define __GELU_CUDA_H

#pragma once
#include <cuda_runtime.h>
#include <vector>

std::vector<float> GeluCUDA(const std::vector<float>& input);

__global__ void Gelu(std::vector<float>& input, std::vector<float>& result);

#endif // __GELU_CUDA_H
