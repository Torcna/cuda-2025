#include "naive_gemm_omp.h"
#include <omp.h>
#include <chrono>

std::vector<float> NaiveGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b,
                                int size)
{
    std::vector<float> result(size * size, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; ++i)
    {
        int row_offset = i * size;
        for (int j = 0; j < size; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < size; ++k)
            {
                sum += a[row_offset + k] * b[k * size + j];
            }
            result[row_offset + j] = sum;
        }
    }
    return result;
}