#include <vector>
#include <cuda_runtime.h>
#include <iostream>

__device__ float gelu_accurate(float x)
{
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coef = 0.044715f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coef * x_cubed);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void gelu_kernel(const float *in, float *out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        out[idx] = gelu_accurate(in[idx]);
    }
}

std::vector<float> GeluCUDA(const std::vector<float> &input)
{
    if (input.empty())
    {
        return {};
    }

    const int size = input.size();
    float *d_input, *d_output;

    // Выделяем память на GPU
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Копируем данные на GPU
    cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Настраиваем параметры запуска ядра
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Запускаем ядро
    gelu_kernel<<<gridSize, blockSize>>>(d_input, d_output, size);

    // Копируем результат обратно на CPU
    std::vector<float> result(size);
    cudaMemcpy(result.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Освобождаем память
    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}
int main()
{
    try
    {
        // Тест корректности
        std::vector<float> test = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, 2.5f}; // Нечётный размер
        auto result = GeluCUDA(test);

        std::cout << "Results:\n";
        for (size_t i = 0; i < test.size(); ++i)
        {
            float expected = 0.5f * test[i] * (1.0f + tanhf(0.79788456f * (test[i] + 0.044715f * test[i] * test[i] * test[i])));
            std::cout << test[i] << " -> " << result[i]
                      << " (expected: " << expected << ")"
                      << (fabs(result[i] - expected) < 1e-5f ? "" : " [ERROR]") << "\n";
        }

        // Тест производительности
        const int N = 1 << 24;
        std::vector<float> large_input(N, 1.0f);

        auto start = std::chrono::high_resolution_clock::now();
        auto large_result = GeluCUDA(large_input);
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "\nPerformance: "
                  << std::chrono::duration<double>(end - start).count()
                  << " seconds for " << N << " elements\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}