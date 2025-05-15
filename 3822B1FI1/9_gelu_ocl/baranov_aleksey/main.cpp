#include <iostream>
#include <vector>
#include "../../9_gelu_ocl/baranov_aleksey/gelu_ocl.h" // Если у вас функция отделена в заголовочном файле.
                                                       // Либо включите сам код функции сюда, если хотите сделать всё в одном файле.

/// Функция GeluOCL реализована, как мы обсуждали ранее.

int main()
{
  // Пример входных данных
  std::vector<float> input = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};

  // Вызов функции GeluOCL
  std::vector<float> output = GeluOCL(input);

  // Вывод результата
  std::cout << "Результаты Gelu: ";
  for (float val : output)
  {
    std::cout << val << " ";
  }
  std::cout << std::endl;

  return 0;
}