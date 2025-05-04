#include <vector>
#include <omp.h>
#include <algorithm>

void ProcessBlock(const std::vector<float> &mat_a, const std::vector<float> &mat_b, std::vector<float> &mat_c,
                  int dim, int ir, int jr, int kr, int tile)
{
    for (int i = ir; i < std::min(ir + tile, dim); ++i)
    {
        for (int k = kr; k < std::min(kr + tile, dim); ++k)
        {
            float a_val = mat_a[i * dim + k];
            int j_end = std::min(jr + tile, dim);
#pragma omp simd
            for (int j = jr; j < j_end; ++j)
            {
                mat_c[i * dim + j] += a_val * mat_b[k * dim + j];
            }
        }
    }
}

std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b,
                                int n)
{
    std::vector<float> res(n * n, 0.0f);
    int tile = 32;

#pragma omp parallel for
    for (int ir = 0; ir < n; ir += tile)
    {
        for (int jr = 0; jr < n; jr += tile)
        {
            for (int kr = 0; kr < n; kr += tile)
            {
                ProcessBlock(a, b, res, n, ir, jr, kr, tile);
            }
        }
    }

    return res;
}
