// A DMMul reference implementation using MKL

#include <algorithm>
#include <random>

#include "benchmark/benchmark.h"
#include "mkl/mkl.h"

// Blocked MMul benchmark
static void blocked_mmul_bench(benchmark::State &s) {
  // Number Dimensions of our matrix
  std::size_t N = s.range(0);

  // Create our random number generators
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(-10, 10);

  // Create input matrices
  float *A = (float *)mkl_malloc(N * N * sizeof(float), 64);
  float *B = (float *)mkl_malloc(N * N * sizeof(float), 64);
  float *C = (float *)mkl_malloc(N * N * sizeof(float), 64);

  // MMul scaling constants
  float alpha = 1.0;
  float beta = 0.0;

  // Initialize them with random values (and C to 0)
  std::generate(A, A + N * N, [&] { return dist(rng); });
  std::generate(B, B + N * N, [&] { return dist(rng); });
  std::generate(C, C + N * N, [&] { return 0; });

  // Main benchmark loop
  for (auto _ : s) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A, N,
                B, N, beta, C, N);
  }

  // Free memory
  mkl_free(A);
  mkl_free(B);
  mkl_free(C);
}
BENCHMARK(blocked_mmul_bench)
    ->Arg(384)
    ->Arg(768)
    ->Arg(1152)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
