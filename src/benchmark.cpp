// Main benchmark function of GEMM

#include "benchmark/benchmark.h"

#include <algorithm>
#include <random>
#include <vector>

// Function prototype for serial GEMM
void serial_gemm(std::vector<double> &A, std::vector<double> &B,
                 std::vector<double> &C, std::size_t N);

static void serial_gemm_bench(benchmark::State &s) {
  // Number Dimensions of our matrix
  std::size_t N = s.range(0);

  // Create our random number generators
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> dist(-10, 10);

  // Create input matrices
  std::vector<double> A(N * N);
  std::vector<double> B(N * N);
  std::vector<double> C(N * N);

  // Initialize them with random values
  std::generate(begin(A), end(A), [&] { return dist(rng); });
  std::generate(begin(B), end(B), [&] { return dist(rng); });

  // Main benchmark loop
  for (auto _ : s) {
    serial_gemm(A, B, C, N);
  }
}
BENCHMARK(serial_gemm_bench)->DenseRange(8, 10);

BENCHMARK_MAIN();
