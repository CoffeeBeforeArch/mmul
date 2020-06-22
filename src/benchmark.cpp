// Main benchmark function of GEMM

#include "benchmark/benchmark.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

// Function prototype for serial GEMM
void serial_gemm(const double *A, const double *B, double *C, std::size_t N);

// Function prototype for blocked GEMM
void blocked_gemm(const double *A, const double *B, double *C, std::size_t N);

// Serial GEMM benchmark
static void serial_gemm_bench(benchmark::State &s) {
  // Number Dimensions of our matrix
  std::size_t N = 1 << s.range(0);

  // Create our random number generators
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> dist(-10, 10);

  // Create input matrices
  double *A = static_cast<double *>(aligned_alloc(64, N * N * sizeof(double)));
  double *B = static_cast<double *>(aligned_alloc(64, N * N * sizeof(double)));
  double *C = static_cast<double *>(aligned_alloc(64, N * N * sizeof(double)));

  // Initialize them with random values (and C to 0)
  std::generate(A, A + N * N, [&] { return dist(rng); });
  std::generate(B, B + N * N, [&] { return dist(rng); });
  std::generate(C, C + N * N, [&] { return 0; });

  // Main benchmark loop
  for (auto _ : s) {
    serial_gemm(A, B, C, N);
  }
}
BENCHMARK(serial_gemm_bench)->DenseRange(8, 10)->Unit(benchmark::kMillisecond);

// Blocked GEMM benchmark
static void blocked_gemm_bench(benchmark::State &s) {
  // Number Dimensions of our matrix
  std::size_t N = 1 << s.range(0);

  // Create our random number generators
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> dist(-10, 10);

  // Create input matrices
  double *A = static_cast<double *>(aligned_alloc(64, N * N * sizeof(double)));
  double *B = static_cast<double *>(aligned_alloc(64, N * N * sizeof(double)));
  double *C = static_cast<double *>(aligned_alloc(64, N * N * sizeof(double)));

  // Initialize them with random values (and C to 0)
  std::generate(A, A + N * N, [&] { return dist(rng); });
  std::generate(B, B + N * N, [&] { return dist(rng); });
  std::generate(C, C + N * N, [&] { return 0; });

  // Main benchmark loop
  for (auto _ : s) {
    blocked_gemm(A, B, C, N);
  }
}
BENCHMARK(blocked_gemm_bench)->DenseRange(8, 10)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
