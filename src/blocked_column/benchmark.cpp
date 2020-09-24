// Main benchmark function of GEMM

#include "benchmark/benchmark.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

// Function prototype for blocked column serial GEMM
void blocked_column_mmul(const double *A, const double *B, double *C,
                         std::size_t N);

// Function prototype for blocked column parallel GEMM w/o atomics
void blocked_column_parallel_atomic_mmul(const double *A, const double *B,
                                         double *C, std::size_t N,
                                         std::atomic<uint64_t> &pos);

// Blocked column GEMM with aligned memory benchmark
static void blocked_column_aligned_mmul_bench(benchmark::State &s) {
  // Number Dimensions of our matrix
  std::size_t N = s.range(0);

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
    blocked_column_mmul(A, B, C, N);
  }

  // Free memory
  free(A);
  free(B);
  free(C);
}
BENCHMARK(blocked_column_aligned_mmul_bench)
    ->Arg(384)
    ->Arg(768)
    ->Arg(1152)
    ->Unit(benchmark::kMillisecond);

// Parallel blocked column GEMM benchmark
static void parallel_blocked_column_atomic_mmul_bench(benchmark::State &s) {
  // Number Dimensions of our matrix
  std::size_t N = s.range(0);

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

  // Set up for launching threads
  std::size_t num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  // Main benchmark loop
  for (auto _ : s) {
    // Atomic uint64_t to keep track of position
    std::atomic<uint64_t> pos{0};

    // Launch threads
    for (std::size_t i = 0; i < num_threads - 1; i++) {
      threads.emplace_back(
          [&] { blocked_column_parallel_atomic_mmul(A, B, C, N, pos); });
    }
    blocked_column_parallel_atomic_mmul(A, B, C, N, pos);

    // Wait for all threads to complete
    for (auto &t : threads) t.join();

    // Clear the threads each iteration of the benchmark
    threads.clear();
  }

  // Free memory
  free(A);
  free(B);
  free(C);
}
BENCHMARK(parallel_blocked_column_atomic_mmul_bench)
    ->Arg(384)
    ->Arg(768)
    ->Arg(1152)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();
