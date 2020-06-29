// Main benchmark function of GEMM

#include "benchmark/benchmark.h"

#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

// Function prototype for serial GEMM
void serial_gemm(const double *A, const double *B, double *C, std::size_t N);

// Function prototype for blocked GEMM
void blocked_gemm(const double *A, const double *B, double *C, std::size_t N);

// Function for naive parallelized GEMM
void parallel_gemm(const double *A, const double *b, double *C, std::size_t N,
                   std::size_t start_row, std::size_t end_row);

// Function for blocked parallelized GEMM
void blocked_parallel_gemm(const double *A, const double *b, double *C,
                           std::size_t N, std::size_t start_row,
                           std::size_t end_row);

// Function prototype for blocked column serial GEMM
void blocked_column_gemm(const double *A, const double *B, double *C,
                         std::size_t N);

// Function prototype for blocked column parallel GEMM w/o atomics
void blocked_column_parallel_atomic_gemm(const double *A, const double *B,
                                         double *C, std::size_t N,
                                         std::atomic<uint64_t> &pos);

// Function prototype for blocked column fwd/bwd serial GEMM
void blocked_column_fwdbwd_gemm(const double *A, const double *B, double *C,
                                std::size_t N);

// Function prototype for blocked column fwd/bwd parallel GEMM
void blocked_column_fwdbwd_parallel_atomic_gemm(const double *A,
                                                const double *B, double *C,
                                                std::size_t N,
                                                std::atomic<uint64_t> &pos);

// Function prototype for blocked column parallelized GEMM w/ atomics
void blocked_column_parallel_gemm(const double *A, const double *B, double *C,
                                  std::size_t N, std::size_t start_col,
                                  std::size_t end_col);

// Serial GEMM benchmark
static void serial_gemm_bench_power_two(benchmark::State &s) {
  // Number Dimensions of our matrix
  std::size_t N = 1 << s.range(0);

  // Create our random number generators
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> dist(-10, 10);

  // Create input matrices
  double *A = new double[N * N];
  double *B = new double[N * N];
  double *C = new double[N * N];

  // Initialize them with random values (and C to 0)
  std::generate(A, A + N * N, [&] { return dist(rng); });
  std::generate(B, B + N * N, [&] { return dist(rng); });
  std::generate(C, C + N * N, [&] { return 0; });

  // Main benchmark loop
  for (auto _ : s) {
    serial_gemm(A, B, C, N);
  }

  // Free memory
  delete[] A;
  delete[] B;
  delete[] C;
}
BENCHMARK(serial_gemm_bench_power_two)
    ->DenseRange(8, 10)
    ->Unit(benchmark::kMillisecond);

// Serial GEMM benchmark
static void serial_gemm_bench(benchmark::State &s) {
  // Number Dimensions of our matrix
  std::size_t N = (1 << s.range(0)) + 16;

  // Create our random number generators
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> dist(-10, 10);

  // Create input matrices
  double *A = new double[N * N];
  double *B = new double[N * N];
  double *C = new double[N * N];

  // Initialize them with random values (and C to 0)
  std::generate(A, A + N * N, [&] { return dist(rng); });
  std::generate(B, B + N * N, [&] { return dist(rng); });
  std::generate(C, C + N * N, [&] { return 0; });

  // Main benchmark loop
  for (auto _ : s) {
    serial_gemm(A, B, C, N);
  }

  // Free memory
  delete[] A;
  delete[] B;
  delete[] C;
}
BENCHMARK(serial_gemm_bench)->DenseRange(8, 10)->Unit(benchmark::kMillisecond);

// Parallel GEMM benchmark
static void parallel_gemm_bench(benchmark::State &s) {
  // Number Dimensions of our matrix
  std::size_t N = (1 << s.range(0)) + 16;

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

  // Calculate values to pass to threads
  // Assumed to be divisable by num_threads (evenly)
  std::size_t n_rows = N / num_threads;

  // Main benchmark loop
  for (auto _ : s) {
    // Launch threads
    std::size_t end_row = 0;
    for (std::size_t i = 0; i < num_threads - 1; i++) {
      auto start_row = i * n_rows;
      end_row = start_row + n_rows;
      threads.emplace_back(
          [&] { parallel_gemm(A, B, C, N, start_row, end_row); });
    }
    parallel_gemm(A, B, C, N, end_row, end_row + n_rows);

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
BENCHMARK(parallel_gemm_bench)
    ->DenseRange(8, 10)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Blocked GEMM benchmark
static void blocked_gemm_bench(benchmark::State &s) {
  // Number Dimensions of our matrix
  std::size_t N = (1 << s.range(0)) + 16;

  // Create our random number generators
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<double> dist(-10, 10);

  // Create input matrices
  double *A = new double[N * N];
  double *B = new double[N * N];
  double *C = new double[N * N];

  // Initialize them with random values (and C to 0)
  std::generate(A, A + N * N, [&] { return dist(rng); });
  std::generate(B, B + N * N, [&] { return dist(rng); });
  std::generate(C, C + N * N, [&] { return 0; });

  // Main benchmark loop
  for (auto _ : s) {
    blocked_gemm(A, B, C, N);
  }

  // Free memory
  delete[] A;
  delete[] B;
  delete[] C;
}
BENCHMARK(blocked_gemm_bench)->DenseRange(8, 10)->Unit(benchmark::kMillisecond);

// Blocked GEMM with aligned memory benchmark
static void blocked_aligned_gemm_bench(benchmark::State &s) {
  // Number Dimensions of our matrix
  std::size_t N = (1 << s.range(0)) + 16;

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

  // Free memory
  free(A);
  free(B);
  free(C);
}
BENCHMARK(blocked_aligned_gemm_bench)
    ->DenseRange(8, 10)
    ->Unit(benchmark::kMillisecond);

// Parallel blocked GEMM benchmark
static void parallel_blocked_gemm_bench(benchmark::State &s) {
  // Number Dimensions of our matrix
  std::size_t N = (1 << s.range(0)) + 16;

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

  // Calculate values to pass to threads
  // Assumed to be divisable by num_threads (evenly)
  std::size_t n_rows = N / num_threads;

  // Main benchmark loop
  for (auto _ : s) {
    // Launch threads
    std::size_t end_row = 0;
    for (std::size_t i = 0; i < num_threads - 1; i++) {
      auto start_row = i * n_rows;
      end_row = start_row + n_rows;
      threads.emplace_back(
          [&] { blocked_parallel_gemm(A, B, C, N, start_row, end_row); });
    }
    blocked_parallel_gemm(A, B, C, N, end_row, end_row + n_rows);

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
BENCHMARK(parallel_blocked_gemm_bench)
    ->DenseRange(8, 10)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Blocked column GEMM with aligned memory benchmark
static void blocked_column_aligned_gemm_bench(benchmark::State &s) {
  // Number Dimensions of our matrix
  std::size_t N = (1 << s.range(0)) + 16;

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
    blocked_column_gemm(A, B, C, N);
  }

  // Free memory
  free(A);
  free(B);
  free(C);
}
BENCHMARK(blocked_column_aligned_gemm_bench)
    ->DenseRange(8, 10)
    ->Unit(benchmark::kMillisecond);

// Parallel blocked column GEMM benchmark
static void parallel_blocked_column_atomic_gemm_bench(benchmark::State &s) {
  // Number Dimensions of our matrix
  std::size_t N = (1 << s.range(0)) + 16;

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
          [&] { blocked_column_parallel_atomic_gemm(A, B, C, N, pos); });
    }
    blocked_column_parallel_atomic_gemm(A, B, C, N, pos);

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
BENCHMARK(parallel_blocked_column_atomic_gemm_bench)
    ->DenseRange(8, 10)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Blocked column GEMM with aligned memory benchmark
static void blocked_column_fwdbwd_aligned_gemm_bench(benchmark::State &s) {
  // Number Dimensions of our matrix
  std::size_t N = (1 << s.range(0)) + 16;

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
    blocked_column_fwdbwd_gemm(A, B, C, N);
  }

  // Free memory
  free(A);
  free(B);
  free(C);
}
BENCHMARK(blocked_column_fwdbwd_aligned_gemm_bench)
    ->DenseRange(8, 10)
    ->Unit(benchmark::kMillisecond);

// Parallel blocked column GEMM benchmark
static void parallel_blocked_column_fwdbwd_atomic_gemm_bench(
    benchmark::State &s) {
  // Number Dimensions of our matrix
  std::size_t N = (1 << s.range(0)) + 16;

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
          [&] { blocked_column_parallel_atomic_gemm(A, B, C, N, pos); });
    }
    blocked_column_fwdbwd_parallel_atomic_gemm(A, B, C, N, pos);

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
BENCHMARK(parallel_blocked_column_fwdbwd_atomic_gemm_bench)
    ->DenseRange(8, 10)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Parallel blocked column GEMM benchmark
static void parallel_blocked_column_gemm_bench(benchmark::State &s) {
  // Number Dimensions of our matrix
  std::size_t N = (1 << s.range(0)) + 16;

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

  // Calculate values to pass to threads
  // Assumed to be divisable by num_threads (evenly)
  std::size_t n_cols = s.range(0) / num_threads;

  // Main benchmark loop
  for (auto _ : s) {
    // Launch threads
    std::size_t end_col;
    for (std::size_t i = 0; i < num_threads; i++) {
      auto start_col = i * n_cols;
      end_col = start_col + n_cols;
      threads.emplace_back([&] {
        blocked_column_parallel_gemm(A, B, C, N, start_col, end_col);
      });
    }

    // Wait for all threads to complete
    for (auto &t : threads) t.join();

    blocked_column_parallel_gemm(A, B, C, N, end_col, end_col + 16);

    // Clear the threads each iteration of the benchmark
    threads.clear();
  }

  // Free memory
  free(A);
  free(B);
  free(C);
}
BENCHMARK(parallel_blocked_column_gemm_bench)
    ->DenseRange(8, 10)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();
