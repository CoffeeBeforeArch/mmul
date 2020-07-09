# Matrix Multiplication (GEMM) Beenchmarks

This repository contains a number of serial and parallel benchmarks for matrix multiplication in C++. Matrix multiplication is a wonderful first operation to try your hand at optimizing for the following reasons:

- It is a very common operation in popular fields (e.g., ML)
- The optimizations are fairly easy to understand (they primarily deal with simple access patterns)
- The optimizations are composable (they work better together!)
- It is fairly easy to parallelize

And many more!

The benchmarks in this repository were written using Google Benchmark. For simplicity, all benchmarks assume square matrix of dimension N x N, where N is 2^8 + 16, 2^9 + 16, and 2^10 + 16. The exception to this is a motivating example for not using a power-of-two dimension (cache associativity), which uses an input dimension of 2^8, 2^9, and 2^10.

## Benchmarks

The following section breaks down the benchmarks contained in each subdirectory.

### Baseline

- `serial_gemm_bench_power_two`
  - A motivating example for not using a power-of-two input dimension
- `serial_gemm_bench`
  - Baseline serial gemm implementation (using the classical triply-nested for loop)
- `parallel_gemm_bench`
  - Baseline parallel gemm implementation (splits rows of output matrix across threads)

### Blocked
- `blocked_gemm_bench`
  - A serial gemm implementation which processes a block of elements at a time to exploit locality in the B matrix
- `blocked_aligned_gemm_bench`
  - Same as `blocked_gemm_bench` but using 64-byte aligned allocations to prevent blocks from spanning cache lines
- `parallel_blocked_gemm_bench`
  - A parallel blocked gemm implementation (splits rows of output matrix across threads)

### Blocked Column
- `blocked_column_aligned_gemm_bench`
  - A serial gemm implementation which processes a block of elements at a time, but traverses output blocks of elements in column-major order to exploit locality in the columns of B between blocks of output elements
- `parallel_blocked_column_atomic_gemm_bench`
  - A parallel blocked column implementation (splits columns between threads) where each thread uses an atomic fetch+add to get a new chunk of columns to process

### Blocked Column Multi Output
- `blocked_column_multi_output_aligned_gemm_bench`
  - A serial gemm implementation which processes a tile of output elements at a time, exploiting locality from each tile of B across output tile elements
- `parallel_blocked_column_multi_output_atomic_gemm_bench`
  - A parallel blocked column multi output implementation (splits columns between threads) where each thread uses an atomic fetch+add to get a new chunk of columns to process

## Contact Information

- Email: CoffeeBeforeArch@gmail.com
- YouTube: [Link](https://www.youtube.com/channel/UCsi5-meDM5Q5NE93n_Ya7GA)
- Twitter: [Link](https://twitter.com/AcceleratorNick)

