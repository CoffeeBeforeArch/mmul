# Matrix Multiplication (MMul) Benchmarks

This repository contains a number of serial and parallel benchmarks for matrix multiplication in C++. Matrix multiplication is a wonderful first operation to try your hand at optimizing for the following reasons:

- It is a very common operation in popular fields (e.g., ML)
- The optimizations are fairly easy to understand (they primarily deal with simple access patterns)
- The optimizations are composable (they work better together!)
- It is fairly easy to parallelize

And many more!

The benchmarks in this repository were written using Google Benchmark. For simplicity, all benchmarks assume square matrix of dimension N x N, where N is 384, 768, and 1152. 

## Benchmarks

The following section breaks down the benchmarks contained in each subdirectory.

### Baseline

- `serial_mmul_bench`
  - Baseline serial mmul implementation (using the classical triply-nested for loop)
- `parallel_mmul_bench`
  - Baseline parallel mmul implementation (splits rows of output matrix across threads)

### Blocked
- `blocked_mmul_bench`
  - A serial mmul implementation which processes a block of elements at a time to exploit locality in the B matrix
- `blocked_aligned_mmul_bench`
  - Same as `blocked_mmul_bench` but using 64-byte aligned allocations to prevent blocks from spanning cache lines
- `parallel_blocked_mmul_bench`
  - A parallel blocked mmul implementation (splits rows of output matrix across threads)

### Blocked Column
- `blocked_column_aligned_mmul_bench`
  - A serial mmul implementation which processes a block of elements at a time, but traverses output blocks of elements in column-major order to exploit locality in the columns of B between blocks of output elements
- `parallel_blocked_column_mmul_bench`
  - A parallel blocked column implementation (splits columns between threads) where work is statically mapped

### Blocked Column Multi Output
- `blocked_column_multi_output_aligned_mmul_bench`
  - A serial mmul implementation which processes a tile of output elements at a time, exploiting locality from each tile of B across output tile elements
- `parallel_blocked_column_multi_output_mmul_bench`
  - A parallel blocked column multi output implementation (splits columns between threads) where work is statically mapped

### GPU Implementations (CUDA)
- `baseline_cuda_mmul`
  - A naive mmul implementation for NVIDIA GPUs written in CUDA
- `shmem_cuda_mmul`
  - A cache-tiled mmul implementation for NVIDIA GPUs using shared memory

## Contact Information

- Email: CoffeeBeforeArch@gmail.com
- YouTube: [Link](https://www.youtube.com/channel/UCsi5-meDM5Q5NE93n_Ya7GA)
- Twitter: [Link](https://twitter.com/AcceleratorNick)

