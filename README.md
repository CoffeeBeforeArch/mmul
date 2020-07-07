# Matrix Multiplication (GEMM) Beenchmarks

This repository contains a number of serial and parallel benchmarks for matrix multiplication in C++. Matrix multiplication is a wonderful first operation to try your hand at optimizing for the following reasons:

- It is a very common operation in popular fields (e.g., ML)
- The optimizations are fairly easy to understand (they primarily deal with simple access patterns)
- The optimizations are composable (they work better together!)
- It is fairly easy to parallelize

And many more!

The benchmarks in this repository were written using Google Benchmark. For simplicity, all benchmarks assume square matrix of dimension N x N, where N is 2^8 + 16, 2^9 + 16, and 2^10 + 16. The exception to this is a motivating example for not using a power-of-two dimension (cache associativity), which uses an input dimension of 2^8, 2^9, and 2^10.

## Contact Information

- Email: CoffeeBeforeArch@gmail.com
- YouTube: [Link](https://www.youtube.com/channel/UCsi5-meDM5Q5NE93n_Ya7GA)
- Twitter: [Link](https://twitter.com/AcceleratorNick)

