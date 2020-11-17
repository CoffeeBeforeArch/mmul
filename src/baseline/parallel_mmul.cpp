// Implementation of parallel MMul function

#include <cstddef>

// Parallel implementation
void parallel_mmul(const float *A, const float *B, float *C, std::size_t N,
                   std::size_t start_row, std::size_t end_row) {
  // For each row assigned to this thread...
  for (std::size_t row = start_row; row < end_row; row++)
    // For each column...
    for (std::size_t col = 0; col < N; col++)
      // For each element in the row-col pair...
      for (std::size_t idx = 0; idx < N; idx++)
        // Accumulate the partial results
        C[row * N + col] += A[row * N + idx] * B[idx * N + col];
}
