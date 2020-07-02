// Implementation of serial GEMM function

#include <cstddef>

// Serial implementation
void serial_transpose_gemm(const double *A, const double *B, double *C,
                           std::size_t N) {
  // For each row...
  for (std::size_t row = 0; row < N; row++)
    // For each col...
    for (std::size_t col = 0; col < N; col++)
      // For each element in the row/col pair...
      for (std::size_t idx = 0; idx < N; idx++)
        // Accumulate the partial results
        C[row * N + col] += A[row * N + idx] * B[col * N + idx];
}

