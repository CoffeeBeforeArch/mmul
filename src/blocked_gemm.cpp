// Implementation of blocked serial GEMM function

#include <vector>

// Blocked serial implementation
void blocked_gemm(const double *A, const double *B, double *C, std::size_t N) {
  // For each row...
  for (std::size_t row = 0; row < N; row++) {
    // For each block in the row...
    // Solve for 8 elements at a time (64 bytes)
    for (std::size_t block = 0; block < N; block += 16) {
      // For each chunk of A/B for this block
      for (std::size_t chunk = 0; chunk < N; chunk += 16) {
        // For each row in the chunk
        for (std::size_t sub_chunk = 0; sub_chunk < 16; sub_chunk++) {
          // Go through all the elements in the sub chunk
          for (std::size_t idx = 0; idx < 16; idx++) {
            C[row * N + block + idx] +=
                A[row * N + chunk + sub_chunk] *
                B[chunk * N + sub_chunk * N + block + idx];
          }
        }
      }
    }
  }
}

