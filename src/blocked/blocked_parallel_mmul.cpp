// Implementation of blocked parallel MMul function

#include <cstddef>

// Blocked parallel implementation
void blocked_parallel_mmul(const float *A, const float *B, float *C,
                           std::size_t N, std::size_t start_row,
                           std::size_t end_row) {
  // For each row...
  for (std::size_t row = start_row; row < end_row; row++)
    // For each block in the row...
    // Solve for 16 elements at a time
    for (std::size_t block = 0; block < N; block += 16)
      // For each chunk of A/B for this block
      for (std::size_t chunk = 0; chunk < N; chunk += 16)
        // For each row in the chunk
        for (std::size_t sub_chunk = 0; sub_chunk < 16; sub_chunk++)
          // Go through all the elements in the sub chunk
          for (std::size_t idx = 0; idx < 16; idx++)
            C[row * N + block + idx] +=
                A[row * N + chunk + sub_chunk] *
                B[chunk * N + sub_chunk * N + block + idx];
}

