// Implementation of blocked partial column serial GEMM function

#include <cstddef>

// Blocked partial column serial implementation
void blocked_partial_column_gemm(const double *A, const double *B, double *C,
                                std::size_t N) {
  // For each chunk for columns...
  for (std::size_t col_chunk = 0; col_chunk < N; col_chunk += 16)
    // For each chunk of rows...
    for (std::size_t row_chunk = 0; row_chunk < N; row_chunk += 16)
      // For each row in this chunk of rows
      for (std::size_t row = 0; row < 16; row++)
        // For each tile of the A/B matrix...
        for (std::size_t tile = 0; tile < N; tile += 16)
          // For each part of these tiles...
          for (std::size_t tile_row = 0; tile_row < 16; tile_row++)
            // Accumulate the partial results
            for (std::size_t idx = 0; idx < 16; idx++)
              C[row_chunk * N + row * N + col_chunk + idx] +=
                  A[row_chunk * N + row * N + tile + tile_row] *
                  B[tile * N + tile_row * N + col_chunk + idx];
}

