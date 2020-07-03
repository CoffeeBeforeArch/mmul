// Implementation of blocked column multi-output serial GEMM function

#include <cstddef>

// Blocked column multi-output serial implementation
void blocked_column_multi_output_gemm(const double *A, const double *B,
                                      double *C, std::size_t N) {
  // For each chunk of columns
  for (std::size_t col_chunk = 0; col_chunk < N; col_chunk += 16)
    // For each chunk of rows
    for (std::size_t row_chunk = 0; row_chunk < N; row_chunk += 16)
      // For each block of elements in this row of this column chunk
      // Solve for 16 elements at a time
      for (std::size_t tile = 0; tile < N; tile += 16)
        // Apply that tile to each row of the row chunk
        for (std::size_t row = 0; row < 16; row++)
          // For each row in the tile
          for (std::size_t tile_row = 0; tile_row < 16; tile_row++)
            // Solve for each element in this tile row
            for (std::size_t idx = 0; idx < 16; idx++)
              C[(row + row_chunk) * N + col_chunk + idx] +=
                  A[(row + row_chunk) * N + tile + tile_row] *
                  B[tile * N + tile_row * N + col_chunk + idx];
}

