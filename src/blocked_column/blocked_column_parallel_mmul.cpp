// Implementation of blocked column parallel MMul function

#include <atomic>

// Blocked serial implementation
void blocked_column_parallel_mmul(const double *A, const double *B, double *C,
                                  std::size_t N, std::size_t start_col,
                                  std::size_t end_col) {
  for (auto col_chunk = start_col; col_chunk < end_col; col_chunk += 16)
    // For each row in that chunk of columns...
    for (std::size_t row = 0; row < N; row++)
      // For each block of elements in this row of this column chunk
      // Solve for 16 elements at a time
      for (std::size_t tile = 0; tile < N; tile += 16)
        // For each row in the tile
        for (std::size_t tile_row = 0; tile_row < 16; tile_row++)
          // Solve for each element in this tile row
          for (std::size_t idx = 0; idx < 16; idx++)
            C[row * N + col_chunk + idx] +=
                A[row * N + tile + tile_row] *
                B[tile * N + tile_row * N + col_chunk + idx];
}

