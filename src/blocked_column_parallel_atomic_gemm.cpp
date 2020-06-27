// Implementation of blocked column parallel GEMM function

#include <atomic>

// Blocked serial implementation
void blocked_column_parallel_atomic_gemm(const double *A, const double *B,
                                         double *C, std::size_t N,
                                         std::atomic<uint64_t> &pos) {
  for (auto col_chunk = pos.fetch_add(16); col_chunk < N;
       col_chunk = pos.fetch_add(16))
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

