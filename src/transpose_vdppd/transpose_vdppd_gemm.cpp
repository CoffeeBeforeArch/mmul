// Implementation of serial GEMM function

#include <immintrin.h>
#include <cstddef>
#include <cstring>

// Hand-vectorized dot product
double dot_product(const double *__restrict v1, const double *v2,
                   const size_t N) {
  auto tmp = 0.0;
  for (size_t i = 0; i < N; i += 2) {
    // Temporary variables to help with intrinsic
    double r[2];
    __m128d rv;

    // Our dot product intrinsic
    rv = _mm_dp_pd(_mm_load_pd(v1 + i), _mm_load_pd(v2 + i), 0xf1);

    // Avoid type punning using memcpy
    memcpy(r, &rv, sizeof(double) * 2);

    tmp += r[0];
  }
  return tmp;
}

// Serial implementation
void serial_transpose_vdppd_gemm(const double *A, const double *B, double *C,
                                 std::size_t N) {
  // For each row...
  for (std::size_t row = 0; row < N; row++)
    // For each col...
    for (std::size_t col = 0; col < N; col++)
      // For each element in the row/col pair...
      // Accumulate the partial results
      C[row * N + col] += dot_product(A, B, N);
}

