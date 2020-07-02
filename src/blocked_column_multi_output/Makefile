# A simple makefile for GEMM benchmarks

CXX_FLAGS = -O3 -march=native -mtune=native -flto -fuse-linker-plugin --std=c++2a
LINKER_FLAGS = -lbenchmark -lpthread

benchmark : blocked_column_multi_output_gemm.o blocked_column_multi_output_parallel_atomic_gemm.o 
	g++ benchmark.cpp blocked_column_multi_output_gemm.o blocked_column_multi_output_parallel_atomic_gemm.o ${CXX_FLAGS} ${LINKER_FLAGS} -o gemm_bench

blocked_column_multi_output_parallel_atomic_gemm.o : blocked_column_multi_output_parallel_atomic_gemm.cpp
	g++ -c blocked_column_multi_output_parallel_atomic_gemm.cpp ${CXX_FLAGS}

blocked_column_multi_output_gemm.o : blocked_column_multi_output_gemm.cpp
	g++ -c blocked_column_multi_output_gemm.cpp ${CXX_FLAGS}

clean :
	rm -rf gemm_bench
	rm -rf *.o

