# A simple makefile for GEMM benchmarks

CXX_FLAGS = -O3 -march=native -mtune=native -flto -fuse-linker-plugin --std=c++2a
LINKER_FLAGS = -lbenchmark -lpthread

benchmark : transpose_gemm.o parallel_transpose_gemm.o
	g++ benchmark.cpp transpose_gemm.o parallel_transpose_gemm.o ${CXX_FLAGS} ${LINKER_FLAGS} -o gemm_bench

parallel_transpose_gemm.o : parallel_transpose_gemm.cpp
	g++ -c parallel_transpose_gemm.cpp ${CXX_FLAGS}

transpose_gemm.o : transpose_gemm.cpp
	g++ -c transpose_gemm.cpp ${CXX_FLAGS}

clean :
	rm -rf gemm_bench
	rm -rf *.o

