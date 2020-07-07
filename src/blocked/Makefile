# A simple makefile for GEMM benchmarks

CXX_FLAGS = -O3 -march=native -mtune=native -flto -fuse-linker-plugin --std=c++2a
LINKER_FLAGS = -lbenchmark -lpthread

benchmark : blocked_gemm.o blocked_parallel_gemm.o
	g++ benchmark.cpp blocked_gemm.o blocked_parallel_gemm.o ${CXX_FLAGS} ${LINKER_FLAGS} -o gemm_bench

blocked_parallel_gemm.o : blocked_parallel_gemm.cpp
	g++ -c blocked_parallel_gemm.cpp ${CXX_FLAGS}

blocked_gemm.o : blocked_gemm.cpp
	g++ -c blocked_gemm.cpp ${CXX_FLAGS}

clean :
	rm -rf gemm_bench
	rm -rf *.o

