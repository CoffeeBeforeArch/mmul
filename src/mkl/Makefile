# A simple makefile for MMul benchmarks

CXX_FLAGS = -O3 -march=native -mtune=native -flto -fuse-linker-plugin --std=c++2a
LINKER_FLAGS = -lbenchmark -lpthread -lmkl_rt

benchmark : mkl_sgemm.cpp 
	g++ mkl_sgemm.cpp ${CXX_FLAGS} ${LINKER_FLAGS} -o mmul_bench

clean :
	rm -rf mmul_bench
	rm -rf *.o

