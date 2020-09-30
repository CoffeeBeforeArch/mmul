# A simple makefile for MMul benchmarks

CXX_FLAGS = -O3 -march=native -mtune=native -flto -fuse-linker-plugin --std=c++2a
LINKER_FLAGS = -lbenchmark -lpthread -lmkl_rt

benchmark : mkl_dgemm.cpp 
	g++ mkl_dgemm.cpp ${CXX_FLAGS} ${LINKER_FLAGS} -o mmul_bench

clean :
	rm -rf mmul_bench
	rm -rf *.o

