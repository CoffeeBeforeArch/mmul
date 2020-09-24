# A simple makefile for GEMM benchmarks

CXX_FLAGS = -O3 -march=native -mtune=native -flto -fuse-linker-plugin --std=c++2a
LINKER_FLAGS = -lbenchmark -lpthread

benchmark : serial_mmul.o parallel_mmul.o 
	g++ benchmark.cpp serial_mmul.o parallel_mmul.o ${CXX_FLAGS} ${LINKER_FLAGS} -o mmul_bench

parallel_mmul.o : parallel_mmul.cpp
	g++ -c parallel_mmul.cpp ${CXX_FLAGS}

serial_mmul.o : serial_mmul.cpp
	g++ -c serial_mmul.cpp ${CXX_FLAGS}

clean :
	rm -rf mmul_bench
	rm -rf *.o

