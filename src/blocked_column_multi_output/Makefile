# A simple makefile for MMul benchmarks

CXX_FLAGS = -O3 -march=native -mtune=native -flto -fuse-linker-plugin --std=c++2a
LINKER_FLAGS = -lbenchmark -lpthread

benchmark : blocked_column_multi_output_mmul.o blocked_column_multi_output_parallel_mmul.o 
	g++ benchmark.cpp blocked_column_multi_output_mmul.o blocked_column_multi_output_parallel_mmul.o ${CXX_FLAGS} ${LINKER_FLAGS} -o mmul_bench

blocked_column_multi_output_parallel_mmul.o : blocked_column_multi_output_parallel_mmul.cpp
	g++ -c blocked_column_multi_output_parallel_mmul.cpp ${CXX_FLAGS}

blocked_column_multi_output_mmul.o : blocked_column_multi_output_mmul.cpp
	g++ -c blocked_column_multi_output_mmul.cpp ${CXX_FLAGS}

clean :
	rm -rf mmul_bench
	rm -rf *.o

