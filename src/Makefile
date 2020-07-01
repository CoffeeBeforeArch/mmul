# A simple makefile for GEMM benchmarks

CXX_FLAGS = -O3 -march=native -mtune=native -flto -fuse-linker-plugin --std=c++2a
LINKER_FLAGS = -lbenchmark -lpthread

benchmark : serial_gemm.o blocked_gemm.o parallel_gemm.o blocked_parallel_gemm.o blocked_column_gemm.o blocked_column_parallel_gemm.o blocked_column_parallel_atomic_gemm.o blocked_column_fwdbwd_gemm.o blocked_column_fwdbwd_parallel_gemm.o blocked_column_multi_output_gemm.o blocked_column_multi_output_parallel_atomic_gemm.o transpose_gemm.o parallel_transpose_gemm.o
	g++ benchmark.cpp serial_gemm.o blocked_gemm.o parallel_gemm.o blocked_parallel_gemm.o blocked_column_gemm.o blocked_column_parallel_gemm.o blocked_column_parallel_atomic_gemm.o blocked_column_fwdbwd_gemm.o blocked_column_fwdbwd_parallel_gemm.o blocked_column_multi_output_gemm.o blocked_column_multi_output_parallel_atomic_gemm.o transpose_gemm.o parallel_transpose_gemm.o ${CXX_FLAGS} ${LINKER_FLAGS} -o gemm_bench

parallel_transpose_gemm.o : parallel_transpose_gemm.cpp
	g++ -c parallel_transpose_gemm.cpp ${CXX_FLAGS}

transpose_gemm.o : transpose_gemm.cpp
	g++ -c transpose_gemm.cpp ${CXX_FLAGS}

blocked_column_multi_output_parallel_atomic_gemm.o : blocked_column_multi_output_parallel_atomic_gemm.cpp
	g++ -c blocked_column_multi_output_parallel_atomic_gemm.cpp ${CXX_FLAGS}

blocked_column_multi_output_gemm.o : blocked_column_multi_output_gemm.cpp
	g++ -c blocked_column_multi_output_gemm.cpp ${CXX_FLAGS}

blocked_column_fwdbwd_parallel_gemm.o : blocked_column_fwdbwd_parallel_gemm.cpp
	g++ -c blocked_column_fwdbwd_parallel_gemm.cpp ${CXX_FLAGS}

blocked_column_fwdbwd_gemm.o : blocked_column_fwdbwd_gemm.cpp
	g++ -c blocked_column_fwdbwd_gemm.cpp ${CXX_FLAGS}

blocked_column_parallel_atomic_gemm.o : blocked_column_parallel_atomic_gemm.cpp
	g++ -c blocked_column_parallel_atomic_gemm.cpp ${CXX_FLAGS}

blocked_column_parallel_gemm.o : blocked_column_parallel_gemm.cpp
	g++ -c blocked_column_parallel_gemm.cpp ${CXX_FLAGS}

blocked_column_gemm.o : blocked_column_gemm.cpp
	g++ -c blocked_column_gemm.cpp ${CXX_FLAGS}

blocked_parallel_gemm.o : blocked_parallel_gemm.cpp
	g++ -c blocked_parallel_gemm.cpp ${CXX_FLAGS}

parallel_gemm.o : parallel_gemm.cpp
	g++ -c parallel_gemm.cpp ${CXX_FLAGS}

blocked_gemm.o : blocked_gemm.cpp
	g++ -c blocked_gemm.cpp ${CXX_FLAGS}

serial_gemm.o : serial_gemm.cpp
	g++ -c serial_gemm.cpp ${CXX_FLAGS}

clean :
	rm -rf gemm_bench
	rm -rf *.o

