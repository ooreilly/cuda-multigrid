all: gauss_seidel

gauss_seidel:
	nvcc gauss_seidel.cu --ptxas-options=-v -arch=sm_75 -lineinfo -o gauss_seidel.x --use_fast_math
clean:
	rm ./gauss_seidel.x
