gauss_seidel: 

%:
	nvcc $@.cu -O3 --ptxas-options=-v -arch=sm_75 -g -Xcompiler -fopenmp -lineinfo -o $@.x --use_fast_math -std=c++11

clean:
	rm ./gauss_seidel.x
