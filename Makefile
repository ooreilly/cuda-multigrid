gauss_seidel: 

%:
	nvcc $@.cu --ptxas-options=-v -arch=sm_75 -g -lineinfo -o $@.x --use_fast_math -std=c++11

clean:
	rm ./gauss_seidel.x
